import equinox as eqx
from typing import Optional, Dict, Tuple
from numbers import Number
import numpy as np
import jax
import jax.numpy as jnp
import h5py

from ..utils.config import logger
from ..utils.math import trapz
from ..utils import angles
from ..population.cosmo import dVcdz_at_z
from .completeness import mask_completeness
from ..data import theta_pe_det, load_galaxy_catalog

__all__ = [
    "empty_catalog",
    "pixelated_catalog",
    "precompute_p_cat",
    "save_p_cat",
    "load_p_cat",
]

#####################################
# EMPTY CATALOG FOR SPECTRAL SIRENS #
#####################################

class empty_catalog(object):
  """A class to handle the computation of the probability of having a galaxy at z
  in the case of an empty galaxy catalog.

  Args:
    p_bkg: Callable ``(cosmo, z) -> jnp.ndarray`` giving the background redshift
        distribution. Pass ``"dVdz"`` (default) to use the comoving volume element.
  """
  def __init__(self, p_bkg="dVdz"):
    self.p_cat   = 0.
    self.N_gal   = 0.
    self.P_compl = 0.
    self.p_bkg   = dVcdz_at_z if p_bkg == "dVdz" else p_bkg

  def p_gal(self, cosmo_lambdas: eqx.Module, z: jnp.ndarray):
    """Compute the ``p_gal`` term over the given redshift grids."""
    return self.p_bkg(cosmo_lambdas, z)


############################
# PIXELATED GALAXY CATALOG #
############################

class pixelated_catalog(object):
  r"""Galaxy catalog for dark-siren cosmology with a pixelated sky localization.

  The class is a thin runtime container: it holds precomputed arrays (``p_cat``,
  ``N_gal``, ``P_compl``) together with the completeness object needed to evaluate
  ``p_bkg`` and ``fR`` at each MCMC step.  All heavy, one-time computation is
  deliberately kept outside this class in the standalone :func:`precompute_p_cat`
  function, which can be called once, saved with :func:`save_p_cat`, and reloaded
  with :func:`load_p_cat`.

  Args:
    completeness (object): A CHIMERA completeness object (e.g. ``mask_completeness``,
        ``dVdz_completeness``).  Must expose ``.p_bkg(cosmo, z, distances=None)``,
        ``.fR(cosmo)``, and ``.P_compl(zgrids)``.
    p_cat (jnp.ndarray): Precomputed galaxy-catalog term, shape ``(Nev, Npix, Nz)``.
        Padded pixels are flagged with ``-100``.
    N_gal (jnp.ndarray): Number of galaxies contributing to each event, shape ``(Nev,)``.
    P_compl (jnp.ndarray): Completeness evaluated on the analysis redshift grids,
        shape ``(Nev, 1, Nz)`` (broadcast-ready over the pixel axis).

  Typical usage::

    # --- one-time setup (run once, then save) ---
    p_cat, N_gal, P_compl = precompute_p_cat(
        completeness, cosmo, z_grids, fname_data_gal, gw_pe_det_pixelated,
        z_err=0.001,
    )
    save_p_cat("galcat.h5", p_cat, N_gal, P_compl)

    # --- every run ---
    p_cat, N_gal, P_compl = load_p_cat("galcat.h5")
    gal_cat = pixelated_catalog(completeness, p_cat, N_gal, P_compl)
  """

  def __init__(self,
               completeness: object,
               p_cat:   jnp.ndarray,
               N_gal:   jnp.ndarray,
               P_compl: jnp.ndarray,
               z_grids: jnp.ndarray 
               ):
    self.completeness = completeness
    self.p_bkg  = self.completeness.p_bkg
    self.fR     = self.completeness.fR
    self.p_cat  = p_cat
    self.N_gal  = N_gal
    self.P_compl = P_compl
    self.z_grids = z_grids

  def p_gal(self,
            cosmo_lambdas: eqx.Module,
            z:  jnp.ndarray,
            dL: Optional[jnp.ndarray] = None,
            ) -> jnp.ndarray:
    """Evaluate the galaxy probability on the analysis redshift grids.

    Args:
      cosmo_lambdas: Cosmological parameters for this MCMC step.
      z (jnp.ndarray): Per-event redshift grids, shape ``(Nev, Nz)``.
      dL (jnp.ndarray, optional): Luminosity-distance grids. Forwarded to
          ``p_bkg``; unused by the built-in completeness models but available
          for custom implementations.

    Returns:
      jnp.ndarray: Shape ``(Nev, Npix, Nz)``.  Padded pixels retain the
          ``-100`` sentinel value.
    """
    fR    = jnp.atleast_3d(self.fR(cosmo_lambdas))               # (Nev, Npix, 1)
    p_bkg = self.p_bkg(cosmo_lambdas, z, dL)[:, jnp.newaxis, :]  # (Nev, 1,    Nz)
    p_gal = fR * self.p_cat + (1. - self.P_compl) * p_bkg
    return jnp.where(self.p_cat != -100., p_gal, -100.)


######################################
# STANDALONE PRECOMPUTATION / I/O    #
######################################

def precompute_p_cat(
    completeness: object,
    cosmo: eqx.Module,
    z_grids: jnp.ndarray,
    fname_data_gal: str,
    data_gw_pixelated: theta_pe_det,
    z_err: Number = 1,
    weights: Optional[jnp.ndarray] = None,
    mask_gal=None,
    sumgauss: str = "dVdz",
    reshuffle: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute the ``p_cat``, ``N_gal``, and ``P_compl`` arrays.

  This is the expensive, one-time step that should be run once and then persisted
  with :func:`save_p_cat`.  The resulting arrays are everything that
  :class:`pixelated_catalog` needs beyond the completeness object itself.

  Args:
    completeness: A CHIMERA completeness object exposing ``.p_bkg`` and ``.P_compl``.
    cosmo (eqx.Module): Cosmology object (used to normalise galaxy redshift kernels).
    z_grids (jnp.ndarray): Per-event redshift grids, shape ``(Nev, Nz)``.
    fname_data_gal (str): Path to the galaxy catalog HDF5 file.
    data_gw_pixelated (theta_pe_det): Pixelated GW data structure.
    z_err (Number): Fractional redshift error; applied as ``z_err * (1 + z)``.
    weights (jnp.ndarray, optional): Per-galaxy weights.
    mask_gal (array-like, optional): Boolean mask to apply to the galaxy catalog.
    sumgauss (str): Galaxy kernel: ``"dVdz"`` (default) or ``"pbkg"``.
    reshuffle (bool): If True, perturb galaxy redshifts by their per-galaxy errors.

  Returns:
    Tuple ``(p_cat, N_gal, P_compl)``:
      - ``p_cat``   – shape ``(Nev, Npix, Nz)``; padded pixels flagged ``-100``.
      - ``N_gal``   – shape ``(Nev,)``.
      - ``P_compl`` – shape ``(Nev, 1, Nz)``.
  """
  # ---- load & prepare galaxy data (numpy throughout for speed) ---------------
  data_gal = load_galaxy_catalog(fname_data_gal, backend='numpy')
  data_gal['w']     = weights if weights is not None else np.ones_like(data_gal['z'])
  data_gal['z_err'] = z_err * (1. + data_gal['z'])

  nevents     = len(data_gw_pixelated.dL)
  max_npixels = data_gw_pixelated.pixels_opt_nsides.shape[1]

  if mask_gal is not None:
    logger.info("Applying mask to galaxy catalog")
    mask_gal = np.asarray(mask_gal)
    data_gal = {k: v[mask_gal] for k, v in data_gal.items()}

  if reshuffle:
    logger.info(f"Perturbing galaxy redshifts with sigma=(1+z)*{z_err}")
    data_gal['z'] = np.random.normal(data_gal['z'], data_gal['z_err'])

  # Precompute pixel indices for every nside present in the GW data
  for ns in np.unique(data_gw_pixelated.opt_nsides):
    pixn = f"pix{ns}"
    if pixn not in data_gal:
      logger.info(f"Computing catalog pixel indexes for nside={ns}")
      data_gal[pixn] = angles.find_pix_RAdec(data_gal['ra'], data_gal['dec'], ns)
    else:
      logger.info(f"Catalog pixel indexes for nside={ns} already present")

  # ---- per-event p_cat computation -------------------------------------------
  p_bkg_fn = completeness.p_bkg  # used only when sumgauss == "pbkg"

  def _select(nside_event, event_pix_indexes, z_min, z_max):
    pixn = f"pix{nside_event}"
    good_pix = event_pix_indexes[event_pix_indexes != -100]
    mask_pix = jnp.isin(data_gal[pixn], good_pix)
    sel = {k: v[mask_pix] for k, v in data_gal.items()}
    mask_z = (sel['z'] > z_min) & (sel['z'] < z_max)
    return {k: v[mask_z] for k, v in sel.items()}

  def _compute_event(nside_event, event_pix_indexes, z_grid):
    pixn = f"pix{nside_event}"
    gal  = _select(nside_event, event_pix_indexes, z_grid[0], z_grid[-1])
    gal_pix  = gal[pixn]
    good_pix = event_pix_indexes[event_pix_indexes != -100]

    if sumgauss == "dVdz":
      p_cat = np.array([_sum_gaussians_ucv(z_grid,
          gal["z"][gal_pix == p], gal["z_err"][gal_pix == p],
          cosmo, weights=gal["w"][gal_pix == p]) for p in good_pix])
    elif sumgauss == "pbkg":
      p_cat = np.array([_sum_gaussians_pbkg(z_grid,
          gal["z"][gal_pix == p], gal["z_err"][gal_pix == p],
          cosmo, p_bkg_fn, weights=gal["w"][gal_pix == p]) for p in good_pix])
    else:
      raise ValueError(f"Unknown sumgauss option '{sumgauss}'. Choose 'dVdz' or 'pbkg'.")

    p_cat[~np.isfinite(p_cat)] = 0.

    if len(good_pix) < max_npixels:
      padding = np.full((max_npixels - len(good_pix), len(z_grid)), -100.)
      p_cat = np.concatenate([p_cat, padding], axis=0)

    N_gal = int(np.sum([len(gal["z"][gal_pix == p]) for p in good_pix]))
    return p_cat, N_gal

  logger.info("Computing p_cat ...")
  zgrids_np = np.asarray(z_grids)
  nsides_np = np.asarray(data_gw_pixelated.opt_nsides)
  pixels_np = np.asarray(data_gw_pixelated.pixels_opt_nsides)

  results = [_compute_event(nsides_np[e], pixels_np[e], zgrids_np[e]) for e in range(nevents)]
  p_cat_list, N_gal_list = zip(*results)

  p_cat   = jnp.asarray(p_cat_list)                                   # (Nev, Npix, Nz)
  N_gal   = jnp.asarray(N_gal_list)                                   # (Nev,)
  P_compl = completeness.P_compl(z_grids)[:, jnp.newaxis, :]          # (Nev, 1, Nz)

  return p_cat, N_gal, P_compl


def save_p_cat(fname: str,
               p_cat:   jnp.ndarray,
               N_gal:   jnp.ndarray,
               P_compl: jnp.ndarray,
               z_grids: jnp.ndarray,
               ) -> None:
  """Save precomputed catalog arrays to an HDF5 file.

  Args:
    fname (str): Output file path.
    p_cat (jnp.ndarray): Shape ``(Nev, Npix, Nz)``.
    N_gal (jnp.ndarray): Shape ``(Nev,)``.
    P_compl (jnp.ndarray): Shape ``(Nev, 1, Nz)``.
  """
  logger.info(f"Saving p_cat to '{fname}'")
  with h5py.File(fname, 'w') as f:
    f.create_dataset('p_cat',   data=np.asarray(p_cat))
    f.create_dataset('N_gal',   data=np.asarray(N_gal))
    f.create_dataset('P_compl', data=np.asarray(P_compl))
    f.create_dataset('z_grids', data=np.asarray(z_grids))
    f.attrs['p_cat_shape']   = p_cat.shape
    f.attrs['N_gal_shape']   = N_gal.shape
    f.attrs['P_compl_shape'] = P_compl.shape


def load_p_cat(fname: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Load precomputed catalog arrays from an HDF5 file.

  Args:
    fname (str): Path to a file previously written by :func:`save_p_cat`.

  Returns:
    Tuple ``(p_cat, N_gal, P_compl)`` as JAX arrays.
  """
  logger.info(f"Loading p_cat from '{fname}'")
  with h5py.File(fname, 'r') as f:
    p_cat   = jnp.array(f['p_cat'][:])
    N_gal   = jnp.array(f['N_gal'][:])
    P_compl = jnp.array(f['P_compl'][:])
    z_grids = jnp.array(f['z_grids'][:])
  return p_cat, N_gal, P_compl, z_grids


####################
# Useful functions #
####################

def _gaussian(x, mu, sigma):
  return np.power(2 * np.pi * sigma**2, -0.5) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def _sum_gaussians_ucv(z_grid, mu, sigma, cosmo_params, weights=None):
  if len(mu) == 0:
    return np.zeros_like(z_grid)
  if weights is None:
    weights = np.ones(len(mu))
  zgrid = z_grid[:, np.newaxis]
  gauss = _gaussian(zgrid, mu, sigma)
  gauss *= np.asarray(dVcdz_at_z(cosmo_params, jnp.asarray(zgrid)))
  norm  = trapz(gauss, zgrid, axis=0)
  return np.sum(weights * gauss / norm, axis=1) / np.sum(weights)


def _sum_gaussians_pbkg(z_grid, mu, sigma, cosmo_params, p_bkg, weights=None):
  if len(mu) == 0:
    return np.zeros_like(z_grid)
  if weights is None:
    weights = np.ones(len(mu))
  zgrid = z_grid[:, np.newaxis]
  gauss = _gaussian(zgrid, mu, sigma) * np.asarray(p_bkg(cosmo_params, jnp.asarray(zgrid)))
  norm  = trapz(gauss, zgrid, axis=0)
  return np.sum(weights * gauss / norm, axis=1) / np.sum(weights)
