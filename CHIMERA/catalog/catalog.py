import equinox as eqx
from typing import Optional, Dict
from numbers import Number
import numpy as np
from plum import dispatch
from ..utils.config import jax, jnp, logger
from ..utils.math import trapz
from ..utils import angles
from ..utils.io import load_set, save_set
from ..population.cosmo import dVcdz_at_z
#from .completeness import mask_completeness
from ..data import theta_pe_det, load_galaxy_catalog, theta_src
import h5py

#####################################
# EMPTY CATALOG FOR SPECTRAL SIRENS #
#####################################

class empty_catalog(object):
  """
  A class to handle the computation of the probability of having a galaxy at z
  in the case of an empty galaxy catalog.

  Args:
    p_bkg (eqx.Module): type of bkg
    completeness (object): A CHIMERA.completeness handling the completeness computation.

  Methods:
    - precompute_p_cat: Precompute the p_cat term and its completeness P_compl
  """
  def __init__(self, p_bkg = "dVdz"):
    self.p_cat = 0.
    self.N_gal = 0.
    self.P_compl = 0.
    if p_bkg == "dVdz":
      self.p_bkg = dVcdz_at_z
    else:
      self.p_bkg = p_bkg

  @dispatch
  def p_gal(self, cosmo_lambdas:eqx.Module, z:jnp.ndarray):
    """Compute the `p_gal` term over the given redshift grids."""
    return self.p_bkg(cosmo_lambdas, z)



############################
# PIXELATED GALAXY CATALOG #
############################

class pixelated_catalog(object):
  r"""A class to handle the computation of the probability of having a galaxy at z
  in the case of a pixelated galaxy catalog.

  Args:
    fname_data_gal (str): Filename of the `.h5` file containing galaxy properties (z, RA, DEC).
    data_gw_pixelated (Dict[str, jnp.ndarray]): Dictionary containing a pixelated GW catalog. Should be the output of
        `CHIMERA.data.load_pixelated_gw_catalog()`.
    cosmo (eqx.Module): A CHIMERA.cosmo object describing cosmological parameters.
    completeness (object): A CHIMERA.completeness handling the completeness computation.
    fname_interp (str): Name of the `.pkl` file containing the catalog interpolator to load (or save).
    z_err (Number): Redshift error to assign to each galaxy to be multiplied by (1+z).
    weights (jnp.ndarray): Host galaxy weights.

  Class Attributes:
    - fname_data_gal (str): Filename of the `.h5` file containing galaxy properties (z, RA, DEC).
    - data_gw_pixelated (Dict[str, jnp.ndarray]): Dictionary containing a pixelated GW catalog.
    - cosmo (eqx.Module): A CHIMERA.cosmo object describing cosmological parameters.
    - completeness (object): A CHIMERA.completeness handling the completeness computation.
    - fname_interp (str): Name of the `.pkl` file containing the catalog interpolator.
    - z_err (Number): Redshift error to assign to each galaxy to be multiplied by (1+z).
    - weights (jnp.ndarray): Host galaxy weights.

  Likelihood functions:
    - precompute_p_cat: Precompute the p_cat term and its completeness P_compl

  """
  def __init__(self,
               completeness: object,
               gal_cat_file: Optional[str] = None,
               cosmo: Optional[eqx.Module] = None,
               z_grids: Optional[jnp.ndarray] = None,
               fname_data_gal: Optional[str] = None,
               data_gw_pixelated: Optional[theta_pe_det] = None,
               z_err: Number = 1,
               weights: Optional[jnp.ndarray] = None,
               mask_gal = None,
               sumgauss: str = "dVdz",
               reshuffle: bool = False,
               out_file: Optional[str] = None
               ):

    self.completeness = completeness
    self.p_bkg = self.completeness.p_bkg
    self.fR = self.completeness.fR
    self.attr_gal_cat = ['max_npixels', 'neff_pixels']
    self.data_gal_cat = ['p_cat', 'N_gal', 'P_compl']
    self.attr_compl = []
    self.data_compl = []

    if gal_cat_file is not None:
      logger.info(f"Loading gal_cat object from {gal_cat_file}")
      load_set(self, gal_cat_file, self.attr_gal_cat, self.data_gal_cat)
    else:
      self.cosmo = cosmo
      self.z_grids = z_grids
      self.fname_data_gal = fname_data_gal
      self.data_gw_pixelated = data_gw_pixelated
      self.z_err = z_err
      self.sumgauss = sumgauss

      # Handle galaxies data -> convert to numpy for "precompute_p_cat" function
      self.data_gal = load_galaxy_catalog(self.fname_data_gal, backend='numpy')
      self.data_gal['w'] = weights if weights is not None else np.ones_like(self.data_gal['z'])
      self.data_gal['z_err'] = self.z_err*(1. + self.data_gal['z'])
      self.nevents = len(self.data_gw_pixelated.dL)
      self.max_npixels = self.data_gw_pixelated.pixels_opt_nsides.shape[1]
      self.neff_pixels = jnp.array([self.data_gw_pixelated.ra_pix[ev][self.data_gw_pixelated.ra_pix[ev]!=-100.].shape[0] for ev in range(self.nevents)])

      if mask_gal is not None:
        logger.info("Applying mask to galaxy catalog")
        mask_gal = np.asarray(mask_gal)
        self.data_gal = {k: v[mask_gal] for k, v in self.data_gal.items()}

      if reshuffle:
        logger.info(f"Perturbing galaxy redshift around true value with sigma=(1+z)*{self.z_err}")
        self.data_gal['z'] = np.random.normal(self.data_gal['z'], self.data_gal['z_err'])

      # Precompute galaxy pixels for all possible nsides
      for ns in np.unique(self.data_gw_pixelated.opt_nsides):
        pixn = f"pix{ns}"
        if pixn not in self.data_gal:
          logger.info(f"Computing catalog pixel indexes for nside={ns}")
          self.data_gal[pixn] = angles.find_pix_RAdec(self.data_gal['ra'], self.data_gal['dec'], ns)
        else:
          logger.info(f"Catalog pixel indexes for nside={ns} already computed")

      logger.info(f"Computing p_cat ...")
      self.precompute_p_cat(self.z_grids)
      if out_file is not None:
        save_set(self, out_file, self.attr_gal_cat, self.data_gal_cat)

  def _select_galaxies_in_event_voxels(self, nside_event, event_pix_indexes, z_min, z_max):
    pixn = f"pix{nside_event}"
    good_ev_pixels = event_pix_indexes[event_pix_indexes != -100]
    mask_pix = jnp.isin(self.data_gal[pixn], good_ev_pixels)
    selected = {k:v[mask_pix] for k,v in self.data_gal.items()}
    mask_z = (selected['z'] > z_min) & (selected['z'] < z_max)
    selected = {k: v[mask_z] for k, v in selected.items()}
    return selected

  def _compute_p_cat_event(self, nside_event, event_pix_indexes, z_grid):
    pixn = f"pix{nside_event}"
    gal_in_loc_vol = self._select_galaxies_in_event_voxels(nside_event, event_pix_indexes, z_grid[0], z_grid[-1])
    gal_pix  = gal_in_loc_vol[pixn]
    good_pix = event_pix_indexes[event_pix_indexes != -100]
    if self.sumgauss=="dVdz":
      p_cat = np.array([_sum_gaussians_ucv(z_grid,
        gal_in_loc_vol["z"][gal_pix == p],
        gal_in_loc_vol["z_err"][gal_pix == p],
        self.cosmo,
        weights=gal_in_loc_vol["w"][gal_pix == p]) for p in good_pix]
      )
    elif self.sumgauss=="pbkg":
      p_cat = np.array([_sum_gaussians_pbkg(z_grid,
        gal_in_loc_vol["z"][gal_pix == p],
        gal_in_loc_vol["z_err"][gal_pix == p],
        self.cosmo,
        self.p_bkg,
        weights=gal_in_loc_vol["w"][gal_pix == p]) for p in good_pix]
      )
    p_cat[~jnp.isfinite(p_cat)] = 0.
    # p_cat has shape (neff_pixel, res_zgrid), padding it to (max_pix_number, res_zgrid)
    if len(good_pix) < self.max_npixels:
      padding = np.full((self.max_npixels - len(good_pix), len(z_grid)), -100.)
      p_cat = np.concatenate([p_cat, padding], axis=0)
    Ngal = np.sum(np.array([len(gal_in_loc_vol["z"][gal_pix == p]) for p in good_pix]))
    return p_cat, Ngal

  def precompute_p_cat(self, zgrids):
    """Store the `p_cat`, `Ngal`, and `P_compl` terms computed over the given redshift grids."""
    # compute everything on numpy
    zgrids_numpy = np.asarray(zgrids)
    nsides_numpy = np.asarray(self.data_gw_pixelated.opt_nsides)
    pixels_numpy = np.asarray(self.data_gw_pixelated.pixels_opt_nsides)
    p_cat, N_gal = zip(*[self._compute_p_cat_event(nsides_numpy[e], pixels_numpy[e], zgrids_numpy[e]) for e in range(self.nevents)])

    # back to jax at the end
    self.p_cat = jnp.asarray(p_cat)
    self.N_gal = jnp.asarray(N_gal)
    #if isinstance(self.completeness, mask_completeness):
    #  logger.info("Setting up the completeness mask indices")
    #  self.completeness.set_gw_mask_idxs(pix_gw=pixels_numpy, nside_gw=nsides_numpy, nest_gw=False)
    #  self.completeness.compute_completeness_interpolants()
    self.P_compl = self.completeness.P_compl(zgrids)[:,jnp.newaxis,:]  # assume no cosmology dependence, shape (Nev, Npix, Nz)

  def p_gal(self, cosmo_lambdas:eqx.Module, z:jnp.ndarray):
    """Compute the background galaxy distribution for selection effects estimation."""
    # Where the magic happens
    fR    = jnp.atleast_3d(self.fR(cosmo_lambdas))
    p_bkg = self.p_bkg(cosmo_lambdas, z)[:,jnp.newaxis,:] # shape (Nev, Npix, Nz)
    p_gal = fR * self.p_cat  +  (1. - self.P_compl) * p_bkg
    return jnp.where(self.p_cat != -100., p_gal, -100.)


#####################
# Useful functions

def _gaussian(x, mu, sigma):
  return np.power(2 * np.pi * (sigma ** 2), -0.5) * np.exp(-0.5 * np.power((x - mu) / sigma, 2.))

def _sum_gaussians_ucv(z_grid, mu, sigma, cosmo_lambdas, weights=None):
  if len(mu) == 0:
    return np.zeros_like(z_grid)
  if weights is None:
    weights = np.ones(len(mu))
  zgrid = z_grid[:, np.newaxis]
  gauss = _gaussian(zgrid, mu, sigma)
  gauss *= np.asarray(dVcdz_at_z(cosmo_lambdas, jnp.asarray(zgrid))) # seems stupid I know
  norm  = trapz(gauss, zgrid, axis=0)
  return np.sum(weights * gauss/norm, axis=1) / np.sum(weights)

def _sum_gaussians_pbkg(z_grid, mu, sigma, cosmo_lambdas, p_bkg, weights=None):
  if len(mu) == 0:
    return np.zeros_like(z_grid)
  if weights is None:
    weights = np.ones(len(mu))
  zgrid = z_grid[:, np.newaxis]
  gauss = _gaussian(zgrid, mu, sigma) * np.asarray(p_bkg(cosmo_lambdas, jnp.asarray(zgrid)))
  norm  = trapz(gauss, zgrid, axis=0)
  return np.sum(weights * gauss/norm, axis=1) / np.sum(weights)
