import os
import equinox as eqx
from typing import Optional, Dict, List, Union
from numbers import Number
import jax
import jax.numpy as jnp
import numpy as np

from ..utils.math import trapz, cumtrapz, interp
from ..utils.config import logger
from ..utils.angles import find_pix_RAdec, convert_pixelization
from ..utils.io import load_set, save_set
from ..utils.mags import schechter_mags
from ..population.cosmo import dVcdz_at_z, Vc_at_z, dL_at_z, flrw
from scipy.ndimage import gaussian_filter1d
import healpy as hp
from sklearn.cluster import AgglomerativeClustering

import jax
rndkey = jax.random.PRNGKey(0)

####################################################
# (IN)COMPLETENESS FOR EMPTY CATALOG (non pixelated) #
######################################################

class zero_completeness(object):
  """A class to compute the completeness of an empty galaxy catalog."""

  def P_compl(self, zgrids):
    """Computes the completeness probability `P_compl` over the given redshift grid."""
    return jnp.zeros_like(zgrids)

  def fR(self):
    """Computes the completeness normalization factor `f_R`."""
    return 0.

#####################################
# COMPLETE HOMOGENEOUS COMPLETENESS #
#####################################


class dVdz_completeness(object):
  """A class to compute the completeness of the luminosity-complete UCV sub-sample of MICE.

  Args:
    cosmo (CHIMERA.cosmo.flrw or CHIMERA.cosmo.mg_flrw): an instance of a cosmology object containing cosmological parameters.
    z_range (jax.ndarray): a 2-element JAX array specifying the redshift range within which the catalog is complete.
    kind (str): The type of completeness to compute. Options are:
      - `'step'`: Step function completeness.
      - `'step_smooth'`: Smooth step function completeness.
    z_sig (Optional[Number], default=None):
      Sigma parameter controlling the smoothness of the completeness step in the `'step_smooth'` mode.
  """
  def __init__(self,
    z_range: Union[List[Number], jnp.ndarray] = [0.073, 1.3],
    kind: str = "step",
    z_sig: Optional[Number] = None,
  ):
    self.z_range = jnp.asarray(z_range)
    self.kind = kind
    self.z_sig = z_sig

  def P_compl(self, zgrids):
    """Computes the completeness probability `P_compl` over the given redshift grid."""
    if self.kind == "step":
      Pcompl = jnp.where(jnp.logical_and(zgrids>self.z_range[0], zgrids<self.z_range[1]), 1., 0.) # shape (Nev, Nz)
    elif self.kind == "step_smooth":
      t_thr = (self.z_range-zgrids)/self.z_sig
      Pcompl = 0.5*(1+jax.scipy.special.erf(t_thr/jnp.sqrt(2))) # shape (Nev, Nz)
    else:
      raise ValueError("kind must be step or step_smooth")
    return Pcompl

  def fR(self, cosmo_lambdas, normalized=False):
    """Computes the completeness normalization factor `f_R`."""
    res = Vc_at_z(cosmo_lambdas, self.z_range)
    fR = res[1] - res[0] # shape (1,)
    return fR

  def p_bkg(self, cosmo_lambdas:eqx.Module, z:jnp.ndarray, distances=None):
    return dVcdz_at_z(cosmo_lambdas, z, distances = distances)
 
############################
# Homogeneous Completeness #
############################

class homogeneous_completeness(object):
  """A class to compute the completeness of a galaxy catalog as a density fraction w.r.t. a theoretical function.
  # to be tested
  """

  def __init__(self,
    z_gal: jnp.ndarray,
    theory_density_func: object,
    cosmo_obj: eqx.Module,
    sky_area_deg2: Number = 58.0,
    z_range: Optional[jnp.ndarray] = None,
    smooth: Optional[Number] = None,
    Nz_to_bin: Optional[Number] = 50,
    Nz_interp: Optional[Number] = 1000,
    resample: Optional[Number] = None,
    weights: Optional[jnp.ndarray] = None,
  ):

    self.z_gal = z_gal
    self.theory_density_func = theory_density_func
    self.cosmo_obj = cosmo_obj
    self.smooth = smooth
    self.sky_area_deg2 = sky_area_deg2
    self.Nz_to_bin = Nz_to_bin
    self.Nz_interp = Nz_interp
    self.resample = resample
    self.weights = weights
    self.z_range = z_range

    self.sky_area_sr = self.sky_area_deg2 * (jnp.pi / 180.)**2  # Convert to steradians

    self.z_range = jnp.array([jnp.min(self.z_gal), jnp.max(self.z_gal)]) if self.z_range is None else self.z_range
    self.z_grid_bin = jnp.linspace(*self.z_range, self.Nz_to_bin + 1)
    self.z_grid_interp = jnp.linspace(*self.z_range, self.Nz_interp + 1)

  # -----------------------
  # Interpolant operations
  # -----------------------
  def get_completeness_interpolant(self, dir_compl):
    """Get completeness interpolant.
    
    Args:
        dir_compl (str): Path to load/save completeness interpolant.
    
    Returns:
        Function to get completeness for any mask at any redshift
    """
    # Case 1: Skip interpolant operations entirely
    if dir_compl == "skip":
      logger.info(f" > Skipping interpolant computation for self.get_completeness_z")
      return False
  
    # Case 2: Compute interpolants
    if dir_compl is None:
      dir_compl = f"completeness_interpolant_{self.nmasks}x_pix{self.nside}{self.nest}.h5"
      logger.info(f" > Computing mask interpolant on z range {self.z_int_range} with res={self.z_int_res}")
      self.compute_completeness_interpolants()
      logger.info(f" > Saving interpolants to '{dir_compl}'")
      save_set(self, dir_compl, self.attr_compl, self.data_compl)
    
    # Case 3: Load interpolant from specified file 
    else:
      logger.info(f" > Loading interpolant from {dir_compl}")
      load_set(self, dir_compl, self.attr_compl, self.data_compl)
      logger.info(f"   - Loaded {self.nmasks} interpolant with shape={self.completeness.shape}, z_int_range={self.z_int_range}")
    
    # Create a vectorizable interpolation function
    def get_completeness(mask_idx, z):
      return interp(z, self.z_int_grid, self.completeness[mask_idx], left=0, right=0)
    
    # 1. For a single mask index with multiple z values
    self.get_completeness_z = jax.vmap(get_completeness, in_axes=(None, 0))


  def compute_completeness_interpolants(self):
    """ Compute completeness over `self.z_int_grid` for each mask. 
    
    Sets:
      self.completeness (jnp.ndarray): Completeness for each mask.
    """
    
    self.completeness = jnp.zeros((self.nmasks, self.z_int_res))
    self.z_int_grid = jnp.linspace(*self.z_int_range, self.z_int_res)
    
    # Calculate completeness for each mask
    self.completeness = compute_completeness(self.cosmo_obj, self.rho_gal_theo, self.z_int_grid, self.z_gal, 
                                              sky_area = self.sky_area, Nz_to_bin=self.Nz_to_bin, 
                                              weights_gal=self.weights_gal, smooth=self.smooth, resample=self.resample,
                                              mask=self.mask_and_gal)

    # Clip completeness between zmin and zmax
    self.completeness = jnp.where((self.z_int_grid >= self.z_min) & (self.z_int_grid <= self.z_max),
                                  self.completeness,
                                  0.)
    return True

  # -----------------------
  # P_compl, p_bkg, and fR 
  # -----------------------

  def P_compl(self, z_grids):
    """Compute P_compl(z) on analysis z_grids for all the GW events.
    Approx.: P_compl is cosmology independent (true for H0)."""
    
    # Map the processing over all events
    return jax.vmap(lambda z_ev : interp(z_ev, self.z_int_grid, self.completeness))(z_grids)


  def p_bkg(self, cosmo_obj, z, distances=None):
    """Compute background probability at arbitrary redshifts, normalized over the range defined by self.z_int_grid.
    
    Args:
        cosmo_obj: Cosmology object.
        z (jnp.ndarray): Redshift value(s) where to evaluate the function.
        distances (jnp.ndarray, optional): Luminosity distance value(s). Currently unused; reserved for
            future implementations where p_bkg may depend on dL (e.g. selection-weighted backgrounds).
            
    Returns:
        jnp.ndarray: Normalized probability values at z points.
    """
    bkg = jnp.where((self.z_int_grid >= self.z_min) & (self.z_int_grid <= self.z_max),
                    self.rho_gal_theo(self.z_int_grid) * dVcdz_at_z(cosmo_obj, self.z_int_grid),
                    0.)
    
    norm = trapz(bkg, self.z_int_grid)

    return interp(z, self.z_int_grid, bkg/norm, left=0, right=0)


  def fR(self, cosmo_obj):
    """Compute fR = ∫ P_compl * p_bkg dz for all masks with a given cosmology. Called within MCMC when cosmology changes.
    
    Parameters:
        cosmo_obj: Cosmological parameters for this iteration
        pix_gw (jnp.ndarray): Pixelization of the GW event
        nside (jnp.ndarray): HEALPix nside parameter.

        nside_gw : Nside of the GW event
        nest_gw: Whether the GW event is in nested pixelization
        
    Returns:
        Array of shape (nmasks,) with fR values
    """
    p_bkg = self.p_bkg(cosmo_obj, self.z_int_grid)

    return trapz(self.completeness * p_bkg, self.z_int_grid)

#####################
# Mask Completeness #
#####################

class mask_completeness(object):
  """
  Compact implementation of a masked completeness class.
  """
  def __init__(self,
               cosmo_obj: eqx.Module,
               rho_gal_theo: object,

               z_gal: jnp.ndarray = None,
               ra_gal: jnp.ndarray = None,
               dec_gal: jnp.ndarray = None,
               weights_gal: jnp.ndarray = None,
               mask_and_gal: jnp.ndarray = None,

               nmasks: int = 6,
               nside: int = 32,
               nest: bool = False,

               Nz_to_bin: int = 50,
               resample: int = None,
               smooth: float = None,

               z_int_range: list = [0., 10.],
               z_int_res: int = 1000,

               z_min: float = 0.,
               z_max: float = jnp.inf,

               dir_masks: str = None,
               dir_compl: str = None,
              ):

    logger.info(f"Initializing `masked_completeness`:")

    self.attr_mask = ["nmasks", "nside", "nest", "npix", "pixarea", "sky_area_mask"]
    self.data_mask = ["ngal_pix", "healpix2mask", "mask_gal", "npix_mask", "ngal_mask"]
    self.attr_compl = ["nmasks", "nside", "nest", "npix", "pixarea", "sky_area_mask"]
    self.data_compl = ["z_int_grid", "completeness"]
    self.get_completeness_z = RuntimeError("Method not initialized. Call compute_completeness_interpolants() first.")
    self.get_completeness_batch = RuntimeError("Method not initialized. Call compute_completeness_interpolants() first.")
    self.gw_mask_idxs = RuntimeError("Attribute not initialized. Call set_gw_mask_idxs() first.")
    self.gw_valid = RuntimeError("Attribute not initialized. Call set_gw_mask_idxs() first.")

    # Get completeness masks (either loaded or computed)
    self.get_masks(ra_gal, dec_gal, nmasks, nside, nest, dir_masks)

    # Set theoretical quantities
    self.cosmo_obj = cosmo_obj
    self.rho_gal_theo = rho_gal_theo

    # Set observed quantities and related parameters
    self.z_gal = z_gal
    self.weights_gal = weights_gal
    self.mask_and_gal = mask_and_gal

    # Set the completeness parameters
    self.Nz_to_bin = Nz_to_bin
    self.resample = resample
    self.smooth = smooth

    # Set z_max to enforce p_bkg = 0 for self.z_min < z < self.z_max
    self.z_min = z_min
    self.z_max = z_max

    # Define grid to compute the background distribution
    self.z_int_range = z_int_range
    self.z_int_res = z_int_res

    # Get completeness interpolants
    self.dir_compl = dir_compl
    self.get_completeness_interpolants(dir_compl)

  # ----------------
  # Mask operations
  # ----------------
  def get_masks(self, ra_gal=None, dec_gal=None, nmasks=None, nside=None, nest=False, dir_masks=None):
    """Get completeness masks - either load from disk or compute from scratch.
    
    Args:
        ra_gal (jnp.ndarray, optional): Right ascension of galaxies in radians.
        dec_gal (jnp.ndarray, optional): Declination of galaxies in radians.
        nmasks (int, optional): Number of masks to create.
        nside (int, optional): HEALPix nside parameter.
        nest (bool, optional): Whether to use nested pixelization.
        dir_masks (str, optional): Path to load/save masks. If None, compute and save.
                                  If "skip", skip mask operations entirely.
    
    Returns:
        bool: True if masks were successfully obtained
    """
    # Case 1: Skip mask operations entirely
    if dir_masks == "skip":
      logger.info(f" > Skipping mask loading or computation. Must be done manually.")
      return False
    
    # Case 2: No path given -- always compute fresh, save to an auto-generated filename
    if dir_masks is None:
      dir_masks = f"completeness_masks_{nmasks}x_pix{nside}{nest}.h5"
      logger.info(f" > Computing completeness masks (nmasks={nmasks}, nside={nside})")
      self.compute_masks(ra_gal, dec_gal, nmasks, nside, nest)
      logger.info(f" > Saving masks to '{dir_masks}'")
      save_set(self, dir_masks, self.attr_mask, self.data_mask)

    # Case 3: Explicit path given -- load it if it already exists, otherwise
    # compute fresh and save to that same path (so a first run on a machine/
    # config with no cache yet doesn't crash, and later runs reuse the cache).
    elif os.path.exists(dir_masks):
      logger.info(f" > Loading masks from {dir_masks}")
      load_set(self, dir_masks, self.attr_mask, self.data_mask)
      logger.info(f"   - Loaded {self.nmasks} masks with nside={self.nside}, nest={self.nest}")
    else:
      logger.info(f" > No masks found at '{dir_masks}' -- computing (nmasks={nmasks}, nside={nside})")
      self.compute_masks(ra_gal, dec_gal, nmasks, nside, nest)
      logger.info(f" > Saving masks to '{dir_masks}'")
      save_set(self, dir_masks, self.attr_mask, self.data_mask)

    logger.info(f"   - Pixels per mask: {self.npix_mask}")
    logger.info(f"   - Galaxies per mask: {self.ngal_mask}")

    return True

  def compute_masks(self, ra_gal, dec_gal, nmasks, nside, nest=False):
    """Compute completeness masks by clustering pixels based on number of galaxies inside them.

    Clustering: agglomerative (ward-linkage) clustering of ALL pixels (including empty
    ones) on ``log(count + 10)``, matching `CHIMERA_private`'s `MaskCompleteness`
    (`_compute_avg_mask`/`compute`) convention -- the log transform and the "+10" offset
    are that implementation's empirical choices for improving cluster separation, kept
    here verbatim for cross-compatibility between the two codebases. This replaces the
    previous k-means-on-raw-counts approach (`scipy.cluster.vq.kmeans`, non-zero pixels
    only, with mask index 0 explicitly reserved for zero-count pixels).

    NOTE: unlike the old k-means approach, sklearn's cluster labels are NOT ordered by
    density -- mask index 0 is no longer guaranteed to be the empty-sky/zero-count mask.
    Any downstream code that assumed that ordering needs to look up the empty-sky mask
    explicitly (e.g. via `self.ngal_mask` or `self.healpix2mask[self.ngal_pix == 0]`)
    rather than hardcoding index 0.

    Args:
      ra_gal (jnp.ndarray): Right ascension of galaxies in radians.
      dec_gal (jnp.ndarray): Declination of galaxies in radians.
      nmasks (int): Number of masks to create.
      nside (int): HEALPix nside parameter.
      nest (bool): Whether to use nested pixelization.
      mask_and_gal (jnp.ndarray, optional): Boolean array for an additional galaxy mask (& logic will be used).
    Sets:
      self.nmasks (int): Number of masks.
      self.nside (int): HEALPix nside parameter.
      self.nest (bool): Whether to use nested pixelization.
      self.npix (int): Number of pixels.
      self.pixarea (float): Area of each pixel in steradians.
      self.ngal_pix (jnp.ndarray): Number of galaxies per pixel.
      self.healpix2mask (jnp.ndarray): Mapping from pixels to masks.
      self.mask_gal (jnp.ndarray): Mapping from galaxies to masks.
      self.npix_mask (jnp.ndarray): Number of pixels in each mask.
      self.ngal_mask (jnp.ndarray): Number of galaxies in each mask.
      self.sky_area_mask (jnp.ndarray): Sky area of each mask in steradians.
    """

    if any(p is None for p in [ra_gal, dec_gal, nmasks, nside]):
      raise ValueError("Required parameters to compute `masked_completeness` masks: ra_gal, dec_gal, nmasks, nside")

    self.nmasks = nmasks
    self.nside, self.nest, self.npix, self.pixarea = nside, nest, hp.nside2npix(nside), hp.nside2pixarea(nside)

    # Count the number of galaxies per pixel
    pix_gal = find_pix_RAdec(ra_gal, dec_gal, nside=nside, nest=nest)
    uniq, inv = jnp.unique(pix_gal, return_inverse=True)
    self.ngal_pix = jnp.zeros(hp.nside2npix(nside), dtype=jnp.int_).at[uniq].set(jnp.bincount(inv))

    # Cluster ALL pixels (zero-count ones included) via agglomerative (ward-linkage)
    # clustering on log(count + 10) -- see docstring note on CHIMERA_private parity.
    # sklearn is not JAX-traceable, so this runs as a plain numpy preprocessing step
    # (same as the k-means call it replaces).
    ngal_pix_log = np.log(np.asarray(self.ngal_pix, dtype=float) + 10.)
    clusterer = AgglomerativeClustering(n_clusters=nmasks, linkage='ward')
    self.healpix2mask = jnp.asarray(clusterer.fit(ngal_pix_log.reshape(-1, 1)).labels_.astype(int))

    # Assign the mask to each galaxy
    self.mask_gal = self.healpix2mask[pix_gal]
    self.npix_mask = jnp.array([jnp.sum(self.healpix2mask == i) for i in range(nmasks)])
    self.ngal_mask = jnp.array([jnp.sum(self.mask_gal == i) for i in range(nmasks)])
    self.sky_area_mask = self.pixarea * self.npix_mask


  # ------------------------
  # Interpolants operations
  # ------------------------
  def get_completeness_interpolants(self, dir_compl):
    """Get completeness interpolants for all masks.
    
    Args:
        dir_compl (str): Path to load/save completeness interpolants.
    
    Returns:
        Function to get completeness for any mask at any redshift
    """
    # Case 1: Skip interpolant operations entirely
    if dir_compl == "skip":
      logger.info(f" > Skipping interpolant computation for self.get_completeness_z and self.get_completeness_batch")
      return False
    
    # Case 2: No path given -- always compute fresh, save to an auto-generated filename
    if dir_compl is None:
      dir_compl = f"completeness_interpolants_{self.nmasks}x_pix{self.nside}{self.nest}.h5"
      logger.info(f" > Computing mask interpolants on z range {self.z_int_range} with res={self.z_int_res}")
      self.compute_completeness_interpolants()
      logger.info(f" > Saving interpolants to '{dir_compl}'")
      save_set(self, dir_compl, self.attr_compl, self.data_compl)

    # Case 3: Explicit path given -- load it if it already exists, otherwise
    # compute fresh and save to that same path (mirrors get_masks above).
    elif os.path.exists(dir_compl):
      logger.info(f" > Loading interpolants from {dir_compl}")
      load_set(self, dir_compl, self.attr_compl, self.data_compl)
      logger.info(f"   - Loaded {self.nmasks} interpolants with shape={self.completeness.shape}, z_int_range={self.z_int_range}")
    else:
      logger.info(f" > No interpolants found at '{dir_compl}' -- computing on z range {self.z_int_range} with res={self.z_int_res}")
      self.compute_completeness_interpolants()
      logger.info(f" > Saving interpolants to '{dir_compl}'")
      save_set(self, dir_compl, self.attr_compl, self.data_compl)


    # Create a vectorizable interpolation function
    def get_completeness(mask_idx, z):
      return interp(z, self.z_int_grid, self.completeness[mask_idx], left=0, right=0)
    
    # 1. For a single mask index with multiple z values
    self.get_completeness_z = jax.vmap(get_completeness, in_axes=(None, 0))
    
    # 2. For a batch of events with paired mask_idx and z values
    self.get_completeness_batch = jax.vmap(get_completeness, in_axes=(0, 0))

  def compute_completeness_interpolants(self):
    """ Compute completeness over `self.z_int_grid` for each mask. 
    
    Sets:
      self.completeness (jnp.ndarray): Completeness for each mask.
    """
    
    self.completeness = jnp.zeros((self.nmasks, self.z_int_res))
    self.z_int_grid = jnp.linspace(*self.z_int_range, self.z_int_res)
    
    # Calculate completeness for each mask
    for i in range(self.nmasks):
      mask_selector = (self.mask_gal == i)

      if self.mask_and_gal is not None:
        mask_selector &= self.mask_and_gal
      
      self.completeness = self.completeness.at[i].set(
          compute_completeness(self.cosmo_obj, self.rho_gal_theo, self.z_int_grid, self.z_gal, 
                                sky_area = self.sky_area_mask[i], Nz_to_bin=self.Nz_to_bin, 
                                weights_gal=self.weights_gal, smooth=self.smooth, resample=self.resample,
                                mask=mask_selector))
    
    # Clip completeness between zmin and zmax
    self.completeness = jnp.where(
        (self.z_int_grid >= self.z_min) & (self.z_int_grid <= self.z_max),
        self.completeness,
        0.
    )
  
    return True


  # -----------------------
  # P_compl, p_bkg, and fR 
  # -----------------------

  def set_gw_mask_idxs(self, pix_gw, nside_gw, nest_gw):
    """Set the GW mask indices for the given pixelization.
    Convert GW to completeness pixelization and get corresponding mask index.

    NOTE (2026-10-20): `pix_gw` (e.g. `data_gw_pixelated.pixels_opt_nsides`) is
    padded with the project-wide `-100` sentinel for events with fewer than the
    maximum number of credible pixels. `-100` cannot be safely cast into an
    integer HEALPix pixel index by `convert_pixelization` (it calls
    `hp.pix2ang` under the hood), so we substitute a dummy valid pixel (0) for
    padded entries before converting, and separately record which entries are
    valid in `self.gw_valid` (mirroring `mth_completeness.set_gw_mask_idxs`,
    which does the same with NaN since its stored values are floats) so that
    `P_compl` can exclude the padded entries when averaging over an event's
    pixels. `fR` does not need `self.gw_valid`: its padded-pixel entries are
    bogus-but-finite, and are masked out downstream anyway by `p_cat`'s own
    `-100` flag in `pixelated_catalog.p_gal`."""
    pix_gw = np.asarray(pix_gw)
    self.gw_valid = jnp.asarray(pix_gw != -100)
    # `pix_gw` comes out of `data.pixelize_gw_catalog` as a float array (it is
    # padded alongside the `-100.` sentinel), but `convert_pixelization` feeds
    # it straight into `hp.pix2ang`, which requires integer pixel indices --
    # hence the explicit `.astype(int)` below, on top of the `-100` -> `0`
    # substitution.
    pix_safe = np.where(self.gw_valid, pix_gw, 0).astype(int)
    self.gw_mask_idxs = self.healpix2mask[convert_pixelization(pix_safe, nside_gw, self.nside, nest_gw, self.nest)]


  def P_compl(self, z_grids):
    """Compute P_compl(z) on analysis z_grids for all the GW events, averaged
    over each event's valid sky pixels (padded entries, flagged in
    `self.gw_valid`, are excluded -- mirrors `mth_completeness.P_compl`).
    Approx.: P_compl is cosmology independent (true for H0)."""

    # Define a function that interpolates one mask's completeness onto a redshift grid
    def interp_single_mask_to_grid(mask_id, z_grid):
        return interp(z_grid, self.z_int_grid, self.completeness[mask_id])

    # For each event, for each pixel, get completeness at each redshift, then
    # average over the event's valid pixels only (shape (Npix, Nz) -> (Nz,))
    def process_event(event_mask_idxs, event_valid, event_z_grid):
        pc_pix = jax.vmap(lambda m_id: interp_single_mask_to_grid(m_id, event_z_grid))(event_mask_idxs)  # (Npix, Nz)
        valid = event_valid[:, jnp.newaxis]  # (Npix, 1)
        n_valid = jnp.maximum(jnp.sum(valid, axis=0), 1)
        return jnp.sum(jnp.where(valid, pc_pix, 0.), axis=0) / n_valid  # (Nz,)

    # Map the processing over all events
    return jax.vmap(process_event)(self.gw_mask_idxs, self.gw_valid, z_grids)


  def p_bkg(self, cosmo_obj, z, distances=None):
    """Compute background probability at arbitrary redshifts, normalized over the range defined by self.z_int_grid.
    
    Args:
        cosmo_obj: Cosmology object.
        z (jnp.ndarray): Redshift value(s) where to evaluate the function.
        distances (jnp.ndarray, optional): Luminosity distance value(s). Currently unused; reserved for
            future implementations where p_bkg may depend on dL (e.g. selection-weighted backgrounds).
            
    Returns:
        jnp.ndarray: Normalized probability values at z points.
    """
    bkg = jnp.where(
        (self.z_int_grid >= self.z_min) & (self.z_int_grid <= self.z_max),
        self.rho_gal_theo(self.z_int_grid) * dVcdz_at_z(cosmo_obj, self.z_int_grid),
        0.
    )
    norm = trapz(bkg, self.z_int_grid)
    return interp(z, self.z_int_grid, bkg/norm, left=0, right=0)
  
  def fR(self, cosmo_obj):
    """Compute fR = ∫ P_compl * p_bkg dz for all masks with a given cosmology.
    This function is called within MCMC iterations when cosmology changes.
    
    Parameters:
        cosmo_obj: Cosmological parameters for this iteration
        pix_gw (jnp.ndarray): Pixelization of the GW event
        nside (jnp.ndarray): HEALPix nside parameter.

        nside_gw : Nside of the GW event
        nest_gw: Whether the GW event is in nested pixelization
        
    Returns:
        Array of shape (nmasks,) with fR values
    """
    # Compute background probability with the current cosmology: (z_points,) → (1, z_points)
    p_bkg = self.p_bkg(cosmo_obj, self.z_int_grid)[None, :]

    # Compute fR for all masks at once
    fR_masks = jax.vmap(lambda x: trapz(x, self.z_int_grid))(self.completeness * p_bkg)

    # Convert GW to completeness pixelization and get corresponding fR values
    fR_values = fR_masks[self.gw_mask_idxs]
    
    # Flatten if input is pseudo-2D
    return fR_values.flatten() if self.gw_mask_idxs.ndim == 1 else fR_values


def compute_completeness(cosmo_params, rho_gal_theo, z_grid, z_gal, sky_area,
                          Nz_to_bin=50, weights_gal=None, smooth=None, resample=None, mask=None):
    """Compute completeness as observed/theoretical density.
    Args:
        cosmo_params (object): Cosmological parameters.
        rho_gal_theo (callable): Theoretical density function.
        z_grid (jnp.ndarray): Redshift grid for output.
        z_gal (jnp.ndarray): Redshift of galaxies.
        sky_area (float): Sky area in steradians.
        Nz_to_bin (int): Number of bins for histogram.
        weights_gal (jnp.ndarray, optional): Weights for galaxies. Defaults to None.
        smooth (float, optional): Smoothing parameter. Defaults to None.
        resample (int, optional): Number of galaxies to resample. Defaults to None.
        mask (jnp.ndarray, optional): Mask for galaxies. Defaults to None.
    Returns:
        jnp.ndarray: Completeness as observed/theoretical density.
    """

    # Compute observed density with optimized binning
    dz = z_grid[1] - z_grid[0]
    z_edges_lowres = jnp.linspace(z_grid[0] - dz/2, z_grid[-1] + dz/2, Nz_to_bin + 1)
    z_bins_lowres = 0.5 * (z_edges_lowres[:-1] + z_edges_lowres[1:])
    dz_lowres = z_edges_lowres[1] - z_edges_lowres[0]
    
    # Extract galaxies in this mask - avoid copying data where possible
    z_gal = z_gal if mask is None else z_gal[mask]
    weights = None if weights_gal is None else (weights_gal if mask is None else weights_gal[mask])
    
    # Resample if needed - could be optimized if this is a bottleneck
    if resample and resample < z_gal.size:
      idx = jax.random.choice(rndkey, jnp.arange(z_gal.size), shape=(resample,), replace=False)
      z_gal = z_gal[idx]
      weights = None if weights is None else weights[idx]
    
    # Compute histogram and volume elements in one step
    N_z_obs, _ = jnp.histogram(z_gal, bins=z_edges_lowres, weights=weights)
    
    # Vectorized computation of volume elements
    dVdz = dVcdz_at_z(cosmo_params, z_bins_lowres)
    V_sky_Mpc = dz_lowres * 1e9 * dVdz * sky_area / (4. * jnp.pi)
    
    # Compute observed density
    rho_obs = N_z_obs / V_sky_Mpc
    
    # Interpolate to desired output bins
    rho_obs = interp(z_grid, z_bins_lowres, rho_obs)
    
    # Apply smoothing if specified
    if smooth:
        rho_obs = gaussian_filter1d(rho_obs, smooth)
    
    # Compute theoretical density once
    rho_theo = rho_gal_theo(z_grid)
    rho_theo = jnp.where(rho_theo < 1e-99, 1e-99, rho_theo)

    # Return clipped completeness (observed/theoretical density, clipped at 1)
    return jnp.minimum(rho_obs / rho_theo, 1.)


###########################################
# Completeness from an m_thr HEALPix map  #
###########################################
#
# Implements the Schechter-luminosity-function method of Appendix B of
# arXiv:2605.27227 (the GWTC-5 cosmology data release paper), reproducing the
# reference implementation `data_completeness_plot_O4b.py` (I. McMahon) shipped
# with the Zenodo record https://zenodo.org/records/20378418, which provides
# `mth_map_{DES_local_bandr,GLADEplus_bandK}_...fits` as pre-computed
# magnitude-threshold HEALPix maps for DES-Y6 (r-band) and GLADE+ (Ks-band).
#
# NOTE (2026-10-20): this used to re-implement the Schechter luminosity
# function locally (`_schechter_lf_mag`) rather than reusing
# `CHIMERA.utils.mags.schechter_mags`; it was consolidated to call
# `schechter_mags` directly (see below) -- it is the single canonical
# implementation of this formula in CHIMERA. `schechter_mags` includes the
# standard 0.4*ln(10) magnitude-space Jacobian prefactor, which is harmless
# here (it cancels exactly in `_build_cumulative_schechter`'s self-normalizing
# F(M) ratio) even though it isn't needed for that use.
#
# `schechter_mags`'s `x` sign convention (x = L/L_star = 10**(0.4*(M_star - M)),
# the standard astronomical convention) went through two reverts in one day --
# briefly fixed, then deliberately flipped to match `CHIMERA_private`'s
# (dormant/unused) convention, then flipped back to this standard form once
# that flip was shown to break this class's agreement with arXiv:2605.27227
# (at z=0.05/0.1/0.2 the flipped sign gave F~0.9997/0.9821/0.7905, not matching
# the paper's "<1% complete by z~0.2"; the standard sign gives
# F~0.6289/0.1472/0.0002, which does match). See the NOTE in
# `CHIMERA.utils.mags.schechter_mags`'s docstring for the full history. Unlike
# an earlier version of this note claimed, the sign convention is NOT
# inconsequential here: `_luminosity_weight(M) = 10**(-0.4*M)` below depends on
# absolute M (not M-M_star), so it does not symmetrize away the sign choice --
# it genuinely reshapes F(M), even though the ratio remains well-defined and
# monotonic under either convention.


def _luminosity_weight(M):
  """Luminosity weight L(M) ∝ 10**(-0.4*M), used to luminosity-weight the Schechter LF
  (matches `luminosity_weights` in the reference script)."""
  return jnp.power(10., -0.4*M)


class mth_completeness(object):
  r"""Completeness of a galaxy catalog from an apparent-magnitude-threshold
  (`m_thr`) HEALPix map, following the Schechter-luminosity-function method of
  Appendix B of arXiv:2605.27227 and the paper authors' reference script
  (see module docstring above).

  For a sky pixel with magnitude threshold ``m_thr``, the absolute-magnitude
  limit at redshift z (no K-correction) is::

      M_lim(z) = m_thr - mu(z),   mu(z) = 5*log10(dL(z)/Mpc) + 25

  and the completeness fraction of that pixel is the luminosity-weighted
  fraction of the (double-bounded) Schechter luminosity function observed
  brighter than that limit::

      P_compl(z) = ∫_{M_bright}^{M_lim(z)} L(M) phi(M) dM
                   / ∫_{M_bright}^{M_faint}  L(M) phi(M) dM

  clipped so that ``M_lim`` stays within ``[M_bright, M_faint]``. This ratio is
  independent of H0 and of ``phi_star`` when ``M_star``, ``M_bright``,
  ``M_faint`` follow the standard ``+5*log10(h)`` convention (both numerator and
  denominator scale identically) -- verified numerically. A fixed internal
  fiducial cosmology (``H0=100``, i.e. ``h=1``) is therefore used to evaluate
  ``mu(z)`` with the Schechter parameters taken at face value (unshifted), which
  gives exactly the same result as using the true analysis cosmology with
  properly shifted parameters. ``P_compl`` and ``fR`` are consequently
  independent of the ``cosmo_obj``/``cosmo_lambdas`` passed around the rest of
  CHIMERA (only ``p_bkg`` depends on it, exactly as for :class:`dVdz_completeness`).

  Each GW event's sky region generally spans several pixels (e.g.
  ``data_gw_pixelated.pixels_opt_nsides``, padded with ``-100``); the per-pixel
  completeness curves are simply averaged over the event's valid pixels to give
  one ``P_compl(z)``/``fR`` per event, matching the shapes expected by
  :func:`CHIMERA.catalog.catalog.precompute_p_cat` and
  :class:`CHIMERA.catalog.catalog.pixelated_catalog` (``P_compl`` of shape
  ``(Nev, Nz)``, ``fR`` of shape ``(Nev, 1)``).

  Args:
    mth_map (array-like): HEALPix map of apparent-magnitude thresholds, one
      value per pixel (0 or non-finite pixels, i.e. outside the survey
      footprint, are treated as fully incomplete).
    nside (int): HEALPix nside of ``mth_map``.
    nest (bool, optional): HEALPix ordering of ``mth_map`` (``True`` for
      NESTED, ``False`` for RING). Defaults to False.
    alpha (float): Schechter faint-end slope.
    M_star (float): Schechter characteristic absolute magnitude (unshifted, see above).
    M_bright (float): bright-end absolute-magnitude cutoff of the luminosity function
      (unshifted).
    M_faint (float): faint-end absolute-magnitude cutoff of the luminosity function
      (unshifted).
    phi_star (float, optional): Schechter normalization; cancels exactly in the
      completeness ratio, kept only for clarity. Defaults to 1.
    M_int_res (int, optional): number of points used to tabulate the cumulative
      luminosity-weighted Schechter integral. Defaults to 3000.
    z_int_range (list, optional): redshift range used to tabulate ``fR``. Defaults to
      ``[1e-4, 10.]`` (kept away from 0 to avoid a divergent distance modulus).
    z_int_res (int, optional): resolution of the ``fR`` redshift grid. Defaults to 1000.
    z_min, z_max (float, optional): enforce ``p_bkg = 0`` outside ``[z_min, z_max]``.

  Example (reproducing the paper's DES r-band / GLADE+ Ks-band parameters,
  ``H0=100`` convention, i.e. unshifted)::

    # DES-Y6 r-band (arXiv:2605.27227, App. B)
    compl_des = mth_completeness(mth_map_des, nside=128, nest=True,
                                  alpha=-1.01, M_star=-20.90, M_bright=-24.29, M_faint=-16.33)

    # GLADE+ Ks-band (arXiv:2605.27227, App. B)
    compl_glade = mth_completeness(mth_map_glade, nside=128, nest=False,
                                    alpha=-1.09, M_star=-23.39, M_bright=-27.0, M_faint=-19.0)
  """

  def __init__(self,
               mth_map: jnp.ndarray,
               nside: int,
               nest: bool = False,
               alpha: Number = -1.09,
               M_star: Number = -23.39,
               M_bright: Number = -27.0,
               M_faint: Number = -19.0,
               phi_star: Number = 1.,
               M_int_res: int = 3000,
               z_int_range: List[Number] = [1e-4, 10.],
               z_int_res: int = 1000,
               z_min: float = 0.,
               z_max: float = jnp.inf,
              ):

    logger.info("Initializing `mth_completeness`:")

    # healpy.read_map returns FITS data in big-endian byte order (">f8"); JAX
    # refuses non-native-endian arrays outright ("Dtype >f8 is not a valid JAX
    # array type"). Force a native-order numpy float64 copy first.
    mth_map_native = np.asarray(mth_map, dtype=np.float64)
    self.mth_map = jnp.where(jnp.isfinite(mth_map_native), mth_map_native, 0.)
    self.nside, self.nest = nside, nest
    self.npix, self.pixarea = hp.nside2npix(nside), hp.nside2pixarea(nside)

    self.alpha, self.M_star, self.phi_star = alpha, M_star, phi_star
    self.M_bright, self.M_faint = M_bright, M_faint
    self.M_int_res = M_int_res

    self.z_int_range = z_int_range
    self.z_int_res = z_int_res
    self.z_int_grid = jnp.linspace(*z_int_range, z_int_res)

    self.z_min, self.z_max = z_min, z_max

    # Fixed fiducial cosmology (H0=100, i.e. h=1) used only to evaluate the
    # distance modulus: the completeness fraction is invariant to this choice
    # when the Schechter parameters are taken unshifted (see class docstring).
    self._fiducial_cosmo = flrw(H0=100., Om0=flrw.default['Om0'])

    self.gw_mth = None  # set by `set_gw_mask_idxs`

    self._build_cumulative_schechter()

    logger.info(f"   - mth map: nside={self.nside}, nest={self.nest}, npix={self.npix}")
    logger.info(f"   - Schechter: alpha={self.alpha}, M_star={self.M_star}, "
                f"M_bright={self.M_bright}, M_faint={self.M_faint}")

  # --------------------------------------------
  # Luminosity-weighted Schechter completeness
  # --------------------------------------------
  def _build_cumulative_schechter(self):
    """Tabulate the normalized cumulative luminosity-weighted Schechter integral

      F(M) = ∫_{M_bright}^{M} L(M') phi(M') dM' / ∫_{M_bright}^{M_faint} L(M') phi(M') dM'

    on a grid M in [M_bright, M_faint], so that F(M_bright)=0 and F(M_faint)=1.

    Sets:
      self.M_grid (jnp.ndarray): magnitude grid, shape (M_int_res,).
      self.F_grid (jnp.ndarray): normalized cumulative integral on self.M_grid.
    """
    self.M_grid = jnp.linspace(self.M_bright, self.M_faint, self.M_int_res)

    integrand = schechter_mags(self.M_grid, self.alpha, self.M_star, self.phi_star) \
                * _luminosity_weight(self.M_grid)

    cumul = cumtrapz(integrand, self.M_grid)
    self.F_grid = cumul / cumul[-1]

  def _frac_observed(self, M_lim):
    """Luminosity-weighted completeness fraction for an absolute-magnitude limit M_lim
    (any shape), clipped to [M_bright, M_faint]."""
    M_lim_clip = jnp.clip(M_lim, self.M_bright, self.M_faint)
    return interp(M_lim_clip, self.M_grid, self.F_grid, left=0., right=1.)

  def _mu_at_z(self, z):
    """Distance modulus mu(z) = 5*log10(dL(z)/Mpc) + 25, no K-correction.
    `dL_at_z` returns distances in Gpc (CHIMERA's internal convention, see
    `flrw.dH`), hence the +15 = 5*log10(1000) unit conversion below."""
    dL_Gpc = jnp.clip(dL_at_z(self._fiducial_cosmo, z), 1e-12, None)
    return 5. * jnp.log10(dL_Gpc) + 40.

  def _P_compl_pixels(self, event_mth, event_z_grid):
    """Per-pixel completeness for one event: shape (Npix, Nz)."""
    mu = self._mu_at_z(event_z_grid)                  # (Nz,)
    M_lim = event_mth[:, jnp.newaxis] - mu[jnp.newaxis, :]  # (Npix, Nz)
    return self._frac_observed(M_lim)

  # ----------------
  # GW pixelization
  # ----------------
  def set_gw_mask_idxs(self, pix_gw, nside_gw, nest_gw):
    """Convert the GW pixelization (e.g. ``data_gw_pixelated.pixels_opt_nsides``,
    with ``-100`` used to pad events with fewer than the maximum number of
    credible pixels) to this map's pixelization and look up the per-pixel
    ``m_thr`` values. Padded entries are stored as NaN so they are excluded
    when `P_compl`/`fR` average over an event's valid pixels."""
    pix_gw   = np.asarray(pix_gw)
    valid    = pix_gw != -100
    # `pix_gw` comes out of `data.pixelize_gw_catalog` as a float array (it is
    # padded alongside the `-100.` sentinel), but `convert_pixelization` feeds
    # it straight into `hp.pix2ang`, which requires integer pixel indices --
    # hence the explicit `.astype(int)` below, on top of the `-100` -> `0`
    # substitution. (NOTE 2026-10-20: this was previously untested against
    # real `pixelize_gw_catalog` output, only against hand-built pixel
    # arrays, which is how this went unnoticed.)
    pix_safe = np.where(valid, pix_gw, 0).astype(int)

    pix_map = convert_pixelization(pix_safe, np.asarray(nside_gw), self.nside, nest_gw, self.nest)
    mth = np.asarray(self.mth_map)[pix_map]

    self.gw_mth = jnp.asarray(np.where(valid, mth, np.nan))

  # -----------------------
  # P_compl, p_bkg, and fR
  # -----------------------
  def P_compl(self, z_grids):
    """Compute P_compl(z) on analysis z_grids for all the GW events, averaged
    over each event's valid sky pixels (see class docstring).
    Approx.: P_compl is cosmology independent (true for H0), see class docstring.

    Args:
      z_grids (jnp.ndarray): per-event redshift grids, shape (Nev, Nz).

    Returns:
      jnp.ndarray: shape (Nev, Nz).
    """
    if self.gw_mth is None:
      raise RuntimeError("Attribute not initialized. Call set_gw_mask_idxs() first.")

    def process_event(event_mth, event_z_grid):
      pc_pix = self._P_compl_pixels(event_mth, event_z_grid)      # (Npix, Nz)
      valid  = jnp.isfinite(event_mth)[:, jnp.newaxis]             # (Npix, 1)
      n_valid = jnp.maximum(jnp.sum(valid, axis=0), 1)
      return jnp.sum(jnp.where(valid, pc_pix, 0.), axis=0) / n_valid  # (Nz,)

    return jax.vmap(process_event)(self.gw_mth, z_grids)

  def p_bkg(self, cosmo_obj, z, distances=None):
    """Homogeneous (uniform comoving-density) background for the out-of-catalog
    term, matching the convention of :class:`dVdz_completeness`."""
    return jnp.where((z >= self.z_min) & (z <= self.z_max),
                      dVcdz_at_z(cosmo_obj, z, distances=distances),
                      0.)

  def fR(self, cosmo_obj):
    """Compute fR = ∫ P_compl * p_bkg dz for each event, with a given cosmology.
    Called within MCMC iterations when cosmology changes.

    Returns:
      jnp.ndarray: shape (Nev, 1), ready for `jnp.atleast_3d` broadcasting
      against `p_cat` of shape (Nev, Npix, Nz) in `pixelated_catalog.p_gal`.
    """
    if self.gw_mth is None:
      raise RuntimeError("Attribute not initialized. Call set_gw_mask_idxs() first.")

    nevents = self.gw_mth.shape[0]
    z_int_grids = jnp.broadcast_to(self.z_int_grid, (nevents, self.z_int_grid.shape[0]))

    P_compl_int = self.P_compl(z_int_grids)                 # (Nev, Nz_int)
    p_bkg_int   = self.p_bkg(cosmo_obj, self.z_int_grid)     # (Nz_int,)

    fR = trapz(P_compl_int * p_bkg_int[jnp.newaxis, :], self.z_int_grid, axis=-1)  # (Nev,)
    return fR[:, jnp.newaxis]
