import equinox as eqx
from typing import Optional, Dict, List, Union
from numbers import Number

from ..utils.config import jax, jnp, logger
from ..utils.math import trapz
from ..utils.angles import find_pix_RAdec, convert_pixelization
from ..utils.io import load_set, save_set
from ..population.cosmo import dVcdz_at_z, Vc_at_z
from scipy.ndimage import gaussian_filter1d
import healpy as hp
from scipy.cluster.vq import kmeans
from plum import dispatch
from ..data import theta_src, theta_pe_det, theta_inj_det

rndkey = jax.random.PRNGKey(0)

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

  @dispatch
  def p_bkg(self, cosmo_lambdas:eqx.Module, theta_src:eqx.Module):
    dVdz = dVcdz_at_z(cosmo_lambdas, theta_src)
    return dVdz

  @dispatch
  def p_bkg(self, cosmo_lambdas:eqx.Module, z:jnp.ndarray):
    return self.p_bkg(cosmo_lambdas, theta_src(z=z))

############################
# Homogeneous Completeness #
############################

class homogeneous_completeness(object):
  """A class to compute the completeness of a galaxy catalog as a density fraction w.r.t. a theoretical function.

  """

  def __init__(self,
    z_gal: jnp.ndarray,
    theory_density_func: object,
    cosmo_lambdas: eqx.Module,
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
    self.cosmo_lambdas = cosmo_lambdas
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

  # def observed_density(self, cosmo_lambdas, z_grid=None):
  #   """
  #   Compute the observed density of galaxies given their redshifts, potentially including weights.
  #   """
  #   z_grid = self.z_grid_bin if z_grid is None else z_grid
  #   if self.resample and self.resample < self.z_gal.size:
  #     z_gal = jax.random.choice(rndkey, self.z_gal, shape=(self.resample,), replace=False)
  #   else:
  #     z_gal = self.z_gal
  #   N_z_obs, _ = jnp.histogram(z_gal, bins=z_grid, weights=self.weights)
  #   z_bins     = 0.5*(z_grid[:-1] + z_grid[1:])
  #   dz         = z_grid[1] - z_grid[0]
  #   V_sky_Mpc  = dz * 1e9 * dVcdz_at_z(cosmo_lambdas, z_bins) * self.sky_area_sr / (4. * jnp.pi)
  #   rho_obs = N_z_obs / V_sky_Mpc
  #   return z_bins, rho_obs

  # def compute_P_compl(self, cosmo_lambdas, z_grid=None):
  #   z_grid = self.z_grid_bin if z_grid is None else z_grid
  #   # Compute observed density
  #   z_bins, rho_obs = self.observed_density(cosmo_lambdas, z_grid)
  #   if self.smooth is not None:
  #     zz = jnp.linspace(*self.z_range, self.Nz_interp)
  #     rho_obs = gaussian_filter1d(jnp.interp(zz, z_bins, rho_obs), self.smooth)
  #     z_bins = zz
  #   # Compute theoretical density
  #   rho_theo = self.theory_density_func(z_bins)
  #   # Compute completeness (observed/theoretical density, clipped at 1)
  #   compl = jnp.minimum(rho_obs / rho_theo, 1.)
  #   return z_bins, rho_obs, rho_theo, compl

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
      return jnp.interp(z, self.z_int_grid, self.completeness[mask_idx], left=0, right=0)

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
    self.completeness = compute_completeness(self.cosmo_lambdas, self.n_gal_theo, self.z_int_grid, self.z_gal,
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
    return jax.vmap(lambda z_ev : jnp.interp(z_ev, self.z_int_grid, self.completeness))(z_grids)

  @dispatch
  def p_bkg(self, cosmo_lambdas:eqx.Module, theta_src:eqx.Module):
    """Compute background probability on arbitrary z_grid, normalized over the range defined by self.z_int_grid.

    Args:
        cosmo: Cosmology object
        z: Redshift value(s) where to evaluate the function

    Returns:
        Normalized probability values at z points
    """
    bkg = jnp.where((self.z_int_grid >= self.z_min) & (self.z_int_grid <= self.z_max),
                    self.n_gal_theo(self.z_int_grid) * dVcdz_at_z(cosmo_lambdas, self.z_int_grid),
                    0.)
    norm = trapz(bkg, self.z_int_grid)
    return jnp.interp(theta_src.z, self.z_int_grid, bkg/norm, left=0, right=0)

  @dispatch
  def p_bkg(self, cosmo_lambdas:eqx.Module, z:jnp.ndarray):
    return self.p_bkg(cosmo_lambdas, theta_src(z=z))

  def fR(self, cosmo_lambdas):
    """Compute fR = ∫ P_compl * p_bkg dz for all masks with a given cosmology. Called within MCMC when cosmology changes.

    Parameters:
        cosmo_lambdas: Cosmological parameters for this iteration
        pix_gw (jnp.ndarray): Pixelization of the GW event
        nside (jnp.ndarray): HEALPix nside parameter.

        nside_gw : Nside of the GW event
        nest_gw: Whether the GW event is in nested pixelization

    Returns:
        Array of shape (nmasks,) with fR values
    """
    p_bkg = self.p_bkg(cosmo_lambdas, self.z_int_grid)
    return trapz(self.completeness * p_bkg, self.z_int_grid)


# Core functions

def compute_completeness(cosmo_lambdas, n_gal_theo, z_grid, z_gal, sky_area,
                          Nz_to_bin=50, weights_gal=None, smooth=None, resample=None, mask=None):
    """Compute completeness as observed/theoretical density.
    Args:
        cosmo_lambdas (object): Cosmological parameters.
        n_gal_theo (callable): Theoretical density function.
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
    dVdz = dVcdz_at_z(cosmo_lambdas, z_bins_lowres)
    V_sky_Mpc = dz_lowres * 1e9 * dVdz * sky_area / (4. * jnp.pi)

    # Compute observed density
    rho_obs = N_z_obs / V_sky_Mpc

    # Interpolate to desired output bins
    rho_obs = jnp.interp(z_grid, z_bins_lowres, rho_obs)

    # Apply smoothing if specified
    if smooth:
        rho_obs = gaussian_filter1d(rho_obs, smooth)

    # Compute theoretical density once
    rho_theo = n_gal_theo(z_grid)
    rho_theo = jnp.where(rho_theo < 1e-99, 1e-99, rho_theo)

    # Return clipped completeness (observed/theoretical density, clipped at 1)
    return jnp.minimum(rho_obs / rho_theo, 1.)
