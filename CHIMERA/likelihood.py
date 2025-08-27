from .utils.config import jax, jnp, logger
from .utils.math import trapz,  kde1d, binning1d, jax_gkde_nd, numba_gkde_nd
from .population.cosmo import ddLdz_at_z
from .population import get_theta_src_and_weights, p_cbc

from jax.experimental import io_callback
import equinox as eqx
from typing import Optional, List, Dict, Union
from numbers import Number
from functools import partial
import numpy as np
from .data import theta_src, theta_pe_det

class hyperlikelihood(object):
  r"""A class for computing the cosmological/populationulation hyperlikelihood of gravitational wave (GW) events.

  Args:
    data_gw (Dict[str, jnp.ndarray]): Dictionary containing GW posterior samples and pixelation metadata.
      - For pixelated data, it includes 'pixels', 'gw_loc2d_pdf', and related arrays.
      - For non-pixelated data, it contains direct event data.
    pe_prior (jnp.ndarray): Prior probability distribution from the parameter estimation (PE) of GW events.
    population (object): A CHIMERA.population instance.
    sel_eff (Optional[object]): A CHIMERA.sekection_effects object used to compute the bias function :math:`\xi(\lambda)`. Defaults to `None`.
    kind_p_gw3d (Optional[str]): Type of 3D GW probability computation. Options: 'approximate', 'marginalized', 'full'.
      Used only for pixelated GW catalogs.
    kernel (str): Kernel type for KDE computations. Options: 'epan' (Epanechnikov kernel), 'gauss' (Gaussian kernel). Defaults to 'epan'.
      Note: 'epan' is only available for 'approximate' or 'marginalized' cases and for non-pixelated data.
    bw_method (Optional[Number]): Bandwidth selection method for KDE computations. Defaults to `None`.
    cut_grid (Number): Cut-off grid size for KDE computations to limit numerical range. Defaults to 2.0.
    binning (bool): Whether to apply binning to the data for efficiency. Defaults to `True`.
      Available only for 'approximate' or 'marginalized' cases and for non-pixelated data.
    num_bins (int): Number of bins used for KDE binning. Defaults to 200.
    positive_weights_thresh (Number): Minimum threshold for effective positive weights in KDE. Defaults to 5.0.
    pe_neff (Number): Minimum effective sample size (Neff) required for valid events. Defaults to 5.0.
    cosmo_prior (Optional[Dict[str, List[Number]]]): Priors on cosmological parameters. Defaults to `None`.
    z_conf_range (Union[Number, list]): Confidence range for the redshift grid, used in redshift estimation. Defaults to 5.0.
    z_int_res (int): Resolution of the redshift grid for computations. Defaults to 300.

  Class Attributes:
    - pixelated (bool): Indicates if the input data is pixelated ('pixels' key exists in `data_gw`).
    - nevents (int): Total number of GW events.
    - z_grids (jnp.ndarray): Precomputed redshift grids for each GW event.

  Other Methods:
    - p_gw3d(**hyper_lambdas): wrapper for `p_gw3dapprox`, `p_gw3dmarg`, or `p_gw3dfull` depenging on `kind_p_gw3d`
    - compute_log_like_nums(**hyper_lambdas): compute the single-events contributions to the hyper-likelihood numerator.
  """
  def __init__(
    self,
    theta_gw_det: theta_pe_det,
    z_grids: jnp.ndarray,
    population: eqx.Module,
    selection_function: Optional[object] = None,
    kind_p_gw3d: Optional[str] = None,
    kernel: str = 'epan',
    bw_method: Optional[Number] = None,
    cut_grid: Number = 2.0,
    binning: bool = True,
    num_bins: int = 200,
    #positive_weights_thresh: Number = 5.0,
    pe_neff: Number = 2.0,
  ):

    # Initialize attributes
    self.theta_gw_det = theta_gw_det
    self.population = population
    self.z_grids = z_grids
    self.selection_function = selection_function
    self.kind_p_gw3d = kind_p_gw3d
    self.kernel = kernel
    self.bw_method = bw_method
    self.cut_grid = cut_grid
    self.binning = binning
    self.num_bins = num_bins
    #self.positive_weights_thresh = positive_weights_thresh
    self.pe_neff = pe_neff

    # Other useful quantity
    self.pixelated = True if self.theta_gw_det.pixels_opt_nsides is not None else False
    self.nevents = len(self.theta_gw_det.dL)
    self.z_int_res = self.z_grids.shape[1]

    # Handle pixelated likelihood
    if self.pixelated:
      assert self.kind_p_gw3d in ['approximate','marginalized','full'], "`kind_p_gw3d` must be one of `approximate`, `marginalized`, or `full`"
      self.max_npixels = self.population.gal_cat.max_npixels
      self.neff_pixels = self.population.gal_cat.neff_pixels
      self.p_gw3d = {
              'approximate': self.p_gw3dapprox,
              'marginalized': self.p_gw3dmarg,
              'full': self.p_gw3dfull
            }[self.kind_p_gw3d]
      if self.kind_p_gw3d == 'full':
        logger.info("`king_p_gw3d` has been set to 'full'. Only available kernel is `gaussian`. The `binning` option is not available.")
      self.compute_numlike_evs = self._compute_numlike_evs_pixelated
    else:
      self.compute_numlike_evs = self._compute_numlike_evs_no_pixels

    logger.info(f'Created hyperlikelihood model. Using {self.nevents} GW events.')

  ######################
  # p_gw1d computation #
  ######################

  def p_gw1d(self, pop_lambdas):
    r"""Computes :math:`p_gw(z | \lambda)`. Used for non-pixelated catalogs or when "king_p_gw3d='approximate'"."""
    # convert thetas
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)

    # compute normalization and n_effective
    norms  = jnp.mean(weights, axis = -1)
    n_effs = jnp.sum(weights, axis = -1)**2 / jnp.sum(weights**2, axis = -1)

    # setup effective evaluation grid
    if self.cut_grid is not None:
      data_min = jnp.min(th_src.z, axis=-1)
      data_max = jnp.max(th_src.z, axis=-1)
      sigma = jnp.std(th_src.z, axis=-1)
      lb = jnp.where(data_min - self.cut_grid*sigma > 0., data_min - self.cut_grid*sigma, 1.e-8)
      ub = data_max + self.cut_grid*sigma
      eff_grids = jnp.linspace(lb, ub, self.z_int_res // 2).T
    else:
      eff_grids = self.z_grids

    # optional binning
    if self.binning:
      zs, weights = jax.vmap(binning1d, in_axes=(0,0,None))(th_src.z, weights, self.num_bins)
    else:
      zs, weights = th_src.z, weights

    # single event routine
    def single_ev_routine(ev):
      condition = (n_effs[ev] >= self.pe_neff)
      # compute kde
      return jax.lax.cond(
        condition,
        lambda _: jnp.interp(self.z_grids[ev], eff_grids[ev], kde1d(zs[ev], eff_grids[ev], weights[ev], self.kernel, self.bw_method)*norms[ev], left = 0., right = 0.),
        lambda _: jnp.zeros(self.z_int_res),
        operand=None
      )

    # v-mapping over all events
    p_gw = jax.vmap(single_ev_routine)(jnp.arange(self.nevents))
    return p_gw

  ###################################
  # p_gw3d approximated computation #
  ###################################

  def p_gw3dapprox(self, pop_lambdas):
    r"""Computes :math:`p_gw(z, RA, Dec | \lambda)` when "king_p_gw3d='approximate'"."""
    p_gw1d = self.p_gw1d(pop_lambdas) # (Nevents, ResGrids)
    p_gw3d = p_gw1d[:,None,:] * self.theta_gw_det.gw_loc2d_pdf[:,:,None] # (Nevents, MaxPixels, ResGrids)
    return p_gw3d

  ###################################
  # p_gw3d marginalized computation #
  ###################################

  def p_gw3dmarg(self, pop_lambdas):
    """Computes p_gw(z, RA, Dec | λ) when kind_p_gw3d='marginalized'."""

    # Get source frame samples and population weights
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    norms = jnp.mean(weights, axis=-1)
    n_effs = jnp.sum(weights, axis=-1) ** 2 / jnp.sum(weights**2, axis=-1)

    def p_gw_single_event(ev): # single event routine
      z = th_src.z[ev]  # (Nsamples,)
      w = weights[ev]   # (Nsamples,)
      norm = norms[ev]
      n_eff = n_effs[ev]
      zgrid = self.z_grids[ev]
      pe_pix = self.theta_gw_det.pixels_pe_opt_nside[ev]   # (Nsamples,)
      pixels = self.theta_gw_det.pixels_opt_nsides[ev]     # (Npixels,)
      gw_pdf = self.theta_gw_det.gw_loc2d_pdf[ev]          # (Npixels,)

      def pixels_loop(i, acc):
        pixel_mask = pe_pix == pixels[i]
        z_masked = jnp.where(pixel_mask, z, jnp.min(z))
        w_masked = jnp.where(pixel_mask, w, 0.0)
        # Optional binning
        z_pix, w_pix = binning1d(z_masked, w_masked, self.num_bins) if self.binning else (z_masked, w_masked)
        # Effective grid
        if self.cut_grid is not None:
          zmin = jnp.maximum(jnp.min(z) - self.cut_grid * jnp.std(z), 1e-8)
          zmax = jnp.max(z) + self.cut_grid * jnp.std(z)
          eff_grid = jnp.linspace(zmin, zmax, self.z_int_res // 2)
        else:
          eff_grid = zgrid
        # KDE on effective grid
        kde_eff = kde1d(z_pix, eff_grid, weights=w_pix, bw_method=self.bw_method)
        kde_interp = jnp.interp(zgrid, eff_grid, kde_eff, left=0.0, right=0.0)
        result = kde_interp * norm * gw_pdf[i]
        return acc.at[i].set(result)

      # check n_eff and the do a lax.fori_loop over all pixels of the ev-th event
      return jax.lax.cond(
        n_eff >= self.pe_neff,
        lambda _: jax.lax.fori_loop(0, self.max_npixels, pixels_loop, jnp.zeros((self.max_npixels, zgrid.shape[0]))),
        lambda _: jnp.zeros((self.max_npixels, zgrid.shape[0])),
        operand=None
      )
    # mapping the single event routine over all events
    return jax.vmap(p_gw_single_event)(jnp.arange(self.nevents))

  ###########################
  # p_gw3d full computation #
  ###########################

  def p_gw3dfull(self, pop_lambdas):
    r"""Computes :math:`p_gw(z, RA, Dec | \lambda)` when "king_p_gw3d='full'"."""
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    norms  = jnp.mean(weights, axis = -1)
    n_effs = jnp.sum(weights, axis = -1)**2 / jnp.sum(weights**2, axis = -1)

    # manage dataset
    dataset = jnp.array([th_src.z, self.theta_gw_det.ra, self.theta_gw_det.dec]) # dataset for kde, shape: (3, Nevents, Nsamples)
    dataset  = jnp.moveaxis(dataset, 0, 1) # shape (Nevents,3,Nsamples)

    # prepare effective grid
    z_std = jnp.std(th_src.z, axis=1, keepdims=True)
    z_max = jnp.max(th_src.z, axis=1, keepdims=True)
    z_min = jnp.min(th_src.z, axis=1, keepdims=True)
    z_masks = (self.z_grids <= z_max + self.cut_grid*z_std) & (self.z_grids >= z_min - self.cut_grid*z_std)

    # Package everything into a single object to pass into callback
    callback_input = (dataset, weights, norms, n_effs, z_masks, self.z_grids)

    def all_events_callback(args):
      dataset_np, weights_np, norms_np, n_effs_np, z_masks_np, z_grids_np = map(np.asarray, args)
      result = np.zeros((self.nevents, self.max_npixels, self.z_int_res))
      for ev in range(self.nevents):
        if n_effs_np[ev] < self.pe_neff:
          continue
        z_grid = z_grids_np[ev]
        z_mask = z_masks_np[ev]
        z_eff_grid = z_grid[z_mask]
        npix = int(self.neff_pixels[ev])
        ra_pix = np.asarray(self.theta_gw_det.ra_pix[ev, :npix])
        dec_pix = np.asarray(self.theta_gw_det.dec_pix[ev, :npix])
        norm = norms_np[ev]
        eff_grid = np.array([
          np.tile(z_eff_grid, npix),
          np.hstack([np.full_like(z_eff_grid, ra) for ra in ra_pix]),
          np.hstack([np.full_like(z_eff_grid, dec) for dec in dec_pix]),
        ])
        eff_mask = np.tile(z_mask, npix)
        dat = dataset_np[ev]
        w = weights_np[ev]
        kde_vals = np.zeros(npix * self.z_int_res)
        kde_vals[eff_mask] = numba_gkde_nd(dat, eff_grid, weights=w, bw_method=self.bw_method, in_log=False)
        result[ev, :npix, :] = kde_vals.reshape(npix, self.z_int_res) * norm
      return result.astype(np.float64)

    return io_callback(
        all_events_callback,
        jax.ShapeDtypeStruct((self.nevents, self.max_npixels, self.z_int_res), jnp.float64),
        callback_input
    )

  ####################################
  # likelihood numerator computation #
  ####################################

  def _compute_numlike_evs_pixelated(self, pop_lambdas):
    # p_gw(z, ra, dec | \theta_gw, \lambda_c, \lambda_m)
    p_gw3d = self.p_gw3d(pop_lambdas) # (Nevents, MaxPixels, ResGrids)
    # p_z of having a cbc at z
    p_z = p_cbc(pop_lambdas, self.z_grids) # (Nevents, MaxPixels, ResGrids)
    # jacobian
    jacobian = ddLdz_at_z(pop_lambdas.cosmo, self.z_grids) * (1.+self.z_grids)**2
    # Integral in all pixels, but avoid the `fake` ones
    integrand = jnp.where(p_z != -100,
      p_gw3d * p_z / jacobian[:,None,:],
      jnp.zeros((self.nevents, self.max_npixels, self.z_int_res))
    ) # (Nevents, MaxPixels, ResGrids)
    like_evs_pixels = trapz(integrand, self.z_grids[:,None,:], axis = -1) # (Nevents, MaxPixels)
    # Sum pixel contributions
    like_evs = jnp.sum(like_evs_pixels, axis = -1) # (Nevents,)
    return like_evs

  def _compute_numlike_evs_no_pixels(self, pop_lambdas):
    # p_gw(z,| \theta_gw, \lambda_c, \lambda_m)
    p_gw = self.p_gw1d(pop_lambdas)
    # p_z of having a cbc at z
    p_z = p_cbc(pop_lambdas, self.z_grids)
    # jacobian
    jacobian = ddLdz_at_z(pop_lambdas.cosmo, self.z_grids) * (1.+self.z_grids)**2
    # Integral
    like_evs = trapz(p_gw*p_z/jacobian, self.z_grids, axis = -1)
    return like_evs

  def compute_log_likenum(self, pop_lambdas):
    """Computes the numerator of the log hyper-likelihood."""
    log_like_evs = jnp.log(self.compute_numlike_evs(pop_lambdas))
    log_like_evs = jnp.nan_to_num(log_like_evs, nan=-jnp.inf)
    log_num = jnp.sum(log_like_evs, axis=-1)
    if not pop_lambdas.scale_free:
      log_num += self.nevents*jnp.log(pop_lambdas.R0*pop_lambdas.Tobs)
    return log_num

  ################################
  # hyper-likelihood computation #
  ################################

  @partial(jax.jit, static_argnums=(0,))
  def compute_log_hyperlike(self, **hyper_lambdas):
    """Computes the of the log hyper-likelihood."""
    pop_lambdas = self.population.update(**hyper_lambdas)
    log_like_num = self.compute_log_likenum(pop_lambdas)
    N_exp = self.selection_function.N_exp(pop_lambdas)
    if pop_lambdas.scale_free:
      return log_like_num - self.nevents*jnp.log(N_exp)
    else:
      return log_like_num - N_exp

  @partial(jax.jit, static_argnums=(0,))
  def __call__(self, **hyper_lambdas):
    return self.compute_log_hyperlike(**hyper_lambdas)

   ######################
   # debugging function #
   ######################

  @partial(jax.jit, static_argnums=(0,))
  def compute_all(self, **hyper_lambdas):
    pop_lambdas = self.population.update(**hyper_lambdas)
    log_like_evs = jnp.log(self.compute_numlike_evs(pop_lambdas))
    log_like_evs = jnp.nan_to_num(log_like_evs, nan=-jnp.inf)
    log_like_num = jnp.sum(log_like_evs, axis=-1)
    N_exp = self.selection_function.N_exp(pop_lambdas)
    if not pop_lambdas.scale_free:
      log_like_num += self.nevents*jnp.log(pop_lambdas.R0*pop_lambdas.Tobs)
      log_hyper = log_like_num  - N_exp
    else:
      log_hyper = log_like_num - self.nevents*jnp.log(N_exp)
    return log_like_evs, log_like_num, jnp.log(N_exp), log_hyper
