from jax.experimental import io_callback
import equinox as eqx
from typing import Optional, Dict
from numbers import Number
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from KDExpress import binned_kde1d, fft_kde1d, scott_bw1d, build_hist_edges, fft_kde3d

from .utils.config import logger
from .utils.math import trapz, kde1d
from .population.cosmo import ddLdz_at_z
from .population import get_theta_src_and_weights, p_cbc
from .data import theta_pe_det


class hyperlikelihood(object):
  r"""A class for computing the cosmological/populationulation hyperlikelihood of gravitational wave (GW) events.

  Args:
    data_gw (Dict[str, jnp.ndarray]): Dictionary containing GW posterior samples, PE prior, and pixelation metadata.
      - For pixelated data, it includes 'pixels', 'gw_loc2d_pdf', and related arrays.
      - For non-pixelated data, it contains direct event data.
    population (object): A CHIMERA.population instance.
    selection_function (Optional[object]): A CHIMERA.selection_effects object used to compute the bias function :math:`\xi(\lambda)`. Defaults to `None`.
    z_grid_res (Optional[int]): Redshift integration grid resolution: for spectral sirens only
    pe_neff (Optional[Number]): Minimum effective sample size (Neff) required for valid events. Defaults to 5.0.
    kind_p_gw3d (Optional[str]): Type of 3D GW probability computation. Options: 'approximate', 'marginalized', 'full'.
      Used only for pixelated GW catalogs.
    kernel (str): Kernel type for KDE computations. Options: 'epan' (Epanechnikov kernel), 'gauss' (Gaussian kernel). Defaults to 'epan'.
      Note: 'epan' is only available for 'approximate' or 'marginalized' cases and for non-pixelated data.
    bw_method (Optional[Number]): Bandwidth selection method for KDE computations. Defaults to `None`.
    cut_grid (Number): Cut-off grid size for KDE computations to limit numerical range. Defaults to 2.0.
    binning (bool): Whether to apply binning to the data for efficiency. Defaults to `True`.
      Available only for 'approximate' or 'marginalized' cases and for non-pixelated data.
    num_bins (int): Number of bins used for KDE binning. Defaults to 200.

  Class Attributes:
    - pixelated (bool): Indicates if the input data is pixelated ('pixels' key exists in `data_gw`).
    - nevents (int): Total number of GW events.
    - pixelization metadata

  Other Methods:
    - p_gw3d(**hyper_lambdas): wrapper for `p_gw3dapprox`, `p_gw3dmarg`, or `p_gw3dfull` depenging on `kind_p_gw3d`
    - compute_log_like_nums(**hyper_lambdas): compute the single-events contributions to the hyper-likelihood numerator.
  """
  def __init__(
    self,
    theta_gw_det: theta_pe_det,
    population: eqx.Module,
    z_grids_res: Optional[int] = 300,
    selection_function: Optional[object] = None,
    pe_neff: Number = 2.0,
    inj_neff: Optional[Number] = None,
    kind_kde: Optional[str] = None,
    kernel: str = 'epan',
    kde_bw: Optional[Number] = None,
    cut_grid: Number = 2.0,
    binning: bool = True,
    num_bins: int = 200,
  ):

    # Initialize attributes
    self.theta_gw_det = theta_gw_det
    self.nevents = len(self.theta_gw_det.dL)
    self.population = population
    self.z_int_res = z_grids_res
    self.selection_function = selection_function
    self.pe_neff = pe_neff
    self.inj_neff = inj_neff if inj_neff is not None else 5*self.nevents
    self.kind_kde = kind_kde
    self.kernel = kernel
    assert self.kernel in ['epan', 'gaussian'], "Kernel must be either 'epan' or 'gaussian'"
    self.kde_bw = kde_bw
    if self.kde_bw is None:
      logger.warning("`kde_bw` is None, using Scott rule for bandwidth.")
    self.cut_grid = cut_grid
    self.num_bins = num_bins

    # Other useful quantity
    self.pixelated = True if self.theta_gw_det.pixels_opt_nsides is not None else False

    # Handle pixelated likelihood
    if self.pixelated:
      self.z_grids = self.population.gal_cat.z_grids
      self.z_int_res = self.z_grids.shape[1]
      self.z_grids_edges = jax.vmap(build_hist_edges)(self.z_grids)
      kdes_available = ['single-1d-binned', 'single-1d-fft', 'single-1d-standard',
                        'many-1d-binned', 'many-1d-fft', 'many-1d-standard', '3d']
      assert self.kind_kde in kdes_available, f"`kind_kde` must be in {kdes_available}"
      kde_handlers = {
        'single-1d-binned': self.p_gw3d_single1d,
        'single-1d-fft': self.p_gw3d_single1d,
        'single-1d-standard': self.p_gw3d_single1d,
        'many-1d-binned': self.p_gw3d_many1d,
        'many-1d-fft': self.p_gw3d_many1d,
        'many-1d-standard': self.p_gw3d_many1d,
        'fft-3d': self.p_gw3d_full
      }
      self.p_gw3d = kde_handlers[self.kind_kde]
      # Handle warnings
      if self.kind_kde in ['single-1d-binned', 'many-1d-binned']:
        if self.num_bins is None and self.cut_grid is None:
          logger.info(f"Using binned KDE with num_bins={self.num_bins}, cut_grid={self.cut_grid}")
      else:
        if self.num_bins is not None:
          logger.info(f"`kind_kde` is {self.kind_kde}. Ignoring `num_bins`")
        if self.cut_grid is not None:
          logger.info(f"`kind_kde` is {self.kind_kde}. Ignoring `cut_grid`")
      if self.kind_kde == '3d':
        if self.kernel != 'gaussian':
          logger.info("3D KDE only supports Gaussian kernel. Setting it to 'gaussian'")
          self.kernel = 'gaussian'
        if self.cut_grid is not None or self.num_bins is not None:
          logger.info("3D KDE ignores `cut_grid` and `num_bins`")

      self.compute_like_num_evs = self._compute_like_num_evs_pixelated

    # Handle spectral sirens likelihood
    else:
      kdes_available = ['binned', 'fft', 'standard']
      assert self.kind_kde in kdes_available, f"`kind_kde` must be in {kdes_available}"
      if self.kind_kde == 'fft' and self.kernel == 'epan':
        logger.info("FFT KDE only supports Gaussian kernel. Setting to 'gaussian'")
        self.kernel = 'gaussian'
      self.compute_like_num_evs = self._compute_like_num_evs_no_pixels

    logger.info(f'Created hyperlikelihood model. Using {self.nevents} GW events.')

  ######################
  # p_gw1d computation #
  ######################

  def p_gw1d(self, pop_lambdas):
    r"""Computes :math:`p_gw(z | \lambda)`. Used for non-pixelated catalogs or when "king_kde='single-1d-'"."""
    # convert thetas
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    if not self.pixelated:
      z_grids = jnp.linspace(jnp.min(th_src.z, axis = -1)*0.5, jnp.max(th_src.z, axis = -1)*2, self.z_int_res).T
      cut_grid = 0
      if self.kind_kde == 'fft':
        z_grids_edges = jax.vmap(build_hist_edges)(z_grids)
    else:
      z_grids = self.z_grids
      cut_grid = self.cut_grid
      z_grids_edges = self.z_grids_edges

    # compute normalization and n_effective
    norms  = jnp.mean(weights, axis = -1)
    n_effs = jnp.sum(weights, axis = -1)**2 / jnp.sum(weights**2, axis = -1)

    # Vectorized KDE
    if self.kind_kde == 'binned':
      kde_vec = jax.vmap(binned_kde1d, in_axes=(0,0,0,None,None,None,None))
      _p_gw = kde_vec(z_grids, th_src.z, weights, self.kde_bw, self.kernel, self.num_bins, cut_grid)
    elif self.kind_kde == 'fft':
      kde_vec = jax.vmap(fft_kde1d, in_axes=(0,0,0,None,None,0))
      _p_gw = kde_vec(z_grids, th_src.z, weights, self.kde_bw, self.kernel, z_grids_edges)
    else:
      kde_vec = jax.vmap(kde1d, in_axes=(0,0,0,None,None))
      _p_gw = kde_vec(th_src.z, z_grids, weights, self.kernel, self.kde_bw)

    # Apply Neff_PE condition and normalize
    p_gw = jnp.where(n_effs[:,None]>self.pe_neff, _p_gw, 0.0)*norms[:,None]
    return p_gw, z_grids

  ###################################
  # p_gw3d approximated computation #
  ###################################

  def p_gw3d_single1d(self, pop_lambdas):
    r"""Computes :math:`p_gw(z, RA, Dec | \lambda)` when "king_p_gw3d='approximate'"."""
    p_gw1d, _ = self.p_gw1d(pop_lambdas) # (Nevents, ResGrids)
    p_gw3d = p_gw1d[:,None,:] * self.theta_gw_det.gw_loc2d_pdf[:,:,None] # (Nevents, MaxPixels, ResGrids)
    return p_gw3d

  ###################################
  # p_gw3d marginalized computation #
  ###################################

  def p_gw3d_many1d(self, pop_lambdas):
    """Computes p_gw(z, RA, Dec | λ) when kind_p_gw3d='marginalized'."""
    # Get source frame samples and population weights
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    norms = jnp.mean(weights, axis=-1)
    n_effs = jnp.sum(weights, axis=-1) ** 2 / jnp.sum(weights**2, axis=-1)

    # Single event routine
    def p_gw_single_event(ev):
      pe_pix = self.theta_gw_det.pixels_pe_opt_nside[ev]
      pixels = self.theta_gw_det.pixels_opt_nsides[ev]
      gw_pdf = self.theta_gw_det.gw_loc2d_pdf[ev]

      # Pre-compute FFT bandwidth once per event (same for all pixels)
      if self.kind_kde == 'many-1d-fft':
        fft_bw = scott_bw1d(th_src.z[ev], weights[ev]) if self.kde_bw is None else self.kde_bw
      else:
        fft_bw = None

      kde_fns = {
        'many-1d-fft': lambda z_pix, w_pix: fft_kde1d(
          self.z_grids[ev], z_pix, w_pix, bw=fft_bw, bin_edges=self.z_grids_edges[ev]
        ),
        'many-1d-binned': lambda z_pix, w_pix: binned_kde1d(
          self.z_grids[ev], z_pix, w_pix,
          scott_bw1d(z_pix, w_pix) if self.kde_bw is None else self.kde_bw,
          self.kernel, self.num_bins, self.cut_grid
        ),
      }
      compute_kde = kde_fns.get(
        self.kind_kde,
        lambda z_pix, w_pix: kde1d(
          z_pix, self.z_grids[ev], w_pix, kernel=self.kernel,
          bw=scott_bw1d(z_pix, w_pix) if self.kde_bw is None else self.kde_bw
        )
      )

      # Pixels loop
      def pixels_loop(pix, acc):
        pixel_mask = pe_pix == pixels[pix]
        w_pix = jnp.where(pixel_mask, weights[ev], 0.0)
        fill = jnp.nan if self.kind_kde == 'many-1d-fft' else 0.0
        z_pix = jnp.where(pixel_mask, th_src.z[ev], fill)
        kde = compute_kde(z_pix, w_pix)
        return acc.at[pix].set(kde * norms[ev] * gw_pdf[pix])

      return jax.lax.fori_loop(
        0, self.theta_gw_det.max_npixels, pixels_loop,
        jnp.zeros((self.theta_gw_det.max_npixels, self.z_int_res))
      )

    # Map over events and check Neff
    pgw = jax.vmap(p_gw_single_event)(jnp.arange(self.nevents))
    return jnp.where(
      n_effs[:, None, None] >= self.pe_neff,
      pgw,
      jnp.zeros((self.nevents, self.theta_gw_det.max_npixels, self.z_int_res))
    )

  ###########################
  # p_gw3d full computation #
  ###########################

  def p_gw3d_full_old(self, pop_lambdas):
    from .utils.math import numba_gkde_nd
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    norms  = jnp.mean(weights, axis = -1)
    n_effs = jnp.sum(weights, axis = -1)**2 / jnp.sum(weights**2, axis = -1)

    # Manage dataset
    dataset = jnp.array([th_src.z, self.theta_gw_det.ra, self.theta_gw_det.dec]) # dataset for kde, shape: (3, Nevents, Nsamples)
    dataset  = jnp.moveaxis(dataset, 0, 1) # shape (Nevents,3,Nsamples)

    # Prepare effective grid
    z_std = jnp.std(th_src.z, axis=1, keepdims=True)
    z_max = jnp.max(th_src.z, axis=1, keepdims=True)
    z_min = jnp.min(th_src.z, axis=1, keepdims=True)
    z_masks = (self.z_grids <= z_max + self.cut_grid*z_std) & (self.z_grids >= z_min - self.cut_grid*z_std)

    # Package everything into a single object to pass into callback0/tree?
    callback_input = (dataset, weights, norms, n_effs, z_masks, self.z_grids)

    def all_events_callback(args):
      dataset_np, weights_np, norms_np, n_effs_np, z_masks_np, z_grids_np = map(np.asarray, args)
      result = np.zeros((self.nevents, self.theta_gw_det.max_npixels, self.z_int_res))
      for ev in range(self.nevents):
        if n_effs_np[ev] < self.pe_neff:
          continue
        z_grid = z_grids_np[ev]
        z_mask = z_masks_np[ev]
        z_eff_grid = z_grid[z_mask]
        npix = int(self.theta_gw_det.neff_pixels[ev])
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
      jax.ShapeDtypeStruct((self.nevents, self.theta_gw_det.max_npixels, self.z_int_res), jnp.float64),
      callback_input
    )

  def p_gw3d_full(self, pop_lambdas):
    r"""Computes :math:`p_gw(z, RA, Dec | \lambda)` when "king_p_gw3d='full'"."""
    # Get source frame samples and population weights
    th_src, weights = get_theta_src_and_weights(pop_lambdas, self.theta_gw_det)
    norms  = jnp.mean(weights, axis = -1)
    n_effs = jnp.sum(weights, axis = -1)**2 / jnp.sum(weights**2, axis = -1)
    dataset = jnp.array([th_src.z, self.theta_gw_det.ra, self.theta_gw_det.dec]) # dataset for kde, shape: (3, Nevents, Nsamples)
    dataset  = jnp.moveaxis(dataset, 0, 2) # shape (Nevents,Nsamples,3)

    # Single event routine
    def p_gw_single_event(ev):
      pgw_ev = fft_kde3d(self.z_grids[ev], self.theta_gw_det.ra_grids[ev], self.theta_gw_det.dec_grids[ev],
          dataset[ev], weights=weights[ev],
          bin_edges=[self.z_grids_edges[ev], self.theta_gw_det.ra_edges[ev], self.theta_gw_det.dec_edges[ev]]
      )*norms[ev]
      # Pixels loop
      def pixels_loop(pix, acc):
        ra_idx, dec_idx = self.theta_gw_det.pixel_indices[ev, pix]
        return acc.at[pix].set(pgw_ev[:, ra_idx, dec_idx])
      return jax.lax.fori_loop(0, self.theta_gw_det.max_npixels, pixels_loop, jnp.zeros((self.theta_gw_det.max_npixels, self.z_int_res)))

    # Map over events and check Neff
    p_gw = jax.vmap(p_gw_single_event)(jnp.arange(self.nevents))
    return jnp.where(n_effs[:,None,None]>self.pe_neff, p_gw, jnp.zeros((self.nevents, self.theta_gw_det.max_npixels, self.z_int_res)))

  ####################################
  # likelihood numerator computation #
  ####################################

  def _compute_like_num_evs_pixelated(self, pop_lambdas):

    # p_gw(z, ra, dec | \theta_gw, \lambda_c, \lambda_m)
    p_gw3d = self.p_gw3d(pop_lambdas) # (Nevents, MaxPixels, ResGrids)
    p_gw3d = jnp.where(jnp.isnan(p_gw3d), jnp.zeros_like(p_gw3d), p_gw3d) # avoid fake pixels

    # p_z of having a cbc at z
    p_z = p_cbc(pop_lambdas, self.z_grids) # (Nevents, MaxPixels, ResGrids)

    # jacobian
    jacobian = ddLdz_at_z(pop_lambdas.cosmo, self.z_grids) * (1.+self.z_grids)**2

    # Integral
    integrand = p_gw3d * p_z / jacobian[:,None,:] # (Nevents, MaxPixels, ResGrids) -> integrand
    like_num_evs_pixels = trapz(integrand, self.z_grids[:,None,:], axis = -1) # (Nevents, MaxPixels) -> integrate on z_grids
    like_num_evs = jnp.sum(like_num_evs_pixels, axis = -1) # (Nevents,) ->  Sum pixel contributions

    return like_num_evs

  def _compute_like_num_evs_no_pixels(self, pop_lambdas):

    # p_gw(z,| \theta_gw, \lambda_c, \lambda_m)
    p_gw, z_grids = self.p_gw1d(pop_lambdas)

    # p_z of having a cbc at z
    p_z = p_cbc(pop_lambdas, z_grids)

    # jacobian
    jacobian = ddLdz_at_z(pop_lambdas.cosmo, z_grids) * (1.+z_grids)**2

    # Integral
    like_num_evs = trapz(p_gw*p_z/jacobian, z_grids, axis = -1)

    return like_num_evs

  ################################
  # hyper-likelihood computation #
  ################################

  @partial(jax.jit, static_argnums=(0,))
  def compute_all(self, **hyper_lambdas):
    pop_lambdas = self.population.update(**hyper_lambdas)
    log_like_num_evs = jnp.log(self.compute_like_num_evs(pop_lambdas))
    N_exp, dNdtheta, xi = self.selection_function.N_exp(pop_lambdas, ret_dNdtheta_and_xi=True)
    if not pop_lambdas.scale_free:
      log_like_num_evs += jnp.log(pop_lambdas.R0*pop_lambdas.Tobs)
      log_like_evs = log_like_num_evs  - N_exp/self.nevents
    else:
      log_like_evs = log_like_num_evs - jnp.log(N_exp)
    log_hyperlike = jnp.sum(log_like_evs, axis=-1)

    # Check Neff
    if self.inj_neff is not None:
      variance2 = jnp.sum((dNdtheta)**2, axis = -1) / self.selection_function.N_inj**2  - xi**2 / self.selection_function.N_inj
      neff = xi**2 / variance2
      neff_cond = neff > self.inj_neff
      log_hyperlike = jnp.where(neff_cond, log_hyperlike, -jnp.inf)
    else:
      neff = None
    # Filter nan/-inf to very small number
    log_hyperlike = jnp.nan_to_num(log_hyperlike, nan=-jnp.inf)
    return log_like_evs, N_exp, neff, log_hyperlike

  @partial(jax.jit, static_argnums=(0,))
  def compute_log_hyperlike(self, **hyper_lambdas):
    """Computes the of the log hyper-likelihood."""
    _, _, _, log_hyperlike = self.compute_all(**hyper_lambdas)
    return log_hyperlike

  @partial(jax.jit, static_argnums=(0,))
  def __call__(self, **hyper_lambdas):
    return self.compute_log_hyperlike(**hyper_lambdas)
