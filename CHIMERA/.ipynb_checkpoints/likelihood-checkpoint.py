from .utils.jax_config import jax, jnp, xp, logger, trapz
from .utils.kde import kde1d, get_neff, binning1d, jax_gkde_nd, numba_gkde_nd
from .cosmo import dVcdz_at_dz, z_from_dGW, ddLdz_at_z, compute_z_grids
from .rate import compute_rate
from .mass import pdf_m1m2

import equinox as eqx
from typing import Optional, List, Dict, Union
from numbers import Number
from functools import partial
import numpy as np

class hyperlikelihood(object):
  def __init__(
    self,
    data_gw: Dict[str, jnp.ndarray],
    pe_prior: jnp.ndarray,
    cosmo_model: eqx.Module,
    mass_model: eqx.Module,
    rate_model: eqx.Module,
    galcat_model: object,
    bias_model: Optional[object] = None,
    kind_p_gw3d: Optional[str] = None,
    kernel: str = 'epan',
    bw_method: Optional[Number] = None,
    cut_grid: Number = 2.0,
    binning: bool = True,
    num_bins: int = 200,
    positive_weights_thresh: Number = 5.0,
    data_neff: Number = 5.0,
    cosmo_prior: Optional[Dict[str, List[Number]]] = None,
    z_conf_range: Union[Number, list] = 5.0,
    z_int_res: int = 300,
  ):
    # Initialize attributes
    self.data_gw = data_gw
    self.pe_prior = pe_prior
    self.cosmo_model = cosmo_model
    self.mass_model = mass_model
    self.rate_model = rate_model
    self.galcat_model = galcat_model
    self.bias_model = bias_model
    self.kind_p_gw3d = kind_p_gw3d
    self.kernel = kernel
    self.bw_method = bw_method
    self.cut_grid = cut_grid
    self.binning = binning
    self.num_bins = num_bins
    self.positive_weights_thresh = positive_weights_thresh
    self.data_neff = data_neff
    self.cosmo_prior = cosmo_prior
    self.z_conf_range = z_conf_range
    self.z_int_res = z_int_res

    # Get some useful attributes
    self.pixelated = 'pixels' in self.data_gw
    self.events_pe = self.data_gw['pe_data'] if self.pixelated else self.data_gw
    self.nevents = len(self.events_pe['dL'])
    self.cosmo_keys = self.cosmo_model.cosmo_keys
    self.mass_keys = self.mass_model.mass_keys
    self.rate_keys = self.rate_model.rate_keys
    self.fiducials_cosmo = self.cosmo_model.fiducial_params
    self.fiducials_mass = self.mass_model.fiducial_params
    self.fiducials_rate = self.rate_model.fiducial_params

    # Handle pixelated likelihood
    if self.pixelated:
      assert self.kind_p_gw3d in ['approximate', 'marginalized', 'full'], \
        "`kind_p_gw3d` must be one of `approximate`, `marginalized`, or `full`"
      self.max_npixels = self.data_gw['pixels'].shape[1]
      self.compute_numlike_evs = self._compute_numlike_evs_pixelated
      self.p_gw3d = {
        'approximate': self.p_gw3dapprox,
        'marginalized': self.p_gw3dmarg,
        'full': self.p_gw3dfull
      }[self.kind_p_gw3d]
      if self.kind_p_gw3d == 'full': logger.info("`king_p_gw3d` has been set to `full`. Only available kernel is `gaussian`. The `binning` option is not available.")
      self.neff_pixels = jnp.array([ self.data_gw['ra_pix'][ev][self.data_gw['ra_pix'][ev]!=-100.].shape[0] for ev in range(self.nevents)])
    else:
      self.compute_numlike_evs = self._compute_numlike_evs_no_pixels

    # Precompute redshift grids
    logger.info(f'Pre-computing redshift grids')
    self.zgrids = compute_z_grids(
      self.cosmo_model,
      self.events_pe['dL'],
      self.cosmo_prior,
      self.z_conf_range,
      self.z_int_res,
    )
    self.zmax = float(jnp.max(self.zgrids)) * (1.0 + 0.1)

    # Precompute p_cat on the redshift grids
    logger.info(f'Pre-computing `p_cat` on redshift grids')
    self.galcat_model.precompute_pcat(self.zgrids)

    # Logging the creation of the model
    logger.info(f'Created hyperlikelihood model. Using {self.nevents} GW events')

  def update_models(self, **hyper_params):
    lc = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.cosmo_keys}
    lm = {k: jnp.atleast_1d(v)  for k, v in hyper_params.items() if k in self.mass_keys}
    lr = {k: jnp.atleast_1d(v)  for k, v in hyper_params.items() if k in self.rate_keys}
    max_size = max(
        max(len(v) for v in lc.values()) if lc else 1,
        max(len(v) for v in lm.values()) if lm else 1,
        max(len(v) for v in lr.values()) if lr else 1,
    )
    for model_params, fiducials, keys in zip([lc, lm, lr],
                                             [self.fiducials_cosmo, self.fiducials_mass, self.fiducials_rate],
                                             [self.cosmo_keys, self.mass_keys, self.rate_keys]):
      for key in keys:
        value = jnp.asarray(model_params.get(key, fiducials[key]))
        if value.size == 1 and max_size > 1:
          model_params[key] = jnp.full((max_size,), value)
        elif key not in model_params:
          model_params[key] = jnp.full((max_size,), fiducials[key])
        else:
          if model_params[key].size == 1:
            model_params[key] = jnp.full((max_size,), model_params[key])
    cosmo_model = self.cosmo_model.from_params(**lc)
    mass_model  = self.mass_model.from_params(**lm)
    rate_model  = self.rate_model.from_params(**lr)
    return cosmo_model, mass_model, rate_model

  ####################
  # p_gw computation #
  ####################

  @partial(jax.jit, static_argnums=(0,))
  def p_gw1d(self, **hyper_params):
    # update population hyperparams
    cosmo_model, mass_model, _ = self.update_models(**hyper_params)
    # kde dataset
    _zs = z_from_dGW(cosmo_model, self.events_pe['dL'], self.zmax)
    # kde weights and neff
    m1s, m2s = self.events_pe['m1det']/(1.+_zs), self.events_pe['m2det']/(1.+_zs)
    _weights = pdf_m1m2(mass_model, m1s, m2s)/self.pe_prior
    norms = jnp.mean(_weights, axis = -1)
    sum_w = jnp.sum(_weights>0, axis = -1) # shape (Nevents,)
    n_eff = jax.vmap(get_neff)(_weights)   # shape (Nevents,)
    # optional binning
    if self.binning:
      zs, weights = jax.vmap(binning1d, in_axes=(0,0,None))(_zs, _weights, self.num_bins)
    else:
      zs, weights = _zs, _weights
    # process 1 event by checking if n_eff and sum_w are good
    def _pgw1ev(idx):
      condition = (n_eff[idx] >= self.data_neff) & (sum_w[idx] >= self.positive_weights_thresh)
      return jax.lax.cond(
        condition,
        lambda _: kde1d(
          zs[idx], self.zgrids[idx], weights[idx], self.kernel, self.bw_method, self.cut_grid
        )*norms[idx],
        lambda _: jnp.zeros(self.z_int_res),  # No update if the condition is not met
        operand=None
      )
    # v-mapping
    p_gw = jax.vmap(_pgw1ev)(jnp.arange(self.nevents))
    return p_gw

  @partial(jax.jit, static_argnums=(0,))
  def p_gw3dapprox(self, **hyper_params):
    p_gw1d = self.p_gw1d(**hyper_params) # (Nevents, ResGrids)
    p_gw3d = p_gw1d[:,None,:] * self.data_gw['gw_loc2d_pdf'][:,:,None] # (Nevents, MaxPixels, ResGrids)
    return p_gw3d

  #@partial(jax.jit, static_argnums=(0,))
  def p_gw3dmarg(self):
    pass # to be implemented

  # uses numba on cpu only
  def p_gw3dfull(self, **hyper_params):
    # update population models
    cosmo_model, mass_model, _ = self.update_models(**hyper_params)
    # kde datasets
    zs = z_from_dGW(cosmo_model, self.events_pe['dL'], self.zmax)
    _dataset = jnp.array([zs, self.events_pe['ra'],  self.events_pe['dec']]) # dataset for kde, shape: (3, Nevents, Nsamples)
    dataset  = jnp.moveaxis(_dataset, 0, 1) # shape (Nevents,3,Nsamples)
    # kde weights and n_eff
    m1s, m2s = self.events_pe['m1det']/(1.+zs), self.events_pe['m2det']/(1.+zs)
    weights = pdf_m1m2(mass_model, m1s, m2s)/self.pe_prior
    n_eff = jax.vmap(get_neff)(weights)   # shape (Nevents,)
    norms = jnp.mean(weights, axis = -1)
    sum_w = jnp.sum(weights>0, axis = -1) # shape (Nevents,)
    # prepare effective grid
    z_std, z_max, z_min = jnp.std(zs, axis=1)[:,None],  jnp.max(zs, axis=1)[:,None], jnp.min(zs, axis=1)[:,None]
    z_grids = self.zgrids
    masks_z = (z_grids <= z_max + self.cut_grid*z_std) & (z_grids >= z_min - self.cut_grid*z_std)
    neff_pixels = self.neff_pixels
    # load to cpu
    dataset = xp.asarray(dataset)
    norms   = xp.asarray(norms)
    weights = xp.asarray(weights)
    p_gw3d  = xp.zeros((self.nevents, self.max_npixels, self.z_int_res))
    z_grids = xp.asarray(z_grids)
    for ev in range(self.nevents):
      # check neff or weights
      if (sum_w[ev]<5) or (n_eff[ev]<self.data_neff):
        # p_gw3d is already zero everywhere
        continue
      # mask
      mask_z = masks_z[ev]
      npix = neff_pixels[ev]
      eff_mask = np.tile(mask_z, npix)
      # effective grid
      z_grid_eff = z_grids[ev][mask_z]
      ra_pix  = xp.asarray(self.data_gw['ra_pix'][ev, 0:npix])
      dec_pix = xp.asarray(self.data_gw['dec_pix'][ev, 0:npix])
      eff_grid = xp.array([xp.tile(z_grid_eff, npix),
                           xp.hstack([xp.full_like(z_grid_eff, x) for x in ra_pix]),
                           xp.hstack([xp.full_like(z_grid_eff, x) for x in dec_pix])])
      # compute
      _p_gw = xp.zeros(shape=(npix*self.z_int_res))
      _p_gw[eff_mask] = numba_gkde_nd(dataset[ev], eff_grid, weights=weights[ev], bw_method=self.bw_method, in_log=False)
      p_gw3d[ev,0:npix,:] = _p_gw.reshape(npix,self.z_int_res) * norms[None,ev]
    return jnp.asarray(p_gw3d)

  ####################################
  # likelihood numerator computation #
  ####################################

  def _compute_numlike_evs_pixelated(self, **hyper_params):
    # update population params
    cosmo_model, mass_model, rate_model = self.update_models(**hyper_params)
    # p_gw(z, ra, dec | \theta_gw, \lambda_c, \lambda_m)
    p_gw3d = self.p_gw3d(**hyper_params) # (Nevents, MaxPixels, ResGrids)
    # p_z of having a cbc at z
    p_gal  = self.galcat_model.compute_pgal(self.zgrids, **hyper_params) # (Nevents, MaxPixels, ResGrids)
    p_rate = compute_rate(rate_model, self.zgrids) # (Nevents, ResGrids)
    jacobian = (1.+self.zgrids)**3 * ddLdz_at_z(cosmo_model, self.zgrids) # (Nevents, ResGrids)
    p_z = p_gal*p_rate[:,None,:]/jacobian[:,None,:] # (Nevents, MaxPixels, ResGrids)
    # Integral in all pixels, but avoid the `fake` ones
    integrand = jnp.where(p_gal != -100,
                          p_gw3d * p_z,
                          jnp.zeros((self.nevents, self.max_npixels, self.z_int_res))
                          ) # (Nevents, MaxPixels, ResGrids)
    like_evs_pixels = trapz(integrand, self.zgrids[:,None,:], axis = -1) # (Nevents, MaxPixels)
    # Sum pixel contributions
    like_evs = jnp.sum(like_evs_pixels, axis = -1) # (Nevents,)
    return like_evs

  def _compute_numlike_evs_no_pixels(self, **hyper_params):
    # update population params
    cosmo_model, mass_model, rate_model = self.update_models(**hyper_params)
    # p_gw(z,| \theta_gw, \lambda_c, \lambda_m)
    p_gw = self.p_gw1d(ret_normed=True, **hyper_params)
    # p_z of having a cbc at z
    prate    = compute_rate(rate_model, self.zgrids)
    jacobian = (1.+self.zgrids)**3 * ddLdz_at_z(cosmo_model, self.zgrids)
    pgal = self.galcat_model.compute_pgal(self.zgrids, **hyper_params)
    p_z  = pgal*prate / jacobian
    # Integral
    like_evs = trapz(p_gw * p_z, self.zgrids, axis = -1)
    return like_evs

  def compute_log_likenum(self, **hyper_params):
    log_like_evs = jnp.log(self.compute_numlike_evs(**hyper_params))
    finite       = jnp.isfinite(log_like_evs)
    log_like_evs = jnp.where(finite, log_like_evs, -1000)
    log_num      = jnp.sum(log_like_evs, axis = -1)
    n_eff_ev     = jnp.sum(finite, axis = -1)
    return log_num, n_eff_ev

  ################################
  # hyper-likelihood computation #
  ################################

  def compute_log_hyperlike(self, **hyper_params):
    log_like_num, n_eff_ev = self.compute_log_likenum(**hyper_params)
    log_bias = jnp.log(self.bias_model(**hyper_params))
    return log_like_num - n_eff_ev*log_bias

  def __call__(self, **hyper_params):
    return self.compute_log_hyperlike(**hyper_params)
