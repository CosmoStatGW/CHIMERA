import equinox as eqx
from typing import Optional, Dict, List
from numbers import Number

from ..utils.config import jax, jnp, logger, trapz
from ..utils import angles
from ..population.cosmo import dVcdz_at_z, Vc_at_z, z_from_dGW, ddLdz_at_z
from ..population.mass import pdf_m1m2
from ..population.rate import _dummy_rate, compute_merger_rate

class scale_param(_dummy_rate):
  R0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1)
  name = 'scale_param'
  keys = ['R0']

class population(object):

  def __init__(self,
    cosmo_obj: eqx.Module,
    mass_obj: eqx.Module,
    rate_obj: eqx.Module,
    galcat_obj: object,
    scale_free=True,
    R0fid = 1,
    Tobs: Number = 1,
    z_max: Number = 10):

    self.cosmo_obj = cosmo_obj
    self.mass_obj = mass_obj
    self.rate_obj = rate_obj
    self.galcat_obj = galcat_obj
    self.Tobs = Tobs
    self.zmax = z_max

    self.scale_free = scale_free
    self.R0fid = R0fid
    if self.scale_free and self.Tobs != 1:
      logger.info("Trying to set Tobs!=1 with scale-free likelihood, ignoring Tobs.")
      self.Tobs = 1
    if self.scale_free and self.Tobs != 1:
      logger.info("Trying to set R0!=1 with scale-free likelihood, forcing R0=1.")
      self.R0fid = 1
    self.scale_param = scale_param(R0 = self.R0fid)

    self.cosmo_keys = self.cosmo_obj.keys
    self.mass_keys = self.mass_obj.keys
    self.rate_keys = self.rate_obj.keys
    self.scale_keys = self.scale_param.keys

    self.fiducials_cosmo = self.cosmo_obj.as_dict
    self.fiducials_mass = self.mass_obj.as_dict
    self.fiducials_rate = self.rate_obj.as_dict
    self.fiducials_scale = self.scale_param.as_dict

  def update_params(self, **hyper_params):
    """Updates the cosmological, mass, rate, and scale-free parameters using the provided hyperparameters"""
    lc = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.cosmo_keys}
    lm = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.mass_keys}
    lr = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.rate_keys}
    ls = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.scale_keys}

    max_size = max(
        max(len(v) for v in lc.values()) if lc else 1,
        max(len(v) for v in lm.values()) if lm else 1,
        max(len(v) for v in lr.values()) if lr else 1,
        max(len(v) for v in ls.values()) if ls else 1,
    )

    for model_params, fiducials, keys in zip([lc, lm, lr, ls],
                                             [self.fiducials_cosmo, self.fiducials_mass, self.fiducials_rate, self.fiducials_scale],
                                             [self.cosmo_keys, self.mass_keys, self.rate_keys, self.scale_keys]):
      for key in keys:
        value = jnp.asarray(model_params.get(key, fiducials[key]))
        if value.size == 1 and max_size > 1:
          model_params[key] = jnp.full((max_size,), value)
        elif key not in model_params:
          model_params[key] = jnp.full((max_size,), fiducials[key])
        else:
          if model_params[key].size == 1:
            model_params[key] = jnp.full((max_size,), model_params[key])
        if key == 'R0' and self.scale_free: # force R0=1 if like is set to be scale_free for consistency
          model_params[key] = jnp.full((max_size,), 1)

    cosmo_params = self.cosmo_obj.set(**lc)
    mass_params  = self.mass_obj.set(**lm)
    rate_params  = self.rate_obj.set(**lr)
    scale_params = self.scale_param.set(**ls)

    return cosmo_params, mass_params, rate_params, scale_params

  def get_redshift_pe_and_pop_weights(self, pe_samples, pe_prior, **hyper_params):
    # update population hyperparams
    cosmo_params, mass_params, _, _ = self.update_params(**hyper_params)
    # kde dataset
    zs = z_from_dGW(cosmo_params, pe_samples['dL'], self.zmax)
    # kde weights and neff
    m1s, m2s = pe_samples['m1det']/(1.+zs), pe_samples['m2det']/(1.+zs)
    weights = pdf_m1m2(mass_params, m1s, m2s)/pe_prior
    return zs, weights

  def compute_pop_rate(self, data, prior, **hyper_params):
    """Computes the population rate"""
    cosmo_params, mass_params, rate_params, scale_params = self.update_params(**hyper_params)
    dL = data['dL']
    m1d = data["m1det"]
    m2d = data["m2det"]

    z = z_from_dGW(cosmo_params, dL, self.zmax)
    m1s, m2s = m1d/(1.+z), m2d/(1.+z)

    jacobian = ddLdz_at_z(cosmo_params, z, dL)*(1.+z)**3
    dNdtheta = pdf_m1m2(mass_params, m1s, m2s)
    dNdtheta *= compute_merger_rate(rate_params, z)
    dNdtheta *=  self.galcat_obj.p_bkg(cosmo_params, z, dL)
    dNdtheta /= (jacobian*prior)
    dNdtheta *= scale_params.R0
    return dNdtheta

  def compute_p_cbc(self, z_grids, **hyper_params):
    cosmo_params, mass_params, rate_params, scale_params = self.update_params(**hyper_params)
    p_gal = self.galcat_obj.compute_pgal(z_grids, **hyper_params)
    p_rate = compute_merger_rate(rate_params, z_grids)
    jacobian = (1.+z_grids)**3 * ddLdz_at_z(cosmo_params, z_grids)
    if p_gal.ndim > p_rate.ndim:
      p_z = jnp.where(p_gal != -100, p_gal * p_rate[:, None, :] / jacobian[:, None, :], -100) # pixelated
    else:
      p_z = p_gal * p_rate / jacobian # no pixels
    return p_z

  def compute_Ncbc(self, **hyper_params):
    """Computes the total number of expected CBC sources."""
    cosmo_params, mass_params, rate_params, scale_params = self.update_params(**hyper_params)
    zz    = jnp.linspace(*[0.001,self.zmax], 10_000)
    dN_dz = compute_merger_rate(rate_params, zz)/(1.+zz) * self.galcat_obj.p_bkg(cosmo_params, zz, None)
    dN_dz *= scale_params.R0
    res = trapz(dN_dz, x = zz, axis = -1)
    res *= self.Tobs
    return res
