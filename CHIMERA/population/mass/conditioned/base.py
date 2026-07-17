import jax.numpy as jnp
import equinox as eqx
from typing import Union, List
from numbers import Number
from plum import dispatch

from CHIMERA.utils.math import cumtrapz, trapz
from CHIMERA.data import theta_src
from ..core import tpl_notnorm, high_pass_filter

################
# MASS PYTREES #
################

class base_mass_conditioned_struct(eqx.Module):
  m_low: float
  m_high: float
  m_grid_res: int = eqx.field(static=True, default=1000)
  m_grid: jnp.ndarray
  cdf_m2_conditioned: jnp.ndarray
  norm_p_m1: float
  default = {'m_low':4.58, 'm_high':86.3, }
  keys: List[str] = eqx.field(static=True)
  name = 'base_mass_conditioned_struct'

  def __init__(self, **kwargs):
    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    # normalizazions
    setattr(self, 'm_grid', jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res))
    p_values = secondary_mass_conditioned_pdf_notnorm(self, self.m_grid, self.m_high)
    cdf_values = cumtrapz(p_values, self.m_grid)
    setattr(self, 'cdf_m2_conditioned', cdf_values)
    integrand_values = primary_mass_pdf_notnorm(self, self.m_grid)
    norm = trapz(integrand_values, x=self.m_grid)
    setattr(self, 'norm_p_m1', norm)

  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    return self.__class__(**fiducials)

#########################
# primary mass function #
#########################

@dispatch
def primary_mass_pdf_notnorm(mass:base_mass_conditioned_struct, m: jnp.ndarray):
  raise ValueError("Primary mass function not implemented for base_mass_paired_struct")

# Utilities
#
@dispatch
def p_m1(mass:base_mass_conditioned_struct, m: jnp.ndarray):
  norm = jnp.maximum(mass.norm_p_m1, jnp.finfo(mass.norm_p_m1.dtype).eps)
  return primary_mass_pdf_notnorm(mass, m)/ norm


@dispatch
def p_m1(mass:base_mass_conditioned_struct, theta:theta_src):
  return p_m1(mass, theta.m1src)

##########################################
# secondary mass function not normalized #
##########################################

@dispatch
def secondary_mass_conditioned_pdf_notnorm(mass:base_mass_conditioned_struct, m2:jnp.ndarray, m1:Union[Number,jnp.ndarray]):
  pdf = tpl_notnorm(m2, mass.beta, mass.m_low, m1)
  pdf *= high_pass_filter(m2, mass.delta_m, mass.m_low)
  return pdf

#######################
# Joint mass fucntion #  --> Principal function used in the analysis!
#######################

@dispatch
def p_m1m2(mass:base_mass_conditioned_struct, m1:jnp.ndarray, m2:jnp.ndarray):
  pm1 = p_m1(mass, m1)
  pm2m1 = secondary_mass_conditioned_pdf_notnorm(mass, m2, m1)
  pm2m1_norm = jnp.interp(m1, mass.m_grid, mass.cdf_m2_conditioned)
  pm2m1_norm = jnp.maximum(pm2m1_norm, jnp.finfo(pm2m1_norm.dtype).eps) # to avoid 0./0. that may occur in the line below
  pm2m1 /= pm2m1_norm
  return pm1 * pm2m1

@dispatch
def p_m1m2(mass:base_mass_conditioned_struct, theta:theta_src):
  return p_m1m2(mass, theta.m1src, theta.m2src)

#################################################
# Primary/Secondary mass fucntions for plotting #
#################################################

def pdf_joint_and_marg(mass:base_mass_conditioned_struct, res=(5000,2500)):
  m1 = jnp.linspace(mass.m_low, mass.m_high, res[0])
  m2 = jnp.linspace(mass.m_low, mass.m_high, res[1])
  m1mesh, m2mesh = jnp.meshgrid(m1, m2)
  p_joint = p_m1m2(mass, m1mesh, m2mesh)
  p1_marg = trapz(p_joint, x=m2, axis=0)
  p1_marg /= trapz(p1_marg, x=m1)
  p2_marg = trapz(p_joint, x=m1, axis=1)
  p2_marg /= trapz(p2_marg, x=m2)
  dict_to_ret = {'m1':m1, 'm2':m2, 'm1mesh':m1mesh, 'm2mesh':m2mesh,
    'p_joint': p_joint, 'p_m1_marg': p1_marg, 'p_m2_marg': p2_marg}
  return dict_to_ret
