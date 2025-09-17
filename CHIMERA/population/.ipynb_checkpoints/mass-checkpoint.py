from ..utils.config import jax, jnp, trapz
from ..utils.kde import cumtrapz
import equinox as eqx
from typing import Union, Tuple
from plum import dispatch
from functools import partial

################
# MASS PYTREES #
################

class base_mass_struct(eqx.Module):
  default = {}
  keys = list(default.keys())
  name = 'base_mass_struct'

  def __init__(self, **kwargs):
    for key in self.default.keys():
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}
  def update(self, **kwargs):
    fiducials = self.as_dict
    fiducials.update(kwargs)
    return self.__class__(**fiducials)

class tpl(base_mass_struct):
  r"""A class to describe a truncated power law mass model implemented as an Equinox module.

  Args:
    alpha (float, optional): The slope of the power law at high masses. Default is 3.4.
    beta (float, optional): The slope of the power law at low masses. Default is 1.1.
    m_low (float, optional): The lower truncation mass. Default is 5.1.
    m_high (float, optional): The upper truncation mass. Default is 87.

  Class Attributes:
    - name (str): The name of the model, set to 'truncated_power_law'.
    - keys (List[str]): A list of parameter names used in the model, specifically ['alpha', 'beta', 'm_low', 'm_high'].

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary.

  Methods:
    - update(**kwargs) (tpl): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  alpha: float
  beta: float
  m_low: float
  m_high: float
  default = {'alpha':2.5, 'beta':1.1, 'm_low':5.1, 'm_high':87.}
  keys = list(default.keys())
  name = 'truncated_power_law'

class bpl(base_mass_struct):
  r"""A class to describe a broken power law mass model implemented as an Equinox module.

  Args:
    alpha_1 (float, optional): The slope of the power law for masses below the break. Default is 1.6.
    alpha_2 (float, optional): The slope of the power law for masses above the break. Default is 5.6.
    beta (float, optional): The slope of the power law at intermediate masses. Default is 1.1.
    delta_m (float, optional): The characteristic mass difference that influences the break. Default is 4.8.
    m_low (float, optional): The lower truncation mass. Default is 5.1.
    m_high (float, optional): The upper truncation mass. Default is 87.
    break_fraction (float, optional): The fraction of the mass function where the break occurs. Default is 0.43.

  Class Attributes:
    - name (str): The name of the model, set to 'broken_power_law'.
    - keys (List[str]): A list of parameter names used in the model, specifically ['alpha_1', 'alpha_2', 'beta', 'delta_m', 'm_low', 'm_high', 'break_fraction'].

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary, where keys are parameter names
      and values are their corresponding `float` values.

  Methods:
    - update(**kwargs) (bpl): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter provided in the input.
  """
  alpha_1: float
  alpha_2: float
  beta: float
  delta_m: float
  m_low: float
  m_high: float
  break_fraction: float
  default = {'alpha_1':1.6, 'alpha_2':5.6, 'beta':1.1, 'delta_m':4.8, 'm_low':5.1, 'm_high':87., 'break_fraction':0.43}
  keys = list(default.keys())
  name = 'broken_power_law'

class plp(base_mass_struct):
  r"""A class to describe a power law mass model plus Gaussian peak, implemented as an Equinox module.

  Args:
    lambda_peak (float, optional): The strength of the Gaussian peak. Default is 0.039.
    alpha (float, optional): The slope of the power law for masses below the break. Default is 3.4.
    beta (float, optional): The slope of the power law for masses above the break. Default is 1.1.
    delta_m (float, optional): The characteristic mass difference that influences the break. Default is 4.8.
    m_low (float, optional): The lower truncation mass. Default is 5.1.
    m_high (float, optional): The upper truncation mass. Default is 87.
    mu_g (float, optional): The mean of the Gaussian peak. Default is 34.
    sigma_g (float, optional): The standard deviation of the Gaussian peak. Default is 3.6.

  Class Attributes:
    - name (str): The name of the model, set to 'power_law_plus_peak'.
    - keys (List[str]): A list of parameter names used in the model, specifically ['lambda_peak', 'alpha', 'beta', 'delta_m', 'm_low', 'm_high', 'mu_g', 'sigma_g'].

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary, where keys are parameter names
      and values are their corresponding `float` values.

  Methods:
    - update(**kwargs) (plp): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter provided in the input.
  """
  lambda_peak: float
  alpha: float
  beta: float
  delta_m: float
  m_low: float
  m_high: float
  mu_g: float
  sigma_g: float
  default = {'lambda_peak':0.039, 'alpha':3.4, 'beta':1.1, 'delta_m':4.8, 'm_low':5.1, 'm_high':87., 'mu_g':34., 'sigma_g':3.6}
  keys = list(default.keys())
  name = 'power_law_plus_peak'

class pl2p(base_mass_struct):
  r"""A class to describe a power law mass model with two Gaussian peaks, implemented as an Equinox module.

  Args:
    lambda_peak (float, optional): The strength of the first Gaussian peak. Default is 0.05.
    lambda1 (float, optional): The strength of the second Gaussian peak. Default is 0.5.
    alpha (float, optional): The slope of the power law for masses below the break. Default is 2.9.
    beta (float, optional): The slope of the power law for masses above the break. Default is 0.9.
    delta_m (float, optional): The characteristic mass difference that influences the break. Default is 4.8.
    m_low (float, optional): The lower truncation mass. Default is 4.6.
    m_high (float, optional): The upper truncation mass. Default is 87.
    mu1_g (float, optional): The mean of the first Gaussian peak. Default is 33.
    sigma1_g (float, optional): The standard deviation of the first Gaussian peak. Default is 3.
    mu2_g (float, optional): The mean of the second Gaussian peak. Default is 68.
    sigma2_g (float, optional): The standard deviation of the second Gaussian peak. Default is 3.

  Class Attributes:
    - name (str): The name of the model, set to 'power_law_plus_double_peak'.
    - keys (List[str]): A list of parameter names used in the model, specifically ['lambda_peak', 'lambda1', 'alpha', 'beta', 'delta_m', 'm_low', 'm_high', 'mu1_g', 'sigma1_g', 'mu2_g', 'sigma2_g'].

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary, where keys are parameter names
      and values are their corresponding `float` values.

  Methods:
    - update(**kwargs) (pl2p): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter provided in the input.
  """
  lambda_peak: float
  lambda1: float
  alpha: float
  beta: float
  delta_m: float
  m_low: float
  m_high: float
  mu1_g: float
  sigma1_g: float
  mu2_g: float
  sigma2_g: float
  default = {'lambda_peak':0.05, 'lambda1':0.5, 'alpha':3.4, 'beta':1.1, 'delta_m':4.8, 'm_low':5.1, 'm_high':87., 'mu1_g':34., 'sigma1_g':3.6, 'mu2_g':68, 'sigma2_g':3}
  keys = list(default.keys())
  name = 'power_law_plus_double_peak'

# Semi-paramtrice PL+spline model: slightly modified w.r.t base_struct_mass to handle spline_basis and spline_coeffs
class pls(base_mass_struct):
  alpha: float
  m_low: float
  m_high: float
  delta_m: float
  beta: float
  spline_coeffs: jnp.ndarray
  spline_basis: jnp.ndarray
  num_knots: int

  default = {
  'alpha': 3.4,
  'm_low': 5.,
  'm_high': 87.,
  'delta_m': 4.8,
  'beta': 1.1,
  }
  keys = list(default.keys()) + ['spline_coeffs', 'spline_basis']
  name = 'powerlaw_plus_spline'

  def __init__(self, **kwargs):
    for key in self.default.keys():
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)

    self.spline_basis = kwargs['spline_basis'] # needed
    self.num_knots = self.spline_basis.shape[1] - 2
    self.spline_coeffs = jnp.zeros(self.num_knots+2)
    _spline_coeffs = kwargs.get('spline_coeffs', jnp.zeros(self.num_knots))
    self.spline_coeffs = self.spline_coeffs.at[1:-1].set(_spline_coeffs)

  def update(self, **kwargs):
    fiducials = self.as_dict
    # handle spline_coeffs
    fiducials['spline_coeffs'] = fiducials['spline_coeffs'][1:-1]
    fiducials.update(kwargs)
    return self.__class__(**fiducials)

#######################
# Core mass functions #
#######################

# Truncated power law PDF (not normalized) and its (analytical) CDF
@jax.jit
def tpl_notnorm(m, alpha, m_low, m_high):
  # not normalized
  return jnp.where((m_low <= m) & (m <= m_high),
    m**alpha,
    0.
  )

@jax.jit
def tpl_cdf(alpha, m_low, m):
  # not normalized. if m = m_high the result is the normalization of the pdf
  return jnp.where(alpha==-1,
    jnp.log(m_low) - jnp.log(m),
    (m**(1 + alpha) - m_low**(1 + alpha)) / (1 + alpha)
  )

# Smoothing function
@jax.jit
def smoothing(m, delta_m, m_low):
  eps = 1.e-99
  log_smoothing = jnp.where(m < m_low,
    -jnp.inf,
    jnp.where(m > (m_low + delta_m),
      0.0,
      -jnp.logaddexp(0.0, (delta_m/(m-m_low+eps) + delta_m/(m-m_low-delta_m+eps)))
    )
  )
  return jnp.exp(log_smoothing)

# Gaussian distributions
@jax.jit
def gaussian(x, mu, sigma):
  log_G = -0.5*jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x-mu)**2/(2.*sigma**2)
  return jnp.exp(log_G)

@jax.jit
def truncated_gaussian(x, mu, sigma, x_min, x_max):
  max_point = (x_max-mu)/(sigma*jnp.sqrt(2.))
  min_point = (x_min-mu)/(sigma*jnp.sqrt(2.))
  norm = 0.5*jax.scipy.special.erf(max_point)-0.5*jax.scipy.special.erf(min_point)
  # trunc gaussian
  return jnp.where( (x_min <= x) & (x <= x_max),
    gaussian(x, mu, sigma) / norm,
    0.
  )

#########################################
# primary mass functions not normalized #
#########################################

@dispatch
@jax.jit
def primary_mass_pdf_notnorm(mass:tpl, m: jnp.ndarray):
  return tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)

@dispatch
@jax.jit
def primary_mass_pdf_notnorm(mass:bpl, m:jnp.ndarray):
  m_break = mass.m_low + mass.break_fraction * (mass.m_high - mass.m_low)
  pdf = jnp.where(m <= m_break,
    tpl_notnorm(m, -mass.alpha_1, mass.m_low, m_break),
    tpl_notnorm(m, -mass.alpha_2, m_break, mass.m_high)
  )
  pdf *= smoothing(m, mass.delta_m, mass.m_low),
  return pdf

@dispatch
@jax.jit
def primary_mass_pdf_notnorm(mass:plp, m: jnp.ndarray):
  P = tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)/tpl_cdf(-mass.alpha, mass.m_low, mass.m_high)
  G = truncated_gaussian(m, mass.mu_g, mass.sigma_g, mass.m_low, mass.mu_g + 5*mass.sigma_g)
  pdf = (1 - mass.lambda_peak)*P + mass.lambda_peak*G
  pdf *= smoothing(m, mass.delta_m, mass.m_low)
  return pdf

@dispatch
@jax.jit
def primary_mass_pdf_notnorm(mass:pl2p, m: jnp.ndarray):
  P  = tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)/tpl_cdf(-mass.alpha, mass.m_low, mass.m_high)
  G1 = truncated_gaussian(x, mass.mu1_g, mass.sigma1_g, mass.m_low, mass.mu1_g + 5*mass.sigma1_g)
  G2 = truncated_gaussian(x, mass.mu2_g, mass.sigma2_g, mass.m_low, mass.mu2_g + 5*mass.sigma2_g)
  pdf = (1-mass.lambda_peak)*P + mass.lambda_peak*mass.lambda1*G1 + mass.lambda_peak*(1. - mass.lambda1)*G2
  pdf *= smoothing(m2, mass.delta_m, mass.m_low)
  return pdf

###########################################
# secondary mass functions not normalized #
###########################################

@dispatch
@jax.jit
def secondary_mass_conditioned_pdf_notnorm(mass:tpl, m2:jnp.ndarray, m1:jnp.ndarray):
  return tpl_notnorm(m2, mass.beta, mass.m_low, m1)

@dispatch
@jax.jit
def secondary_mass_conditioned_pdf_notnorm(mass:Union[bpl,plp,pl2p], m2:jnp.ndarray, m1:jnp.ndarray):
  pdf = tpl_notnorm(m2, mass.beta, mass.m_low, m1)
  pdf *= smoothing(m2, mass.delta_m, mass.m_low)
  return pdf

#########################################
# primary mass functions normalizations #
#########################################

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def primary_mass_normalization(mass:tpl, res:int=1000):
  return tpl_cdf(-mass.alpha, mass.m_low, mass.m_high)

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def primary_mass_normalization(mass:Union[bpl,plp,pl2p], res:int=1000):
  mm = jnp.logspace(jnp.log10(mass.m_low), jnp.log10(mass.m_high), res)
  integrand_values = primary_mass_pdf_notnorm(mass, mm)
  norm = trapz(integrand_values, x=mm)
  return norm

###########################################
# secondary mass functions normalizations #
###########################################

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def secondary_mass_conditioned_normalization(mass:tpl, m1:jnp.ndarray, res:int=1000):
  return tpl_cdf(mass.beta, mass.m_low, m1)

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def secondary_mass_conditioned_normalization(mass:Union[bpl,plp,pl2p], m1:jnp.ndarray, res:int=1000):
  m_grid = jnp.logspace(jnp.log10(mass.m_low), jnp.log10(mass.m_high), res)
  p_values = secondary_mass_conditioned_pdf_notnorm(mass, m_grid, mass.m_high)
  cdf_values = cumtrapz(p_values, m_grid)
  return jnp.interp(m1, m_grid, cdf_values)

#######################
# Joint mass fucntion #
#######################

@partial(jax.jit, static_argnames = ['res'])
def pdf_m1m2(mass:base_mass_struct, m1:jnp.ndarray, m2:jnp.ndarray, res:int = 1000):
  p_m1 = primary_mass_pdf_notnorm(mass, m1)
  p_m1 /= primary_mass_normalization(mass, res)
  p_m2m1 = secondary_mass_conditioned_pdf_notnorm(mass, m2, m1)
  p_m2m1 /= secondary_mass_conditioned_normalization(mass, m1, res)
  p_m2m1 = jnp.where(jnp.isnan(p_m2m1), 0., p_m2m1)  # treat 0./0. that may have occured the line above
  return p_m1 * p_m2m1

#################################################
# Primary/Secondary mass fucntions for plotting #
#################################################

def pdf_joint_and_marg(mass, res=(5000,2500)):
  m1 = jnp.linspace(mass.m_low, mass.m_high, res[0])
  m2 = jnp.linspace(mass.m_low, mass.m_high, res[1])
  m1mesh, m2mesh = jnp.meshgrid(m1, m2)
  p_joint = pdf_m1m2(mass, m1mesh, m2mesh)
  p1_marg = trapz(p_joint, x=m2, axis=0)
  p1_marg /= trapz(p1_marg, x=m1)
  p2_marg = trapz(p_joint, x=m1, axis=1)
  p2_marg /= trapz(p2_marg, x=m2)

  dict_to_ret = {'m1':m1, 'm2':m2, 'm1mesh':m1mesh, 'm2mesh':m2mesh,
    'p_joint': p_joint, 'p_m1_marg': p1_marg, 'p_m2_marg': p2_marg}
  return dict_to_ret
