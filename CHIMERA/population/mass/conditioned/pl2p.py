import jax.numpy as jnp
from plum import dispatch
from .base import base_mass_conditioned_struct
from ..core import tpl_notnorm, tpl_cdf, truncated_gaussian, smoothing

class pl2p(base_mass_conditioned_struct):
  r"""A class to describe a power law mass model with two Gaussian peaks, implemented as an Equinox module.

  Args:
    alpha (float, optional): The slope of the power law for masses below the break.
    beta (float, optional): The slope of the power law for masses above the break.
    m_low (float, optional): The lower truncation mass.
    m_high (float, optional): The upper truncation mass.
    mu_g_low (float, optional): The mean of the first Gaussian peak.
    sigma_g_low (float, optional): The standard deviation of the first Gaussian peak.
    mu_g_high (float, optional): The mean of the second Gaussian peak.
    sigma_g_high (float, optional): The standard deviation of the second Gaussian peak.
    lambda_g (float, optional): The strength of the first Gaussian peak.
    lambda_g_low (float, optional): The strength of the second Gaussian peak.
    delta_m (float, optional): The characteristic mass difference that influences the break.

  Class Attributes:
    - name (str): The name of the model, set to 'power_law_plus_double_peak'.
    - keys (List[str]): A list of parameter names used in the model,.

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary, where keys are parameter names
      and values are their corresponding `float` values.

  Methods:
    - update(**kwargs) (pl2p): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter provided in the input.
  """

  alpha: float
  beta: float
  mu_g_low: float
  sigma_g_low: float
  mu_g_high: float
  sigma_g_high: float
  lambda_g: float
  lambda_g_low: float
  delta_m: float
  default = {**base_mass_conditioned_struct.default, 'alpha':2.9, 'beta':1.04, 'm_low':4.58, 'm_high':86.3, 'mu_g_low':9.67, 'sigma_g_low':0.74, 'mu_g_high':30.65, 'sigma_g_high':6.3, 'lambda_g':0.38, 'lambda_g_low':0.85, 'delta_m':4.8}
  name = 'power_law_plus_double_peak'

@dispatch
def primary_mass_pdf_notnorm(mass:pl2p, m: jnp.ndarray):  
  P = tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)/tpl_cdf(mass.m_high, -mass.alpha, mass.m_low)
  G1 = truncated_gaussian(m, mass.mu_g_low, mass.sigma_g_low, mass.m_low, mass.mu_g_low + 5*mass.sigma_g_low)
  G2 = truncated_gaussian(m, mass.mu_g_high, mass.sigma_g_high, mass.m_low, mass.mu_g_high + 5*mass.sigma_g_high)
  pdf = (1-mass.lambda_g)*P + mass.lambda_g*mass.lambda_g_low*G1 + mass.lambda_g*(1. - mass.lambda_g_low)*G2
  pdf *= smoothing(m, mass.delta_m, mass.m_low)
  #return pdf
  return jnp.where(mass.mu_g_low <= mass.mu_g_high, pdf, jnp.nan)
