import jax.numpy as jnp
from plum import dispatch
from .base import base_mass_conditioned_struct
from ..core import truncated_pl, truncated_gaussian, high_pass_filter


class plp(base_mass_conditioned_struct):
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
  mu_g: float
  sigma_g: float
  default = {**base_mass_conditioned_struct.default,
    'm_low':5.26,
    'm_high':94.4,
    'lambda_peak':0.04,
    'alpha':3.6,
    'beta':1.08,
    'delta_m':3.3,
    'mu_g':28.6,
    'sigma_g':5.15}
  name = 'power_law_plus_peak'

@dispatch
def primary_mass_pdf_notnorm(mass:plp, m: jnp.ndarray):
  P = truncated_pl(m, -mass.alpha, mass.m_low, mass.m_high)
  G = truncated_gaussian(m, mass.mu_g, mass.sigma_g, mass.m_low, mass.mu_g + 5*mass.sigma_g)
  pdf = (1 - mass.lambda_peak)*P + mass.lambda_peak*G
  pdf *= high_pass_filter(m, mass.delta_m, mass.m_low)
  return pdf
