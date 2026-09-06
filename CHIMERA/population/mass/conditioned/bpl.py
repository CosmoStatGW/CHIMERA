import jax.numpy as jnp
from plum import dispatch
from .base import base_mass_conditioned_struct
from ..core import broken_pl, high_pass_filter

class bpl(base_mass_conditioned_struct):
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
  break_fraction: float
  default = {**base_mass_conditioned_struct.default, 'alpha_1':1.6, 'alpha_2':5.6, 'beta':1.1, 'delta_m':4.8, 'break_fraction':0.43}
  name = 'broken_power_law'

@dispatch
def primary_mass_pdf_notnorm(mass:bpl, m:jnp.ndarray):
  pdf = broken_pl(m, mass.alpha_1, mass.alpha_2, mass.b, mass.m_low, mass.m_high)
  pdf *= high_pass_filter(m, mass.delta_m, mass.m_low)
  return pdf
