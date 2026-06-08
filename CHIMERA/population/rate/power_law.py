from plum import dispatch
import jax.numpy as jnp
from .core import base_rate_struct

class power_law(base_rate_struct):
  r"""A normalized Madau-Dickinson merger rate model implemented as an Equinox module.

  Args:
    gamma (float): power-law exponent.
  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.
  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary.
  Methods:
    - update(**kwargs) (power_law): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  gamma: float
  name = 'power_law'
  default = {'gamma':1.7}
  keys = list(default.keys())

@dispatch
def merger_rate(rate: power_law, z: jnp.ndarray):
  """Computes the merger rate."""
  return (1.+z)**rate.gamma

class trunc_power_law(base_rate_struct):
	gamma: float
	zmax: float
	name = 'trunc_power_law'
	default = {'gamma':1.9, 'zmax':1.3}
	keys = list(default.keys())

@dispatch
def merger_rate(rate: trunc_power_law, z: jnp.ndarray):
  """Computes the merger rate."""
  pdf = (1.+z)**rate.gamma
  norm = ((1+rate.zmax)**(rate.gamma+1) - 1)/(rate.gamma+1)
  return jnp.where(z<rate.zmax, pdf/norm, 0.)
