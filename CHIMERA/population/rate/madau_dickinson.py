from plum import dispatch
import jax.numpy as jnp
from .core import base_rate_struct

class madau_dickinson(base_rate_struct):
  r"""A normalized Madau-Dickinson merger rate model implemented as an Equinox module.

  Args:
    gamma (float): Primary power-law exponent.
    kappa (float): Secondary power-law exponent.
    zp (float): Pivot redshift scale.
  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.
  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary.
  Methods:
    - update(**kwargs) (madau_dickisnon): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  gamma: float
  kappa: float
  zp: float
  name = 'madau_dickinson'
  default = {'gamma':3.27, 'kappa':2.88, 'zp':2.45}
  keys = list(default.keys())

@dispatch
def merger_rate(rate: madau_dickinson, z: jnp.ndarray):
  """Computes the merger rate."""
  rate_md_not_norm = (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )
  one_over_norm = 1. + (1.+rate.zp)**(-rate.gamma-rate.kappa)
  return one_over_norm*rate_md_not_norm


class trunc_madau_dickinson(base_rate_struct):
	gamma: float
	kappa: float
	zp: float
	zmax: float
	name = 'trunc_madau_dickinson'
	default = {'gamma':2.7, 'kappa':3.0, 'zp':2., 'zmax':1.3}
	keys = list(default.keys())

@dispatch
def merger_rate(rate: trunc_madau_dickinson, z: jnp.ndarray):
	"""Computes the merger rate."""
	rate_md_not_norm = (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )
	one_over_norm = 1. + (1.+rate.zp)**(-rate.gamma-rate.kappa)
	return jnp.where(z<rate.zmax, one_over_norm*rate_md_not_norm, 0.)
