from ..utils.config import jax, jnp
import equinox as eqx
from plum import dispatch
from ..data import theta_src, theta_pe_det, theta_inj_det

################
# RATE PYTREES #
################

class base_rate_struct(eqx.Module):
  default = {}
  keys = list(default.keys())
  name = 'base_rate_struct'

  def __init__(self, **kwargs):
    for key in self.default.keys():
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      # No change - return original object
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    return self.__class__(**fiducials)

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
  default = {'gamma':2.7, 'kappa':3.0, 'zp':2.}
  keys = list(default.keys())

class trunc_madau_dickinson(base_rate_struct):
	gamma: float
	kappa: float
	zp: float
	zmax: float
	name = 'trunc_madau_dickinson'
	default = {'gamma':2.7, 'kappa':3.0, 'zp':2., 'zmax':1.3}
	keys = list(default.keys())

class trunc_power_law(base_rate_struct):
	gamma: float
	zmax: float
	name = 'trunc_power_law'
	default = {'gamma':1.9, 'zmax':1.3}
	keys = list(default.keys())

class z2_exp(base_rate_struct):
  r"""Merger rate R(z) = z^2 * exp(-(z/z0)^alphaz).
  Rises as z^2 at low z and is suppressed above z0.
  From icaroverse LISA MBHB/EMRI snippet.
  Args:
    z0 (float): characteristic cutoff redshift.
    alphaz (float): sharpness of the exponential cutoff.
  """
  z0: float
  alphaz: float
  name = 'z2_exp'
  default = {'z0': 2.0, 'alphaz': 2.0}
  keys = list(default.keys())

class exp_cutoff_power_law(base_rate_struct):
  r"""Merger rate R(z) = (1+z)^gamma * exp(-(z/z0)^alphaz).
  Power-law evolution at low z with exponential suppression above z0.
  Recovers standard power_law for z0 -> inf.
  Args:
    gamma (float): low-z power-law exponent.
    z0 (float): characteristic cutoff redshift.
    alphaz (float): sharpness of the exponential cutoff.
  """
  gamma: float
  z0: float
  alphaz: float
  name = 'exp_cutoff_power_law'
  default = {'gamma': 1.7, 'z0': 2.0, 'alphaz': 2.0}
  keys = list(default.keys())

##################
# RATE FUNCTIONS #
##################

# power law

@dispatch
def merger_rate(rate: power_law, z: jnp.ndarray):
  """Computes the merger rate."""
  return (1.+z)**rate.gamma

@dispatch
def merger_rate(rate: trunc_power_law, z: jnp.ndarray):
  """Computes the merger rate."""
  pdf = (1.+z)**rate.gamma
  norm = ((1+rate.zmax)**(rate.gamma+1) - 1)/(rate.gamma+1)
  return jnp.where(z<rate.zmax, pdf/norm, 0.)

# madau dickinson

@dispatch
def merger_rate(rate: madau_dickinson, z: jnp.ndarray):
  """Computes the merger rate."""
  rate_md_not_norm = (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )
  one_over_norm = 1. + (1.+rate.zp)**(-rate.gamma-rate.kappa)
  return one_over_norm*rate_md_not_norm

@dispatch
def merger_rate(rate: trunc_madau_dickinson, z: jnp.ndarray):
	"""Computes the merger rate."""
	rate_md_not_norm = (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )
	one_over_norm = 1. + (1.+rate.zp)**(-rate.gamma-rate.kappa)
	return jnp.where(z<rate.zmax, one_over_norm*rate_md_not_norm, 0.)


@dispatch
def merger_rate(rate: z2_exp, z: jnp.ndarray):
  """Computes the merger rate."""
  return z**2 * jnp.exp(-(z / rate.z0)**rate.alphaz)

@dispatch
def merger_rate(rate: exp_cutoff_power_law, z: jnp.ndarray):
  """Computes the merger rate."""
  return (1. + z)**rate.gamma * jnp.exp(-(z / rate.z0)**rate.alphaz)

# Final dipatch for functions that needs theta_src as argument instead of array:

@dispatch
def merger_rate(rate: base_rate_struct, theta:theta_src):
  return merger_rate(rate, theta.z)
