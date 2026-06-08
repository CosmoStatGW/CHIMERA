import jax.numpy as jnp
import equinox as eqx
from plum import dispatch
from CHIMERA.data import theta_src

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


@dispatch
def merger_rate(rate: base_rate_struct, z: jnp.ndarray):
  raise NotImplementedError(f"merger_rate not implemented for {rate.name}")

# Final dipatch for functions that needs theta_src as argument instead of array:

@dispatch
def merger_rate(rate: base_rate_struct, theta:theta_src):
  return merger_rate(rate, theta.z)
