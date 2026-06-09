import equinox as eqx
from typing import List
import jax.numpy as jnp
from ...utils.math import cumtrapz

########################
# COSMOLOGICAL PYTREES #
########################

class flrw(eqx.Module):
  """Parameters describing a cosmological FLRW model implemented as an Equinox module.
  Args:
    H0 (float, optional): The Hubble constant in km/s/Mpc. Default is 70.
    Om0 (float, optional): The matter density parameter. Default is 0.25.
    Ok0 (float, optional): The curvature density parameter. Default is 0.
    Or0 (float, optional): The radiation density parameter. Default is 0.
    w0 (float, optional): The present value of the dark energy equation of state parameter. Default is -1.
    wa (float, optional): The rate of change of the dark energy equation of state parameter. Default is 0
  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.
  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary.
    - Ode0 (float): The dark energy density parameter, calculated as `1 - Om0 - Or0 - Ok0`.
    - dH (float): The Hubble distance in units of Mpc, calculated as `299792.458e-3 / H0`.
  Methods:
    - update(**kwargs) (flrw): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  H0: float
  Om0: float
  Ok0: float
  Or0: float
  w0: float
  wa: float
  z_max: float
  z_grid_interp: jnp.ndarray
  integral_invE_interp:jnp.ndarray
  z_grid_res: int = eqx.field(static=True)

  default = {'H0':67.9, 'Om0':0.3065, 'Ok0':0., 'Or0':0., 'w0':-1., 'wa':0., 'z_max':10., 'z_grid_res': 1500}
  keys: List[str] = eqx.field(static=True)
  name = 'flrw'

  def __init__(self, **kwargs):
    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    _setup_interp(self)

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

  @property
  def Ode0(self):
    return 1.0 - self.Om0 - self.Or0 - self.Ok0

  @property
  def dH(self):
    return 299792.458e-3 / self.H0

class mg_flrw(flrw):
  """Parameters describing a cosmological FLRW model with Modified Gravity propagation implemented as an Equinox module.

  Args:
    H0 (float, optional): The Hubble constant in km/s/Mpc. Default is 70.
    Om0 (float, optional): The matter density parameter. Default is 0.25.
    Ok0 (float, optional): The curvature density parameter. Default is 0.
    Or0 (float, optional): The radiation density parameter. Default is 0.
    w0 (float, optional): The present value of the dark energy equation of state parameter. Default is -1.
    wa (float, optional): The rate of change of the dark energy equation of state parameter. Default is 0.
    Xi0 (float, optional): The MG (modified gravity) parameter Xi0. Default is 1.
    n (float, optional): The MG parameter `n`. Default is 0.

  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.

  Properties:
    - as_dict (Dict[str, float]): Returns the current model parameters as a dictionary.
    - Ode0 (float): The dark energy density parameter, calculated as `1 - Om0 - Or0 - Ok0`.
    - dH (float): The Hubble distance in units of Mpc, calculated as `299792.458e-3 / H0`.

  Methods:
    - update(**kwargs) (mg_flrw): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  Xi0: float
  n: float
  name = 'mg_flrw'
  default = {**flrw.default, 'Xi0':1., 'n':0.}

# Common functions
def _setup_interp(cosmo:flrw):
  # utility function used in each struct initialization
  setattr(cosmo, 'z_grid_interp',jnp.concatenate([jnp.array([0]), jnp.logspace(-10, jnp.log10(cosmo.z_max), cosmo.z_grid_res-1)]))
  Ez = E_at_z(cosmo, cosmo.z_grid_interp)
  setattr(cosmo, 'integral_invE_interp', cumtrapz(1./Ez, cosmo.z_grid_interp))

def E_at_z(cosmo: flrw, z: jnp.ndarray):
  """Computes the dimensionless Hubble parameter E(z)."""
  w_z = cosmo.w0 + cosmo.wa * z / (1 + z)
  Ez = jnp.sqrt(cosmo.Om0*(1.+z)**3 +
    cosmo.Or0*(1.+z)**4 +
    cosmo.Ok0*(1.+z)**2 +
    cosmo.Ode0*(1.+z)**(3.*(1.+w_z))
  )
  return Ez

def int_invE_at_z(cosmo: flrw, z: jnp.ndarray):
  return jnp.interp(z, cosmo.z_grid_interp, cosmo.integral_invE_interp)
