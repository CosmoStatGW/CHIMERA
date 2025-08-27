from ..utils.config import jax, jnp
from ..utils.math import cumtrapz
import equinox as eqx
from typing import Optional, Union, List, Dict
from numbers import Number
from plum import dispatch
from ..data import theta_src, theta_pe_det, theta_inj_det

########################
# COSMOLOGICAL PYTREES #
########################

class base_cosmology_struct(eqx.Module):
  z_max: float
  z_grid_interp: jnp.ndarray
  integral_invE_interp:jnp.ndarray
  z_grid_res: int = eqx.field(static=True)
  default = {'z_max':10., 'z_grid_res':1000}
  keys: List[str] = eqx.field(static=True)
  name = 'base_cosmology_struct'

  def __init__(self, **kwargs):
    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    setup_interp(self)

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

# utility function --> need to define E_at_z for each cosmology!
def setup_interp(cosmo:base_cosmology_struct):
  setattr(cosmo, 'z_grid_interp',jnp.concatenate([jnp.array([0]), jnp.logspace(-10, jnp.log10(cosmo.z_max), cosmo.z_grid_res-1)]))
  Ez = E_at_z(cosmo, cosmo.z_grid_interp)
  setattr(cosmo, 'integral_invE_interp', cumtrapz(1./Ez, cosmo.z_grid_interp))

# Specific implementations

class flrw(base_cosmology_struct):
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
  name = 'flrw'
  default = {**base_cosmology_struct.default, 'H0':70., 'Om0':0.25, 'Ok0':0., 'Or0':0., 'w0':-1., 'wa':0., 'z_max':10., 'z_grid_res': 1500}

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

##########################
# COSMOLOGICAL FUNCTIONS #
##########################

# Common functions
def E_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  """Computes the dimensionless Hubble parameter E(z)."""
  w_z = cosmo.w0 + cosmo.wa * z / (1 + z)
  Ez = jnp.sqrt(cosmo.Om0*(1.+z)**3 +
    cosmo.Or0*(1.+z)**4 +
    cosmo.Ok0*(1.+z)**2 +
    cosmo.Ode0*(1.+z)**(3.*(1.+w_z))
  )
  return Ez

def int_invE_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  return jnp.interp(z, cosmo.z_grid_interp, cosmo.integral_invE_interp)

def dCr_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  """Computes the radial comoving distance at z."""
  int_invEz = int_invE_at_z(cosmo, z)
  dCr = cosmo.dH * int_invEz
  return dCr

def dCt_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  """Computes the transverse comoving distance at z."""
  dCr     = dCr_at_z(cosmo, z)
  sqrtOk0 = jnp.sqrt(jnp.abs(cosmo.Ok0+1.e-10))
  dH      = cosmo.dH
  dCt     = jnp.where(cosmo.Ok0 == 0.0,
    dCr,
    jnp.where(cosmo.Ok0 > 0.0,
      (dH / sqrtOk0) * jnp.sinh(sqrtOk0 * dCr / dH),
      (dH / sqrtOk0) * jnp.sin(sqrtOk0 * dCr / dH)
    )
  )
  return dCt

def dA_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the angular distance at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  dA  = dCt/(1.+z)
  return dA

@dispatch

def Vc_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the comoving volume at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo,z)

  regOk0 = cosmo.Ok0 + 1e-10
  sqrtOk0 = jnp.sqrt(jnp.abs(regOk0))
  dH  = cosmo.dH
  Vc  = jnp.where(cosmo.Ok0 == 0.0,
    4.*jnp.pi*dCt**3 / 3.,
    jnp.where(cosmo.Ok0 > 0.0,
      (4.*jnp.pi*dH**3/(2.*regOk0))*((dCt/dH)*jnp.sqrt(1+regOk0*dCt**2/dH**2)
      -jnp.arcsinh(sqrtOk0*dCt/dH)/sqrtOk0),
      (4. * jnp.pi * dH**3/(2.*regOk0))*(
      (dCt/dH)*jnp.sqrt(1+regOk0*dCt**2/dH**2)
      - jnp.arcsin(sqrtOk0*dCt/dH)/sqrtOk0)
    )
  )
  return Vc

@dispatch
def dVcdz_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the differential comoving volume at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo,z)
  E_z = E_at_z(cosmo,z)
  dVc = 4*jnp.pi * cosmo.dH * dCt**2 / E_z
  return dVc

# Functions for `flrw` only

@dispatch
def _dL2dCt(cosmo: flrw, distances: jnp.ndarray, z: jnp.ndarray):
  return distances/(1.+z)

@dispatch
def dL_at_z(cosmo: flrw, z: jnp.ndarray):
  """Computes the luminosity distance at z."""
  dCt = dCt_at_z(cosmo, z)
  dL  = dCt*(1.+z)
  return dL

@dispatch
def ddLdz_at_z(cosmo: flrw, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the differential luminosity distance at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  E_z = E_at_z(cosmo, z)
  ddL = dCt + (cosmo.dH/E_z)*(1.+z)
  return ddL

# Functions for `mg_flrw` only

@dispatch
def Xi_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  """Computes the MG factor :math:`\Xi(z)`"""
  return cosmo.Xi0 + (1. - cosmo.Xi0)/((1.+z)**cosmo.n)

@dispatch
def _dL2dCt(cosmo: mg_flrw, distances: jnp.ndarray, z: jnp.ndarray):
  Xiz =  Xi_at_z(cosmo,z)
  dLflrw = distances/Xiz
  dCt = dLflrw/(1.+z)
  return dCt

@dispatch
def dL_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  """Computes the luminosity distance at z."""
  dCt = dCt_at_z(cosmo, z)
  dL  = dCt*(1.+z)
  Xiz = Xi_at_z(cosmo, z)
  return dL*Xiz

@dispatch
def ddLdz_at_z(cosmo: mg_flrw, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the differential luminosity distance at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  dLflrw  = dCt*(1.+z)
  Ez      = E_at_z(cosmo, z)
  ddLflrw = dCt + (cosmo.dH/Ez)*(1.+z)
  Xiz  = Xi_at_z(cosmo,z)
  dXiz = cosmo.n*(cosmo.Xi0-1.)/((1.+z)**(cosmo.n+1))
  return ddLflrw*Xiz + dLflrw*dXiz

# dL2z converter
@dispatch
def z_from_dGW(cosmo: Union[flrw, mg_flrw], dGWs: jnp.ndarray):
  """Computes the redshifts correspoding to the given the GW distances."""
  dGW_values = dL_at_z(cosmo, cosmo.z_grid_interp)
  return jnp.interp(dGWs, dGW_values, cosmo.z_grid_interp)


# Final dipatch for functions that needs theta_src as argument instead of array:

@dispatch
def dVcdz_at_z(cosmo: Union[flrw,mg_flrw], theta:theta_src):
  return dVcdz_at_z(cosmo, theta.z, theta.original_distances)

@dispatch
def Vc_at_z(cosmo: Union[flrw, mg_flrw], theta:theta_src):
  return Vc_at_z(cosmo, theta.z, theta.original_distances)

@dispatch
def ddLdz_at_z(cosmo: Union[flrw,mg_flrw], theta:theta_src):
  return ddLdz_at_z(cosmo, theta.z, theta.original_distances)
