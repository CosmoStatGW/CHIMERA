from ..utils.config import jax, jnp
import equinox as eqx
from typing import Optional, Union, List, Dict
from numbers import Number
from plum import dispatch
from functools import partial

########################
# COSMOLOGICAL PYTREES #
########################

class flrw(eqx.Module):
  """Parameters describing a cosmological FLRW model implemented as an Equinox module.

  Args:
    H0 (jax.Array, optional): The Hubble constant in km/s/Mpc. Default is 70.
    Om0 (jax.Array, optional): The matter density parameter. Default is 0.25.
    Ok0 (jax.Array, optional): The curvature density parameter. Default is 0.
    Or0 (jax.Array, optional): The radiation density parameter. Default is 0.
    w0 (jax.Array, optional): The present value of the dark energy equation of state parameter. Default is -1.
    wa (jax.Array, optional): The rate of change of the dark energy equation of state parameter. Default is 0.

  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.

  Properties:
    - as_dict (Dict[str, jax.Array]): Returns the current model parameters as a dictionary.
    - Ode0 (jax.Array): The dark energy density parameter, calculated as `1 - Om0 - Or0 - Ok0`.
    - dH (jax.Array): The Hubble distance in units of Mpc, calculated as `299792.458e-3 / H0`.

  Methods:
    - to_dict() (Dict[str, jax.Array]): Returns the fiducial parameters of the model as a dictionary.
    - set(**kwargs) (flrw): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  H0: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=70.)
  Om0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.25)
  Ok0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.)
  Or0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.,)
  w0: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=-1.)
  wa: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=0.)
  name = 'flrw'
  keys = ['H0', 'Om0', 'Ok0', 'Or0', 'w0', 'wa']
  @property
  def as_dict(self):
    return {
      k: getattr(self, k) for k in self.keys
    }
  @classmethod
  def to_dict(self):
    return {
      k: getattr(self, k) for k in self.keys
    }
  @classmethod
  def set(cls, **kwargs):
    fiducials = cls.to_dict()
    params = {key: jnp.asarray(kwargs.get(key, fiducials[key]))
      for key in cls.keys}
    max_size = max(p.size for p in params.values())
    for key, value in params.items():
      if value.size == 1 and max_size > 1:
        params[key] = jnp.full((max_size,), value)
    return cls(**params)
  @property
  def Ode0(self):
    return 1.0 - self.Om0 - self.Or0 - self.Ok0
  @property
  def dH(self):
    return 299792.458e-3 / self.H0 # Gpc

class mg_flrw(flrw):
  """Parameters describing a cosmological FLRW model with Modified Gravity propagation implemented as an Equinox module.

  Args:
    H0 (jax.Array, optional): The Hubble constant in km/s/Mpc. Default is 70.
    Om0 (jax.Array, optional): The matter density parameter. Default is 0.25.
    Ok0 (jax.Array, optional): The curvature density parameter. Default is 0.
    Or0 (jax.Array, optional): The radiation density parameter. Default is 0.
    w0 (jax.Array, optional): The present value of the dark energy equation of state parameter. Default is -1.
    wa (jax.Array, optional): The rate of change of the dark energy equation of state parameter. Default is 0.
    Xi0 (jax.Array, optional): The MG (modified gravity) parameter Xi0. Default is 1.
    n (jax.Array, optional): The MG parameter `n`. Default is 0.

  Class Attributes:
    - name (str): The name of the model.
    - keys (List[str]): A list of parameter names used in the model.

  Properties:
    - as_dict (Dict[str, jax.Array]): Returns the current model parameters as a dictionary.
    - Ode0 (jax.Array): The dark energy density parameter, calculated as `1 - Om0 - Or0 - Ok0`.
    - dH (jax.Array): The Hubble distance in units of Mpc, calculated as `299792.458e-3 / H0`.

  Methods:
    - to_dict() (Dict[str, jax.Array]): Returns the fiducial parameters of the model as a dictionary.
    - set(**kwargs) (flrw): Creates a new instance of the model, updating any parameters provided in `kwargs`.
      Parameters are automatically broadcasted to the same shape as the largest parameter.
  """
  H0: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=70.)
  Om0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.25)
  Ok0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.)
  Or0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.,)
  w0: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=-1.)
  wa: jax.Array  = eqx.field(converter=jnp.atleast_1d, default=0.)
  Xi0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1.)
  n: jax.Array   = eqx.field(converter=jnp.atleast_1d, default = 0.)

  name = 'mg_flrw'
  keys = ['H0', 'Om0', 'Ok0', 'Or0', 'w0', 'wa', 'Xi0', 'n']

##########################
# COSMOLOGICAL FUNCTIONS #
##########################

# Common functions

@dispatch
@jax.jit
def E_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  """Computes the dimensionless Hubble parameter E(z)."""
  w_z = cosmo.w0 + cosmo.wa * z / (1 + z)
  Ez = jnp.sqrt(cosmo.Om0*(1.+z)**3 +
    cosmo.Or0*(1.+z)**4 +
    cosmo.Ok0*(1.+z)**2 +
    cosmo.Ode0*(1.+z)**(3.*(1.+w_z))
  )
  return Ez

@dispatch
@partial(jax.jit, static_argnums=2)
def int_invE_at_z(cosmo: Union[flrw, mg_flrw],
  z: jnp.ndarray,
  res: int = 500):
  """Computes the integrated inverse dimensionless Hubble parameter, that is 1/E(z')dz' from 0 to z."""
  max_z    = jnp.max(jnp.atleast_1d(z))
  z_values = jnp.linspace(0.0, max_z, res)
  dz       = z_values[1] - z_values[0]
  e_vals   = 1. / E_at_z(cosmo, z_values)

  cum_integral_values = jnp.empty_like(z_values)
  cum_integral_values = cum_integral_values.at[0].set(0)
  cum_integral_values = cum_integral_values.at[1:].set(jnp.cumsum(e_vals[:-1] + e_vals[1:])/2 * dz)

  return jnp.interp(z, z_values, cum_integral_values)

@dispatch
@jax.jit
def dCr_at_z(cosmo: Union[flrw, mg_flrw], z: jnp.ndarray):
  """Computes the radial comoving distance at z."""
  int_invEz = int_invE_at_z(cosmo, z)
  dCr = cosmo.dH * int_invEz
  return dCr

@dispatch
@jax.jit
def dCt_at_z(cosmo: Union[flrw, mg_flrw],
  z: jnp.ndarray):
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

@dispatch
@jax.jit
def dA_at_z(cosmo: Union[flrw, mg_flrw],
  z: jnp.ndarray,
  distances: Optional[jnp.ndarray] = None):
  """Computes the angular distance at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  dA  = dCt/(1.+z)
  return dA

@dispatch
@jax.jit
def Vc_at_z(cosmo: Union[flrw, mg_flrw],
  z: jnp.ndarray,
  distances: Optional[jnp.ndarray] = None):
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
@jax.jit
def dVcdz_at_z(cosmo: Union[flrw, mg_flrw],
  z: jnp.ndarray,
  distances: Optional[jnp.ndarray] = None):
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
@jax.jit
def _dL2dCt(cosmo: flrw,
  distances: jnp.ndarray,
  z: jnp.ndarray):
  return distances/(1.+z)

@dispatch
@jax.jit
def dL_at_z(cosmo: flrw, z: jnp.ndarray):
  """Computes the luminosity distance at z."""
  dCt = dCt_at_z(cosmo, z)
  dL  = dCt*(1.+z)
  return dL

@dispatch
@jax.jit
def ddLdz_at_z(cosmo: flrw,
  z: jnp.ndarray,
  distances: Optional[jnp.ndarray] = None):
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
@jax.jit
def Xi_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  """Computes the MG factor :math:`\Xi(z)`"""
  return cosmo.Xi0 + (1. - cosmo.Xi0)/((1.+z)**cosmo.n)

@dispatch
@jax.jit
def _dL2dCt(cosmo: mg_flrw,
  distances: jnp.ndarray,
  z: jnp.ndarray):
  Xiz =  Xi_at_z(cosmo,z)
  dLflrw = distances/Xiz
  dCt = dLflrw/(1.+z)
  return dCt

@dispatch
@jax.jit
def dL_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  """Computes the luminosity distance at z."""
  dCt = dCt_at_z(cosmo, z)
  dL  = dCt*(1.+z)
  Xiz = Xi_at_z(cosmo, z)
  return dL*Xiz

@dispatch
@jax.jit
def ddLdz_at_z(cosmo: mg_flrw,
  z: jnp.ndarray,
  distances: Optional[jnp.ndarray] = None):
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
@partial(jax.jit, static_argnums=(2,3))
def z_from_dGW(cosmo: Union[flrw, mg_flrw],
  dGWs: jnp.ndarray,
  max_z: Number,
  res: int=500):
  """Computes the redshifts correspoding to the given the GW distances."""
  z_values   = jnp.linspace(0.0, max_z, res)
  dGW_values = dL_at_z(cosmo, z_values)
  return jnp.interp(dGWs, dGW_values, z_values)

# compute z_grids

@dispatch
def compute_z_grids(cosmo: Union[flrw, mg_flrw],
  events_dL: jnp.ndarray,
  cosmo_prior: Optional[Dict[str, List[Number]]] = None,
  z_conf_range: Union[Number, list] = 5,
  z_int_res: int = 300):
  """Computes the redshift grids on which GW events have support, given some cosmological priors.

  Args:
    cosmo (eqx.Module): a CHIMERA.cosmo object describing cosmological params.
    events_dL (jnp.ndarray): GW events luminosity distance samples
    cosmo_prior (optional, Dict[str, List[Numbers]]): dict containing tuple of parameters, describing the priors over cosmological parameters.
    z_conf_range (Number or List): sigma or quantiles defining the redshift grid extenstion.
    z_int_res (int): number fo points of the redshift grid.

  Returns:
    jnp.ndarray: the redshift grids.
  """
  if isinstance(z_conf_range, list):
    dL_min, dL_max = jnp.percentile(events_dL, z_conf_range, axis=1)
  elif isinstance(z_conf_range, Number):
    mu     = jnp.mean(events_dL, axis=1)
    sig    = jnp.std(events_dL, axis=1)
    dL_min = mu - z_conf_range * sig
    dL_max = mu + z_conf_range * sig

  dL_min = jnp.where(dL_min < 1.e-6, 1.e-6, dL_min)

  # update cosmo prior
  cp = {k:[v, v] for k,v in cosmo.as_dict.items()}
  if cosmo_prior is not None:
    cp.update(cosmo_prior)

  if cosmo.name=='flrw':
    lc_low = {"H0" :cp['H0'][0],
      "Om0":cp['Om0'][0],
      "Ok0":cp['Ok0'][0],
      "Or0":cp['Or0'][0],
      "w0" :cp['w0'][0],
      "wa" :cp['wa'][0]
    }

    lc_high = {"H0" :cp['H0'][1],
      "Om0":cp['Om0'][1],
      "Ok0":cp['Ok0'][1],
      "Or0":cp['Or0'][1],
      "w0" :cp['w0'][1],
      "wa" :cp['wa'][1]
    }
    max_z = 20.
  else:
    lc_low  = {"H0" :cp['H0'][0],
      "Om0":cp['Om0'][0],
      "Ok0":cp['Ok0'][0],
      "Or0":cp['Or0'][0],
      "w0" :cp['w0'][0],
      "wa" :cp['wa'][0],
      "Xi0":cp['Xi0'][1],
      "n"  :cp['n'][1]
    }

    lc_high = {"H0" :cp['H0'][1],
      "Om0":cp['Om0'][1],
      "Ok0":cp['Ok0'][1],
      "Or0":cp['Or0'][1],
      "w0" :cp['w0'][1],
      "wa" :cp['wa'][1],
      "Xi0":cp['Xi0'][0],
      "n"  :cp['n'][1] # true if cp['Xi0'][0] < 1, otherwise should be cp['n'][0]
    }
    max_z = 20.

  cosmo1 = cosmo.set(**lc_low)
  cosmo2 = cosmo.set(**lc_high)

  z_min  = z_from_dGW(cosmo1, dL_min, max_z, 10_000)
  z_max  = z_from_dGW(cosmo2, dL_max, max_z, 10_000)

  zgrids = jnp.linspace(z_min, z_max, z_int_res, axis=1)

  return zgrids
