import jax.numpy as jnp
from typing import Optional
from plum import dispatch

from .core import E_at_z
from .distances import dCt_at_z, _dL2dCt
from ...data import theta_src

@dispatch
def Vc_at_z(cosmo, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)

  regOk0 = cosmo.Ok0 + 1e-10
  sqrtOk0 = jnp.sqrt(jnp.abs(regOk0))
  dH = cosmo.dH

  return jnp.where(
    cosmo.Ok0 == 0.0,
    4. * jnp.pi * dCt**3 / 3.,
    jnp.where(
      cosmo.Ok0 > 0.0,
      (4. * jnp.pi * dH**3 / (2. * regOk0)) *
      ((dCt / dH) * jnp.sqrt(1 + regOk0 * dCt**2 / dH**2) -
        jnp.arcsinh(sqrtOk0 * dCt / dH) / sqrtOk0),
      (4. * jnp.pi * dH**3 / (2. * regOk0)) *
      ((dCt / dH) * jnp.sqrt(1 + regOk0 * dCt**2 / dH**2) -
        jnp.arcsin(sqrtOk0 * dCt / dH) / sqrtOk0)
    )
  )

@dispatch
def dVcdz_at_z(cosmo, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  E_z = E_at_z(cosmo, z)
  return 4 * jnp.pi * cosmo.dH * dCt**2 / E_z

# Theta source dispatches
@dispatch
def dVcdz_at_z(cosmo, theta: theta_src):
  return dVcdz_at_z(cosmo, theta.z, theta.original_distances)

@dispatch
def Vc_at_z(cosmo, theta: theta_src):
  return Vc_at_z(cosmo, theta.z, theta.original_distances)
