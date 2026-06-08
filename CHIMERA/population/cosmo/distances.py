import jax.numpy as jnp
from typing import Optional
from plum import dispatch
from .core import flrw, mg_flrw, E_at_z, int_invE_at_z
from ...data import theta_src

# Shared functions

def dCr_at_z(cosmo, z: jnp.ndarray):
  """Computes the radial comoving distance at z."""
  int_invEz = int_invE_at_z(cosmo, z)
  dCr = cosmo.dH * int_invEz
  return dCr

def dCt_at_z(cosmo, z: jnp.ndarray):
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

def dA_at_z(cosmo, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  """Computes the angular distance at z."""
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  dA  = dCt/(1.+z)
  return dA

# FLRW-specific distance functions

@dispatch
def _dL2dCt(cosmo: flrw, distances: jnp.ndarray, z: jnp.ndarray):
    return distances / (1. + z)

@dispatch
def dL_at_z(cosmo: flrw, z: jnp.ndarray):
  return dCt_at_z(cosmo, z) * (1. + z)

@dispatch
def ddLdz_at_z(cosmo: flrw, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)
  E_z = E_at_z(cosmo, z)
  return dCt + (cosmo.dH / E_z) * (1. + z)

# MG-FLRW specific distance functions

@dispatch
def Xi_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  return cosmo.Xi0 + (1. - cosmo.Xi0) / ((1. + z)**cosmo.n)

@dispatch
def _dL2dCt(cosmo: mg_flrw, distances: jnp.ndarray, z: jnp.ndarray):
  Xiz = Xi_at_z(cosmo, z)
  dLflrw = distances / Xiz
  return dLflrw / (1. + z)

@dispatch
def dL_at_z(cosmo: mg_flrw, z: jnp.ndarray):
  dCt = dCt_at_z(cosmo, z)
  dL = dCt * (1. + z)
  Xiz = Xi_at_z(cosmo, z)
  return dL * Xiz

@dispatch
def ddLdz_at_z(cosmo: mg_flrw, z: jnp.ndarray, distances: Optional[jnp.ndarray] = None):
  if distances is not None:
    dCt = _dL2dCt(cosmo, distances, z)
  else:
    dCt = dCt_at_z(cosmo, z)

  dLflrw = dCt * (1. + z)
  Ez = E_at_z(cosmo, z)
  ddLflrw = dCt + (cosmo.dH / Ez) * (1. + z)
  Xiz = Xi_at_z(cosmo, z)
  dXiz = cosmo.n * (cosmo.Xi0 - 1.) / ((1. + z)**(cosmo.n + 1))
  return ddLflrw * Xiz + dLflrw * dXiz

# Redshift converter
@dispatch
def z_from_dGW(cosmo, dGWs: jnp.ndarray):
  dGW_values = dL_at_z(cosmo, cosmo.z_grid_interp)
  return jnp.interp(dGWs, dGW_values, cosmo.z_grid_interp)

# Theta source dispatches
@dispatch
def ddLdz_at_z(cosmo, theta: theta_src):
  return ddLdz_at_z(cosmo, theta.z, theta.original_distances)
