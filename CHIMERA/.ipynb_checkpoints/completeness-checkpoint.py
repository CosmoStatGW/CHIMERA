import equinox as eqx
from typing import Optional, Dict, List
from numbers import Number
from .utils.config import jax, jnp
from .cosmo import dVcdz_at_dz, Vc_at_z

######################################################
# (IN)COMPLETENESS FOR EMPTY CATALOG (non pixelated) #
######################################################

class zero_completeness(eqx.Module):
  def compute_P_compl(self, zgrids):
    return jnp.zeros_like(zgrids)
  def compute_fR(self):
    return 0.

##############################
# COMPLETE MICE COMPLETENESS #
##############################

class mice_completeness(eqx.Module):
  # cosmo model
  cosmo_model: eqx.Module
  # settings
  z_range: jax.Array = eqx.field(converter=jnp.asarray, default_factory=lambda: jnp.array([0.073, 1.3]))
  kind: str = "step"
  z_sig: Optional[Number] = None

  def compute_P_compl(self, zgrids):
    if self.kind == "step":
      return jnp.where(jnp.logical_and(zgrids>self.z_range[0], zgrids<self.z_range[1]), 1., 0.)
    elif self.kind == "step_smooth":
      # not tested
      t_thr= (self.z_range-self.zgrids)/self.z_sig
      return 0.5*(1+jax.scipy.special.erf(t_thr/jnp.sqrt(2)))
    else:
      raise ValueError("kind must be step or step_smooth")

  def compute_fR(self, cosmo_model, normalized=False):
    # Pcompl is constant, fR is just Vc in zrange
    res = Vc_at_z(cosmo_model, self.z_range)
    return res[1] - res[0]

  # def _fR_integrand(zz, lambda_cosmo):
  #   return np.where(zz<1.3, 1, 0)*np.array(fLCDM.dV_dz(zz, lambda_cosmo))
  # def _fR_integrated(lambda_cosmo):
  #   return quad(_fR_integrand, 0, 10, args=({"H0": 70, "Om0": 0.3}))[0]  # general

# need to implement here the mask completeness
