import equinox as eqx
from typing import Optional, Dict

from .core.jax_config import jax, jnp, logger
from .cosmo import dVcdz_at_dz, Vc_at_z

class empty_catalog(eqx.Module):

    # population models
    cosmo_model: eqx.Module
    # data
    data_gal: Optional[Dict[str, jnp.ndarray]] = None

    def precompute_pcat(self, zgrids):
        return jnp.zeros_like(zgrids)
    
    def _compute_compl(self, zgrids): 
        # not "precompute" because may be cosmologically dependent
        # cosmo_model is the update one as called by compute_pgal
        return jnp.zeros_like(zgrids)
    
    def _compute_fR(self, cosmo_model, normalized=False, dummy=True):
        # cosmo_model is the update one as called by compute_pgal
        if dummy:
            # to avoid unecessary computation
            return 0. 
        else: 
            # true formula
            zs    = jnp.array([1.3,20])
            # assuming that cosmo_model params have already been uploaded
            res   = jax.vmap(Vc_at_z, in_axes=(0,None))(cosmo_model, zs) # jax.vmap to be checked
            norm  = res[1] if normalized else 1.0
            return res[0]/norm

    def compute_pgal(self, zgrids, **hyper_params):

        cosmo_model = self.cosmo_model.from_params(**hyper_params)
                
        fR    = self._compute_fR(cosmo_model, dummy=True)        
        compl = self._compute_compl(zgrids)
        
        dVdz  = jax.vmap(dVcdz_at_dz, in_axes=(0,None))(cosmo_model, zgrids)
        
        return fR*self.pcat + (1.-compl)*dVdz