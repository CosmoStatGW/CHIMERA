from .core.jax_config import jax, jnp, logger, trapz
from .cosmo import dVcdz_at_dz, z_from_dGW, ddLdz_at_z, Vc_at_z
from .rate import compute_rate
from .mass import pdf_m1m2

import equinox as eqx
from typing import Callable, Optional, Dict

class bias(eqx.Module):

    # inj data
    inj_data: Dict[str, jnp.ndarray]
    inj_prior: jnp.ndarray
    N_inj: int
    
    # population models
    cosmo_model: eqx.Module
    mass_model: eqx.Module
    rate_model: eqx.Module

    p_bkg: Optional[Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]] = lambda cosmo,z: dVcdz_at_dz(cosmo,z,distances=self.inj_data['dL'])

    # check Neff
    neff_inj_min: Optional[float] = 5.
    
    # normalization rate
    z_det_range: Optional[jnp.ndarray] = None
    z_int_res_bias: Optional[float] = None
    Tobs: Optional[float] = 1.0

    # other settings
    max_z_inj: Optional[float] = 2.0

    def __post_init__(self):
       object.__setattr__(self, 'cosmo_keys', self.cosmo_model.cosmo_keys)
       object.__setattr__(self, 'mass_keys', self.mass_model.mass_keys)
       object.__setattr__(self, 'rate_keys', self.rate_model.rate_keys)

       object.__setattr__(self, 'fiducials_cosmo', self.cosmo_model.get_fiducial_params())
       object.__setattr__(self, 'fiducials_mass', self.mass_model.get_fiducial_params())
       object.__setattr__(self, 'fiducials_rate', self.rate_model.get_fiducial_params()) 

       object.__setattr__(self, 'check_Neff', self.neff_inj_min is not None)
       object.__setattr__(self, 'normalized', self.z_det_range is not None)
       object.__setattr__(self, 'use_dVdz', self.p_bkg is None)

       logger.info(f'Created bias model')

    def update_models(self, **hyper_params):
        lc = {k: jnp.atleast_1d(v) for k, v in hyper_params.items() if k in self.cosmo_keys}
        lm = {k: jnp.atleast_1d(v)  for k, v in hyper_params.items() if k in self.mass_keys}
        lr = {k: jnp.atleast_1d(v)  for k, v in hyper_params.items() if k in self.rate_keys}

        max_size = max(
            max(len(v) for v in lc.values()) if lc else 1,
            max(len(v) for v in lm.values()) if lm else 1,
            max(len(v) for v in lr.values()) if lr else 1,
        )
        
        for model_params, fiducials, keys in zip([lc, lm, lr], 
                                                 [self.fiducials_cosmo, self.fiducials_mass, self.fiducials_rate], 
                                                 [self.cosmo_keys, self.mass_keys, self.rate_keys]):
            for key in keys:
                value = jnp.asarray(model_params.get(key, fiducials[key]))
                if value.size == 1 and max_size > 1:
                    model_params[key] = jnp.full((max_size,), value)
                elif key not in model_params:
                    model_params[key] = jnp.full((max_size,), fiducials[key])
                else:
                    if model_params[key].size == 1:
                        model_params[key] = jnp.full((max_size,), model_params[key])

        cosmo_model = self.cosmo_model.from_params(**lc)
        mass_model = self.mass_model.from_params(**lm)
        rate_model = self.rate_model.from_params(**lr)
    
        return cosmo_model, mass_model, rate_model

    def get_rate(self, **hyper_params):
        
        cosmo_model, mass_model, rate_model = self.update_models(**hyper_params)
        
        dL    = self.inj_data['dL']
        m1det = self.inj_data["m1det"]
        m2det = self.inj_data["m2det"]

        z_inj   = jax.vmap(z_from_dGW, in_axes = (0,None,None))(cosmo_model, dL, self.max_z_inj)
        m1s_inj, m2s_inj = m1det/(1.+z_inj), m2det/(1.+z_inj)
        dNdtheta = self.Tobs*jax.vmap(pdf_m1m2, in_axes=(0,0,0))(mass_model, m1s_inj, m2s_inj)*\
                   jax.vmap(compute_rate, in_axes=(0,0))(rate_model, z_inj)/(1.+z_inj)*\
                   jax.vmap(self.p_bkg, in_axes=(0,0))(cosmo_model, z_inj)/\
                   (jax.vmap(ddLdz_at_z, in_axes=(0,0,None))(cosmo_model, z_inj, dL)*(1.+z_inj)**2) /\
                   self.inj_prior 

        if self.normalized:
            norm = self.N_exp(**hyper_params)
            dNdtheta /= norm

        return dNdtheta

    def compute(self, **hyper_params):

        dNdtheta = self.get_rate(**hyper_params)
        xi   = jnp.sum(dNdtheta, axis = -1) / self.N_inj
        
        s2   = jnp.sum(dNdtheta**2, axis = -1) / self.N_inj**2  - xi**2 / self.N_inj
        neff = xi**2 / s2
        if self.check_Neff:
            neff_cond = jnp.atleast_1d(neff) < self.neff_inj_min
            xi        = jnp.where(neff_cond, 0.0, xi)        
        return xi

    def N_exp(self, **hyper_params):
        # to be checked
        cosmo_model, mass_model, rate_model = self.update_models(**hyper_params)
        zz    = jnp.linspace(*self.z_det_range, self.z_int_res_bias)
        dN_dz = jax.vmap(compute_rate, in_axes = (0,None))(rate_model, zz)/(1.+zz) *\
                jax.vmap(self.p_bkg, in_axes=(0,None))(cosmo_model, zz)
        res   = trapz(dN_dz, x = zz, axis = -1)
        return res