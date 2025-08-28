from .core.jax_config import jax, jnp
import equinox as eqx
from typing import Optional, Union, Tuple, Dict
from plum import dispatch
from functools import partial

################
# RATE PYTREES #
################

class dummy_rate(eqx.Module):

    name      = 'dummy_rate'
    rate_keys = []
    
    @classmethod
    def get_fiducial_params(cls):
        return {
            field: field_info.default
            for field, field_info in cls.__dataclass_fields__.items()
            if not field_info.metadata.get("static", False)
        }
        
    @classmethod
    def from_params(cls, **kwargs):

        fiducials = cls.get_fiducial_params()
        params = {key: jnp.asarray(kwargs.get(key, fiducials[key]))
                  for key in cls.rate_keys}
        
        max_size = max(p.size for p in params.values())
        for key, value in params.items():
            if value.size == 1 and max_size > 1:
                params[key] = jnp.full((max_size,), value)

        return cls(**params)

class power_law(dummy_rate):
    gamma: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.7)

    name      = 'power_law'
    rate_keys = ['gamma'] 
    # everything else is inherited from `dummy_rate`

class madau_dickinson(dummy_rate):
    gamma: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.7)
    kappa: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.)
    zp: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.)

    name      = 'madau_dickinson'
    rate_keys = ['gamma', 'kappa', 'zp'] 
    # everything else is inherited from `dummy_rate`

class madau_dickinson_norm(dummy_rate):
    gamma: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.7)
    kappa: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.)
    zp: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.)
    R0: jax.Array = eqx.field(converter=jnp.atleast_1d, default=100_000.)

    name      = 'madau_dickinson_norm'
    rate_keys = ['gamma', 'kappa', 'zp', 'R0'] 
    # everything else is inherited from `dummy_rate`

##################
# RATE FUNCTIONS #
##################

# dummy

@dispatch
@jax.jit
def compute_rate(rate: dummy_rate, z: jnp.ndarray):
    
    return jnp.ones_like(z)

# power law

@dispatch
@jax.jit
def compute_rate(rate: power_law, z: jnp.ndarray):
    
    return (1.+z)**rate.gamma
    
# madau dickinson

@dispatch
@jax.jit
def compute_rate(rate: madau_dickinson, z: jnp.ndarray):
    
    return (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )

# madau dickinson norm

@dispatch
@jax.jit
def compute_rate(rate: madau_dickinson_norm, z: jnp.ndarray):
    
    rate_md_not_norm = (1.+z)**rate.gamma / (1. + ( (1.+z)/(1.+rate.zp) )**(rate.gamma+rate.kappa) )
    prefactor = rate.R0 * (1. + (1.+rate.zp)**(-rate.gamma-rate.kappa))
    return prefactor*rate_md_not_norm