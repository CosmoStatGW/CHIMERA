from .core.jax_config import jax, jnp, trapz
import equinox as eqx
from typing import Optional, Union, Tuple, Dict
from plum import dispatch
from functools import partial

################
# MASS PYTREES #
################

class dummy_mass(eqx.Module):

    #dummy_par: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.)
    name      = 'dummy_mass'
    mass_keys = []
    
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
                  for key in cls.mass_keys}
        
        max_size = max(p.size for p in params.values())
        for key, value in params.items():
            if value.size == 1 and max_size > 1:
                params[key] = jnp.full((max_size,), value)

        return cls(**params)

class tpl(dummy_mass):
    alpha: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.4)
    beta: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1.1)
    m_low: jax.Array = eqx.field(converter=jnp.atleast_1d, default=5.1)
    m_high: jax.Array = eqx.field(converter=jnp.atleast_1d, default=87.)

    name      = 'truncated_power_law'
    mass_keys = ['alpha', 'beta', 'm_low', 'm_high'] 
    # everything else is inherited from `dummy_mass`

class bpl(dummy_mass):
    alpha_1: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1.6)
    alpha_2: jax.Array = eqx.field(converter=jnp.atleast_1d, default=5.6)
    beta: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1.1)
    delta_m: jax.Array = eqx.field(converter=jnp.atleast_1d, default=4.8)
    m_low: jax.Array = eqx.field(converter=jnp.atleast_1d, default=5.1)
    m_high: jax.Array = eqx.field(converter=jnp.atleast_1d, default=87.)
    break_fraction: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.43)

    name      = 'broken_power_law'
    mass_keys = ['alpha_1', 'alpha_2', 'beta', 'delta_m', 'm_low', 'm_high', 'break_fraction'] 
    # everything else is inherited from `dummy_mass`

class plp(dummy_mass):
    lambda_peak: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.039)
    alpha: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.4)
    beta: jax.Array = eqx.field(converter=jnp.atleast_1d, default=1.1)
    delta_m: jax.Array = eqx.field(converter=jnp.atleast_1d, default=4.8)
    m_low: jax.Array = eqx.field(converter=jnp.atleast_1d, default=5.1)
    m_high: jax.Array = eqx.field(converter=jnp.atleast_1d, default=87.)
    mu_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=34.)
    sigma_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.6)

    name      = 'power_law_plus_peak'
    mass_keys = ['lambda_peak', 'alpha', 'beta', 'delta_m', 'm_low', 'm_high', 'mu_g', 'sigma_g'] 
    # everything else is inherited from `dummy_mass`

class pl2p(dummy_mass):
    lambda_peak: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.05)
    lambda1: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.5)
    alpha: jax.Array = eqx.field(converter=jnp.atleast_1d, default=2.9)
    beta: jax.Array = eqx.field(converter=jnp.atleast_1d, default=0.9)
    delta_m: jax.Array = eqx.field(converter=jnp.atleast_1d, default=4.8)
    m_low: jax.Array = eqx.field(converter=jnp.atleast_1d, default=4.6)
    m_high: jax.Array = eqx.field(converter=jnp.atleast_1d, default=87.)
    mu1_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=33.)
    sigma1_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.)
    mu2_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=68.)
    sigma2_g: jax.Array = eqx.field(converter=jnp.atleast_1d, default=3.)

    name      = 'power_law_plus_double_peak'
    mass_keys = ['lambda_peak', 'lambda1', 'alpha', 'beta', 'delta_m', 'm_low', 'm_high', 'mu1_g', 'sigma1_g', 'mu2_g', 'sigma2_g']
    # everything else is inherited from `dummy_mass`

#######################
# Core mass functions #
#######################

# Truncated power law PDF (not normalized) and its (analytical) CDF

@jax.jit
def tpl_notnorm(m, alpha, m_low, m_high): 
    # not normalized
    return jnp.where((m_low < m) & (m < m_high), 
                      m**alpha, 
                      0.
                    )

@jax.jit
def tpl_cdf(alpha, m_low, m):
    # not normalized. if m = m_high the result is the normalization of the pdf 
    return jnp.where(alpha==-1, 
                        jnp.log(m_low) - jnp.log(m), 
                        (m**(1 + alpha) - m_low**(1 + alpha)) / (1 + alpha)
                       )

# Smoothing function

@jax.jit
def log_smoothing(m, delta_m, m_low):
    
    return jnp.where(m <= m_low,
                     -jnp.inf,
                     jnp.where(m >= (m_low + delta_m),
                               0.0,
                               -jnp.logaddexp(0.0, (delta_m / (m - m_low) + delta_m / (m - m_low - delta_m)))
                              )
                     )
    
# Smoother power law PDF for m2 used in the classes BPL, PLP, PL2P

@jax.jit
def spl_pdf(m, beta, m_low, m_high, delta_m): 
    # not normalized
    logpdf = jnp.where((m_low < m) & (m < m_high),
                       beta * jnp.log(m) + log_smoothing(m, delta_m, m_low),
                      -jnp.inf
                     )
    return jnp.exp(logpdf)

@partial(jax.jit, static_argnames = ['res'])
def spl_cdf_values(beta, m_low, m_high, delta_m, res=300):
    
    m_low_val  = m_low.at[0].get() if jnp.ndim(m_low) > 0 else m_low
    m_high_val = m_high.at[0].get() if jnp.ndim(m_high) > 0 else m_high
    
    mm = jnp.linspace(m_low_val, m_high_val, res)
    p_values = spl_pdf(mm, beta, m_low_val, m_high_val, delta_m)

    dm = mm[1:] - mm[:-1]

    cdf_values = jnp.zeros_like(mm)
    cdf_values = cdf_values.at[0].set( 0.5 * p_values[0] * (m_high_val - m_low_val))
    cdf_values = cdf_values.at[1:].set(jnp.cumsum(0.5 * (p_values[:-1] + p_values[1:]) * dm))
    
    return mm, cdf_values
    return mm[0], cdf_values[0]

# Gaussian (for PLP and PL2P)

@jax.jit
def log_gaussian(x, mu, sigma):
    return -0.5*jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x-mu)**2/(2.*sigma**2)

@jax.jit
def gaussian(x, mu, sigma):
    return jnp.exp(log_gaussian(x,mu,sigma))

##########################################################
# log_pdf_m1_notnorm and cdf_m1 for all classes, but TPL #
##########################################################

@dispatch
@jax.jit
def log_pdf_m1_notnorm(mass: bpl, m: jnp.ndarray):
    # not normalized
    m_break = mass.m_low + mass.break_fraction * (mass.m_high - mass.m_low)
    
    return jnp.where((m > mass.m_low) & (m < m_break),
                     -mass.alpha_1 * jnp.log(m) + log_smoothing(m, mass.delta_m, mass.m_low),
                      jnp.where((m < mass.m_high) & (m > m_break),
                                -mass.alpha_2 * jnp.log(m) + jnp.log(m_break) * (-mass.alpha_1 + mass.alpha_2) \
                                + log_smoothing(m, mass.delta_m, mass.m_low),
                                -jnp.inf
                               )
                    )
@dispatch
@jax.jit
def log_pdf_m1_notnorm(mass: plp, m: jnp.ndarray):
    # not normalized
    P = tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)/tpl_cdf(-mass.alpha, mass.m_low, mass.m_high)
    G = gaussian(m, mass.mu_g, mass.sigma_g)

    upper_bound = jnp.where(mass.m_high > (mass.mu_g + 10 * mass.sigma_g), mass.m_high, mass.mu_g + 10 * mass.sigma_g)
    
    condition = (m >= mass.m_low) & (m <= upper_bound)
    
    return jnp.where(condition,
                     jnp.log((1 - mass.lambda_peak) * P + mass.lambda_peak * G) + log_smoothing(m, mass.delta_m, mass.m_low),
                     -jnp.inf)

@dispatch
@jax.jit
def log_pdf_m1_notnorm(mass: pl2p, m: jnp.ndarray):
    
    P  = tpl_notnorm(m, -mass.alpha, mass.m_low, mass.m_high)/tpl_cdf(-mass.alpha, mass.m_low, mass.m_high)
    G1 = gaussian(m, mass.mu1_g, mass.sigma1_g)
    G2 = gaussian(m, mass.mu2_g, mass.sigma2_g)
    
    upper_bound = jnp.where(mass.m_high > (mass.mu2_g + 10*mass.sigma2_g), mass.m_high, mass.mu2_g + 10*mass.sigma2_g)
    condition   = (m >= mass.m_low) & (m <= upper_bound)
    
    return jnp.where(condition,
                     jnp.log((1-mass.lambda_peak)*P + mass.lambda_peak*mass.lambda1*G1 + mass.lambda_peak*(1. - mass.lambda1)*G2) + \
                     log_smoothing(m, mass.delta_m, mass.m_low),
                     -jnp.inf)

##################
# MASS FUNCTIONS #
##################

# Compute normalized PDF for m1 for each class, but TPL

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def pdf_m1(mass: Union[bpl, plp, pl2p], m1: jnp.ndarray, res:int = 250):
    #norm
    m_low  = mass.m_low.at[0].get() if mass.m_low.ndim > 0 else mass.m_low
    m_high = mass.m_high.at[0].get() if mass.m_high.ndim > 0 else mass.m_high
    
    mm = jnp.linspace(m_low, m_high, res)
    dm = mm[1] - mm[0]
    integrand_values = jnp.exp(log_pdf_m1_notnorm(mass, mm))
    norm = trapz(integrand_values, dx=dm)
    
    return jnp.exp(log_pdf_m1_notnorm(mass, m1))/norm

# Compute normalized PDF for (m1,m2) for each class, but TPL

@dispatch
@partial(jax.jit, static_argnames = ['res'])
def pdf_m1m2(mass: Union[bpl, plp, pl2p], m1: jnp.ndarray, m2: jnp.ndarray, res:int = 250):

    pm1 = pdf_m1(mass, m1, res = res)

    mm, cdf_m2_vals = spl_cdf_values(mass.beta, mass.m_low, mass.m_high, mass.delta_m, res = res)
    cdf_m2          = jnp.interp(m1, mm, cdf_m2_vals)
    
    condition = (mass.m_low < m2) & (m2 < m1) & (m1 < mass.m_high)
        
    return jnp.where(condition,
                     pm1 * spl_pdf(m2, mass.beta, mass.m_low, m1, mass.delta_m)/cdf_m2,
                     0.)

# PDF for m1 and (m1,m2) in the TPL case are computed separately (no smoothing on m2 and analytical CDF)

@dispatch
@jax.jit
def pdf_m1(mass: tpl, m: jnp.ndarray):
    norm = tpl_cdf(mass.alpha, mass.m_low, mass.m_high)
    return tpl_notnorm(m, mass.alpha, mass.m_low, mass.m_high)/norm

@dispatch
@jax.jit
def pdf_m1m2(mass: tpl, m1: jnp.ndarray, m2: jnp.ndarray):

    condition = (mass.m_low < m2) & (m2 < m1) & (m1 < mass.m_high)     
    return jnp.where(condition,
                     pdf_m1(mass, m1)*\
                     tpl_notnorm(m2, mass.beta, mass.m_low, m1)/tpl_cdf(mass.beta, mass.m_low, m1),
                     0.)

# How to vectorize over params=
# pdf_m1m2_vec = jax.vmap(pdf_m1m2, in_axes = (0,None, None)) and so on

# p_m2_given_m1 is missing!