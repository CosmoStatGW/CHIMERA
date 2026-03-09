"""Mass functions designed for MBHBs and EMRIs.

Two parametrizations, both returning p(m1_src, m2_src) in linear space
so they plug into CHIMERA's pop_wrapper without modification.

  dpl_logm1m2   : p(log_m1) x p(log_m2 | log_m1)  [conditional power law]
  dpl_logm1_logq: p(log_m1) x p(log_q)            [independent power law in q=m2/m1]

Jacobians (absorbed internally in these funcitons):
  (log_m1, log_m2) -> (m1, m2): |J| = m1 * m2 * ln10^2
  (log_m1, log_q)  -> (m1, m2): |J| = m1^2 * q * ln10^2

"""

from ..utils.config import jnp
from ..utils.math import cumtrapz, trapz
import equinox as eqx
from typing import List
from numbers import Number
from plum import dispatch
from ..data import theta_src


#######################
# Core mass functions #
#######################

def smooth_dpl_logspace(x, alpha_low, alpha_high, x_break, delta):
    """Smooth double power law in x = log10(m), unnormalised.

    p(x) ~ [1-S(x)] x^{-alpha_low} + S(x) x_break^{alpha_high-alpha_low} x^{-alpha_high}
    S(x) = sigmoid((x - x_break) / delta)

    alpha_low must be negative for a rising slope below the break.
    """
    c     = x_break ** (alpha_high - alpha_low)
    log_S = -jnp.logaddexp(0., -(x - x_break) / delta)
    S     = jnp.exp(log_S)
    return (1. - S) * x ** (-alpha_low) + S * c * x ** (-alpha_high)


def powerlaw_logspace(y, beta, y_min, y_max):
    """Power law p(y) ~ y^beta for y in [y_min, y_max], zero outside."""
    return jnp.where((y_min <= y) & (y <= y_max), y ** beta, 0.)


def _powerlaw_logspace_cdf(beta, y_min, y_top):
    """CDF of powerlaw_logspace from y_min to y_top."""
    return jnp.where(
        beta == -1.,
        jnp.log(y_top) - jnp.log(y_min),
        (y_top ** (beta + 1.) - y_min ** (beta + 1.)) / (beta + 1.)
    )



class base_mass_logspace(eqx.Module):
    """Base for MBHB mass models in log10 space.

    Primary normalisation is precomputed at init via trapz on a log-grid.
    Subclasses define the specific parametrization; p_m1m2 returns p(m1,m2) in linear space.

    Default mass range [1e3, 1e9] Msun covers both EMRI and MBHB populations.
    """
    m_low:    float
    m_high:   float
    grid_res: int         = eqx.field(static=True)
    logm_grid:    jnp.ndarray
    norm_p_logm1: float
    keys: List[str]       = eqx.field(static=True)
    default = {'m_low': 1e3, 'm_high': 1e9, 'grid_res': 1000}
    name    = 'base_mass_logspace'

    def __init__(self, **kwargs):
        self.keys = list(self.default.keys())
        for key in self.keys:
            setattr(self, key, kwargs.get(key, self.default[key]))
        _get_normalizations_logspace(self)

    @property
    def as_dict(self):
        return {k: getattr(self, k) for k in self.keys}

    def update(self, **kwargs):
        keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
        if not keys_to_update:
            return self
        fiducials = self.as_dict
        fiducials.update(keys_to_update)
        return self.__class__(**fiducials)


def _get_normalizations_logspace(mass: base_mass_logspace):
    logm_low  = jnp.log10(mass.m_low)
    logm_high = jnp.log10(mass.m_high)
    logm_grid = jnp.linspace(logm_low, logm_high, mass.grid_res)
    setattr(mass, 'logm_grid', logm_grid)
    p_vals = _primary_logspace_notnorm(mass, logm_grid)
    norm   = trapz(p_vals, x=logm_grid)
    setattr(mass, 'norm_p_logm1', norm)



#######################
#       Models        #
#######################

class dpl_logm1m2(base_mass_logspace):
    """Smooth DPL in (log10_m1, log10_m2).

    p(log_m1) = smooth_dpl_logspace(log_m1; alpha_low, alpha_high, x_break, delta)
    p(log_m2 | log_m1) ~ (log_m2)^beta  for log_m2 in [log_m_low, log_m1]

    Output: p(m1, m2) = p(log_m1, log_m2) / (m1 * m2 * ln10^2)
    """
    alpha_low:  float
    alpha_high: float
    x_break:    float
    delta:      float
    beta:       float
    default = {
        **base_mass_logspace.default,
        'alpha_low':  -7.3,
        'alpha_high':  13.2,
        'x_break':     5.28,
        'delta':       0.1,
        'beta':        1.0,
    }
    name = 'dpl_logm1m2'


class dpl_logm1_logq(base_mass_logspace):
    """Smooth DPL in log10_m1, independent power law in log10_q.

    p(log_m1) = smooth_dpl_logspace(log_m1; alpha_low, alpha_high, x_break, delta)
    p(log_q)  ~ q^beta_q  for q = m2/m1 in [q_min, 1]

    Effective lower bound: q_min_eff = max(q_min, m_low/m1) ensures m2 >= m_low.
    Normalisation: (1 - q_min_eff^beta_q) / (beta_q * ln10)

    Output: p(m1, m2) = p(log_m1, log_q) / (m1^2 * q * ln10^2)
    """
    alpha_low:  float
    alpha_high: float
    x_break:    float
    delta:      float
    beta_q:     float
    q_min:      float
    default = {
        **base_mass_logspace.default,
        'alpha_low':  -7.3,
        'alpha_high':  13.2,
        'x_break':     5.28,
        'delta':       0.1,
        'beta_q':      1.0,
        'q_min':       0.1,
    }
    name = 'dpl_logm1_logq'


@dispatch
def _primary_logspace_notnorm(mass: dpl_logm1m2, logm: jnp.ndarray):
    return smooth_dpl_logspace(logm, mass.alpha_low, mass.alpha_high, mass.x_break, mass.delta)

@dispatch
def _primary_logspace_notnorm(mass: dpl_logm1_logq, logm: jnp.ndarray):
    return smooth_dpl_logspace(logm, mass.alpha_low, mass.alpha_high, mass.x_break, mass.delta)


@dispatch
def p_m1m2(mass: dpl_logm1m2, m1: jnp.ndarray, m2: jnp.ndarray):
    logm1    = jnp.log10(m1)
    logm2    = jnp.log10(m2)
    logm_low = jnp.log10(mass.m_low)

    p_logm1 = smooth_dpl_logspace(logm1, mass.alpha_low, mass.alpha_high,
                                   mass.x_break, mass.delta) / mass.norm_p_logm1

    p_logm2_nn = powerlaw_logspace(logm2, mass.beta, logm_low, logm1)
    cdf_m2     = _powerlaw_logspace_cdf(mass.beta, logm_low, logm1)
    p_logm2    = p_logm2_nn / jnp.where(cdf_m2 > 0., cdf_m2, 1.)
    p_logm2    = jnp.where(cdf_m2 > 0., p_logm2, 0.)

    return jnp.where(m2 < m1, p_logm1 * p_logm2 / (m1 * m2 * jnp.log(10.)**2), 0.)


@dispatch
def p_m1m2(mass: dpl_logm1_logq, m1: jnp.ndarray, m2: jnp.ndarray):
    logm1 = jnp.log10(m1)
    q     = m2 / m1

    p_logm1 = smooth_dpl_logspace(logm1, mass.alpha_low, mass.alpha_high,
                                   mass.x_break, mass.delta) / mass.norm_p_logm1

    q_min_eff = jnp.maximum(mass.q_min, mass.m_low / m1)
    norm_logq = (1. - q_min_eff ** mass.beta_q) / (mass.beta_q * jnp.log(10.))

    p_logq_nn = jnp.where((q >= q_min_eff) & (q <= 1.), q ** mass.beta_q, 0.)

    p_logq    = p_logq_nn / jnp.where(norm_logq > 0., norm_logq, 1.)
    p_logq    = jnp.where(norm_logq > 0., p_logq, 0.)

    valid = (q >= q_min_eff) & (q <= 1.) & (m1 >= mass.m_low) & (m1 <= mass.m_high)
    return jnp.where(valid, p_logm1 * p_logq / (m1 ** 2 * q * jnp.log(10.)**2), 0.)


@dispatch
def p_m1m2(mass: dpl_logm1m2, theta: theta_src):
    return p_m1m2(mass, theta.m1src, theta.m2src)

@dispatch
def p_m1m2(mass: dpl_logm1_logq, theta: theta_src):
    return p_m1m2(mass, theta.m1src, theta.m2src)

