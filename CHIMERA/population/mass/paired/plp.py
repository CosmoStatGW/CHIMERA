import jax.numpy as jnp
from plum import dispatch

from .base import base_mass_paired_struct, mass_pdf_notnorm, pairing_function
from ..core import truncated_pl, truncated_gaussian, high_pass_filter


class plp(base_mass_paired_struct):
  r"""Paired Power-Law + Gaussian Peak (PLP) mass model.

  Joint distribution:

    p(m1, m2) = p̃(m1) * p̃(m2) * f(m1, m2) / Z ,   m2 <= m1

  with the *same* marginal shape for both masses:

    p̃(m) = [(1 - λ_peak) * PL(m) + λ_peak * G(m)] * S(m; m_low, δ_m)

  where:
    - PL(m) = m^{-α} / tpl_cdf(m_high; -α, m_low)  is a truncated power law,
    - G(m) is a truncated Gaussian with mean μ_g and width σ_g,
    - S(m; m_low, δ_m) is the left-side smoothing window of the LVK,

  and the pairing function is a power law in the mass ratio:

    f(m1, m2) = (m2/m1)^β * θ(m1 - m2)

  The 2-D normalisation constant Z is computed numerically by the base class.

  Parameters
  ----------
  lambda_peak : float
    Fraction of the distribution in the Gaussian peak.
  alpha : float
    Spectral index of the power-law component (positive → falling spectrum).
  beta : float
    Mass-ratio power-law index of the pairing function.
  delta_m : float
    Smoothing scale at the low-mass end (solar masses).
  m_low : float
    Minimum mass (solar masses).
  m_high : float
    Maximum mass (solar masses).
  mu_g : float
    Mean of the Gaussian peak (solar masses).
  sigma_g : float
    Standard deviation of the Gaussian peak (solar masses).
  """

  lambda_peak: float
  alpha: float
  beta: float
  delta_m: float
  mu_g: float
  sigma_g: float

  default = {
    **base_mass_paired_struct.default,
    'm_low':       5.26,
    'm_high':      94.4,
    'lambda_peak': 0.04,
    'alpha':       3.6,
    'beta':        1.08,
    'delta_m':     3.3,
    'mu_g':        28.6,
    'sigma_g':     5.15,
  }
  name = 'paired_power_law_plus_peak'


# ---------------------------------------------------------------------------
# Marginal shape (same for m1 and m2)
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm(mass: plp, m: jnp.ndarray) -> jnp.ndarray:
  """p̃(m) = [(1-λ) PL(m) + λ G(m)] * S(m).

  The power-law component is internally normalised (divided by its own CDF
  at m_high) so that the mixing fraction λ_peak is meaningful before the
  final 2-D normalisation Z is applied.
  """
  # Normalised truncated power law
  P = truncated_pl(m, -mass.alpha, mass.m_low, mass.m_high) 

  # Truncated Gaussian peak (truncated between m_low and μ + 5σ for speed)
  G = truncated_gaussian(m, mass.mu_g, mass.sigma_g, mass.m_low, mass.mu_g + 5.0 * mass.sigma_g)

  pdf = (1.0 - mass.lambda_peak) * P + mass.lambda_peak * G
  return pdf


# ---------------------------------------------------------------------------
# Pairing function  f(m1, m2) = q^β,  q = m2/m1,  for q <= 1
# ---------------------------------------------------------------------------

@dispatch
def pairing_function(mass: plp,
                     m1: jnp.ndarray,
                     m2: jnp.ndarray) -> jnp.ndarray:
  """f(m1, m2) = (m2/m1)^β  for m2 <= m1, else 0.

  The power-law index β encodes the preference for equal-mass (β > 0) or
  unequal-mass (β < 0) binaries.  β = 0 gives a flat mass-ratio distribution.
  """
  q = m2 / m1
  return jnp.where(q <= 1.0, jnp.power(q, mass.beta), 0.0)
