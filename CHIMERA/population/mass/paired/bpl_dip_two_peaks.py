import jax.numpy as jnp
from plum import dispatch

from .base import base_mass_paired_struct, mass_pdf_notnorm, pairing_function
from ..core import tpl_notnorm, tpl_cdf, truncated_gaussian, high_pass_filter, low_pass_filter, notch_filter

class bpl_dip_two_peaks(base_mass_paired_struct):
  """
  Paired Broken Power Law + Three Gaussian Peaks with Dip (Broken Triple Peak)

    p(m) = [(1-λ_g) * BPL(m) +
              λ_g * λ_1 * G₁(m) +
              λ_g * (1-λ_1) * λ_2 * G₂(m)] *
            F_h(m; m_min, δ_m_min) *
            F_l(m; m_max, δ_m_max) *
            F_n(m; m_d_low, m_d_high, δ_d_min, δ_d_max, A)

  where:
      - BPL(m) is a broken power law with indices α₁ (low) and α₂ (high)
      - G₁, G₂ are truncated Gaussians with means μ_g1, μ_g2, μ_g3
      - F_h is the high-pass filter (smoothing at low mass end)
      - F_l is the low-pass filter (smoothing at high mass end)
      - F_n is the notch filter (creates the dip/gap)
      - Pairing function: f(m1,m2) = q^β with q=m2/m1, and β depends on q

  The break point b for the broken power law is derived from the dip position:
      b = (m_break - m_min) / (m_max - m_min)
      m_break = 0.5 * (m_d_low + m_d_high + δ_d_min - δ_d_max)

  Parameters
  ----------
  m_low : float
      Minimum source mass [M☉] (mmin in ICAROGW)
  m_high : float
      Maximum source mass [M☉] (mmax in ICAROGW)
  alpha_1 : float
      First power law index for primary mass (below break)
  alpha_2 : float
      Second power law index for primary mass (above break)
  beta_bottom : float
      First power law index for secondary mass pairing (below m_break)
  beta_top : float
      Second power law index for secondary mass pairing (above m_break)
  mu_g_low : float
      Mean of Gaussian 1 [M☉]
  sigma_g_low : float
      Std. dev. of Gaussian 1 [M☉]
  mu_g_high : float
      Mean of Gaussian 2 [M☉]
  sigma_g_high : float
      Std. dev. of Gaussian 2 [M☉]
  lambda_g : float
      Fraction of events in the Gaussians (∈ [0,1])
  lambda_1 : float
      Fraction of Gaussian events in Gaussian 1 (∈ [0,1])
  bottomsmooth : float
      Smoothing parameter at low mass end [M☉] (delta_m_min in ICAROGW)
  topsmooth : float
      Smoothing parameter at high mass end [M☉] (delta_m_max in ICAROGW)
  leftdip : float
      Left side of the dip [M☉] (m_d_low in ICAROGW)
  rightdip : float
      Right side of the dip [M☉] (m_d_high in ICAROGW)
  leftdipsmooth : float
      Smoothing of left side of the dip [M☉] (delta_d_min in ICAROGW)
  rightdipsmooth : float
      Smoothing of right side of the dip [M☉] (delta_d_max in ICAROGW)
  deep : float
      Deepness of the dip (∈ [0,1]) (A in ICAROGW)
  """

  # Model parameters
  alpha_1: float
  alpha_2: float
  beta_bottom: float  # beta_1 in PDF
  beta_top: float     # beta_2 in PDF
  mu_g_low: float
  sigma_g_low: float
  mu_g_high: float
  sigma_g_high: float
  lambda_g: float
  lambda_1: float
  bottomsmooth: float   # delta_m_min
  topsmooth: float      # delta_m_max
  leftdip: float        # m_d_low
  rightdip: float       # m_d_high
  leftdipsmooth: float  # delta_d_min
  rightdipsmooth: float # delta_d_max
  deep: float           # A
  m_break: float
  b: float
  name = 'paired_broken_double_peak_dip'

  default = {
    **base_mass_paired_struct.default,
    'm_low': 0.961834913949035,
    'm_high': 96.86391602156012,
    'alpha_1': 2.1553739953196223,
    'alpha_2': 1.848352853466305,
    'beta_bottom': 1.500961280411781,
    'beta_top': 2.5908978874521305,
    'mu_g_low': 9.056012551383695,
    'sigma_g_low': 0.6948024105466937,
    'mu_g_high': 27.134217763853904,
    'sigma_g_high': 8.384405984258168,
    'lambda_g': 0.2651222683400193,
    'lambda_1': 0.7091553393032455,
    'bottomsmooth': 0.07554021010270083,
    'topsmooth': 0.0991675925989498,
    'leftdip': 2.2922222205817913,
    'rightdip': 6.875503912163882,
    'leftdipsmooth': 0.12602841524720307,
    'rightdipsmooth': 0.1253143500,
    'deep': 0.4849206386665612
  }
  @property
  def m_break(self):
    return 0.5 * (self.leftdip + self.leftdipsmooth + self.rightdip - self.rightdipsmooth)

  @property
  def b(self):
    return (self.m_break - self.m_low) / (self.m_high - self.m_low)

# ---------------------------------------------------------------------------
# Marginal shape (same for m1 and m2)
# ---------------------------------------------------------------------------

def _broken_power_law(m, alpha_1, alpha_2, b, m_low, m_high):
    m_break = m_low + b * (m_high - m_low)
    pl_low_at_break = m_break ** (-alpha_1)
    pl_high_at_break = m_break ** (-alpha_2)
    scale_factor = pl_low_at_break / pl_high_at_break

    pl_low = tpl_notnorm(m, -alpha_1, m_low, m_break)
    pl_high = tpl_notnorm(m, -alpha_2, m_break, m_high)

    bpl_unnorm =  pl_low + scale_factor * pl_high

    norm_low = tpl_cdf(m_break, -alpha_1, m_low)
    norm_high = tpl_cdf(m_high, -alpha_2, m_break)
    bpl_norm = norm_low + scale_factor * norm_high

    return bpl_unnorm / bpl_norm

@dispatch
def mass_pdf_notnorm(mass: bpl_dip_two_peaks, m: jnp.ndarray) -> jnp.ndarray:
    """
    Un-normalised marginal mass PDF p(m) for the Broken Triple Peak model.
    p(m) = [(1-λ_g) * BPL(m) + λ_g * λ_1 * G₁(m) + λ_g * (1-λ_1) * λ_2 * G₂(m) + λ_g * (1-λ_1) * (1-λ_2) * G₃(m)] * F_h(m) * F_l(m) * F_n(m)
    """
    # 1. Broken Power Law component (normalized between m_low and m_high)
    bpl = _broken_power_law(m, mass.alpha_1, mass.alpha_2, mass.b, mass.m_low, mass.m_high)

    # 2. Gaussian components (truncated between m_low and m_high)
    g1 = truncated_gaussian(m, mass.mu_g_low, mass.sigma_g_low, mass.m_low, mass.mu_g_low + 5*mass.sigma_g_low)
    g2 = truncated_gaussian(m, mass.mu_g_high, mass.sigma_g_high, mass.m_low, mass.mu_g_high + 5*mass.sigma_g_high)

    # 3. Combine components
    # λ_g is the fraction of events in the Gaussians
    # λ_1 is the fraction of Gaussian events in Gaussian 1
    # Gaussian 2 gets the remaining fraction
    gaussian_component = (
      mass.lambda_g * mass.lambda_1 * g1 +
      mass.lambda_g * (1.0 - mass.lambda_1) * g2
    )

    pdf = (1.0 - mass.lambda_g) * bpl + gaussian_component

    # 4. Apply filters
    pdf *= high_pass_filter(m, mass.bottomsmooth, mass.m_low) # High-pass filter (smoothing at low mass end)
    pdf *= low_pass_filter(m, mass.topsmooth, mass.m_high) # Low-pass filter (smoothing at high mass end)
    pdf *= notch_filter(  # Notch filter (creates the dip/gap)
      m,
      mass.leftdip, mass.rightdip,
      mass.leftdipsmooth, mass.rightdipsmooth,
      mass.deep
    )
    peak_ordering_condition = mass.mu_g_low <= mass.mu_g_high
    return jnp.where(peak_ordering_condition, pdf, jnp.nan)

@dispatch
def pairing_function(mass: bpl_dip_two_peaks,
                     m1: jnp.ndarray,
                     m2: jnp.ndarray) -> jnp.ndarray:
    """
    Pairing function f(m1, m2) with two power law indices.

    f(m1, m2) = (m2/m1)^β₁ for m2 < m_break
    f(m1, m2) = (m2/m1)^β₂ for m2 ≥ m_break

    where the break point is determined by the dip position.
    This models binaries that include at least one neutron star (NS) or black hole (BH).
    """
    q = m2 / m1
    mask = q <= 1.0
    beta = jnp.where(m2 < mass.m_break, mass.beta_bottom, mass.beta_top)
    return jnp.where(mask, jnp.power(q, beta), 0.0)
