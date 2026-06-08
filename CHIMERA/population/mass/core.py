import jax
import jax.numpy as jnp

#########################
# Useful math functions #
#######################

# Truncated power law 
"""
def tpl_notnorm(m, alpha, m_low, m_high):
  # not normalized
  return jnp.where((m_low <= m) & (m <= m_high),
    m**alpha,
    0.
  )
"""

def smooth_step_up(x, threshold, steepness):
    return jax.nn.sigmoid(steepness*(x-threshold))

def smooth_step_down(x, threshold, steepness):
    return jax.nn.sigmoid(-steepness*(x-threshold))

def tpl_notnorm(x, alpha, xmin, xmax, steepness=200.0):
  # not normalized
  lower_bound = smooth_step_up(x, xmin, steepness)
  upper_bound = smooth_step_down(x, xmax, steepness)
  return x**alpha * lower_bound * upper_bound

def tpl_cdf(m, alpha, m_low):
  # not normalized. if m = m_high the result is the normalization of the pdf
  return jnp.where(alpha==-1,
    jnp.log(m_low) - jnp.log(m),
    (m**(1 + alpha) - m_low**(1 + alpha)) / (1 + alpha)
  )

# Smoothing function
def smoothing(m, delta_m, m_low):
  eps = 1.e-99
  log_smoothing = jnp.where(m < m_low,
    -jnp.inf,
    jnp.where(m > (m_low + delta_m),
      0.0,
      -jnp.logaddexp(0.0, (delta_m/(m-m_low+eps) + delta_m/(m-m_low-delta_m+eps)))
    )
  )
  return jnp.exp(log_smoothing)

# Gaussian distributions
def gaussian(x, mu, sigma):
  log_G = -0.5*jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x-mu)**2/(2.*sigma**2)
  return jnp.exp(log_G)

def truncated_gaussian(x, mu, sigma, x_min, x_max):
  max_point = (x_max-mu)/(sigma*jnp.sqrt(2.))
  min_point = (x_min-mu)/(sigma*jnp.sqrt(2.))
  norm = 0.5*jax.scipy.special.erf(max_point)-0.5*jax.scipy.special.erf(min_point)
  # trunc gaussian
  return jnp.where( (x_min <= x) & (x <= x_max),
    gaussian(x, mu, sigma) / norm,
    0.
  )
