import jax
import jax.numpy as jnp

#########################
# Useful math functions #
#######################

# Truncated power law

def smooth_step_up(x, threshold, steepness):
    return jax.nn.sigmoid(steepness*(x-threshold))

def smooth_step_down(x, threshold, steepness):
    return jax.nn.sigmoid(-steepness*(x-threshold))

def _truncated_pl_notnorm(x, alpha, xmin, xmax, steepness=200.0):
  lower_bound = smooth_step_up(x, xmin, steepness)
  upper_bound = smooth_step_down(x, xmax, steepness)
  return x**alpha * lower_bound * upper_bound 

def _pl_norm(alpha, xmin, xmax):
    u = 1.0 + alpha
    is_zero = u == 0.

    # Inner where: substitute zero denominator with dummy value (1.0) to prevent NaN in unselected branch
    safe_u = jnp.where(is_zero, 1.0, u)

    val_power = (xmax**safe_u - xmin**safe_u) / safe_u
    val_limit = jnp.log(xmax / xmin)

    # Outer where: pick exact log limit at u = 0, power-law integral otherwise
    return jnp.where(is_zero, val_limit, val_power)
  
def truncated_pl(m, alpha, xmin, xmax, steepness=200.0):
  pdf = _truncated_pl_notnorm(m, alpha, xmin, xmax, steepness)
  pdf /= _pl_norm(alpha, xmin, xmax) 
  return pdf

def broken_pl(m, alpha_1, alpha_2, b, m_low, m_high, steepness=200.0):
    m_break = m_low + b * (m_high - m_low)
    
    # Compute the two components using smooth truncated PL
    pl_low_notnorm = _truncated_pl_notnorm(m, -alpha_1, m_low, m_break, steepness)
    pl_high_notnorm = _truncated_pl_notnorm(m, -alpha_2, m_break, m_high, steepness)
        
    # Get values at break point
    scale_factor  = m_break**(-alpha_1) / m_break**(-alpha_2)
   
    # Combine components
    pdf = pl_low_notnorm + scale_factor * pl_high_notnorm
    norm = _pl_norm(-alpha_1, m_low, m_break) + scale_factor*_pl_norm(-alpha_2, m_break, m_high)
    pdf /= norm
    return pdf


def high_pass_filter(m, delta_m, m_low):
    below = m <= m_low
    above = m >= (m_low + delta_m)
    transition = ~below & ~above

    # Safe evaluation of terms: map non-transition indices to safe dummy values (0.5 * delta_m offset)
    m_shifted = jnp.where(transition, m - m_low, 0.5 * delta_m)
    
    term1 = delta_m / m_shifted
    term2 = delta_m / (m_shifted - delta_m)
    z = term1 + term2

    log_smoothing_mid = -jnp.logaddexp(0.0, z)

    log_smoothing = jnp.where(
        below, 
        -jnp.inf,
        jnp.where(above, 0.0, log_smoothing_mid)
    )
    return jnp.exp(log_smoothing)


def low_pass_filter(m, delta_m, m_high):
    above = m >= m_high
    below = m <= (m_high - delta_m)
    transition = ~above & ~below

    # Safe evaluation of terms: map non-transition indices to safe dummy values (0.5 * delta_m offset)
    m_shifted = jnp.where(transition, m_high - m, 0.5 * delta_m)

    term1 = delta_m / m_shifted
    term2 = delta_m / (m_shifted - delta_m)
    z = term1 + term2

    log_smoothing_mid = -jnp.logaddexp(0.0, z)

    log_smoothing = jnp.where(
        above, 
        -jnp.inf,
        jnp.where(below, 0.0, log_smoothing_mid)
    )
    return jnp.exp(log_smoothing)
def notch_filter(m, m_low, m_high, delta_m_low, delta_m_high, A):
    F_high = high_pass_filter(m, delta_m_low, m_low)
    F_low = low_pass_filter(m, delta_m_high, m_high)
    return jnp.exp(jnp.log(1.0 - A * F_high * F_low))

# Gaussian distributions
def gaussian(x, mu, sigma):
  log_G = -0.5*jnp.log(2 * jnp.pi) - jnp.log(sigma) - (x-mu)**2/(2.*sigma**2)
  return jnp.exp(log_G)

def truncated_gaussian(x, mu, sigma, x_min, x_max, steepness = 200):
  max_point = (x_max-mu)/(sigma*jnp.sqrt(2.))
  min_point = (x_min-mu)/(sigma*jnp.sqrt(2.))
  norm = 0.5*jax.scipy.special.erf(max_point)-0.5*jax.scipy.special.erf(min_point)
  
  lower_bound = smooth_step_up(x, x_min, steepness)
  upper_bound = smooth_step_down(x, x_max, steepness)
  g = gaussian(x, mu, sigma)
  return g * lower_bound * upper_bound / norm
