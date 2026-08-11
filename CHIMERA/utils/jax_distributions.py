import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, erfinv

# ============================================================
# Generic JAX-compatible inverse CDF
# ============================================================

def generic_ppf(cdf, u, low, high, iterations=80,):
  """
  Generic inverse CDF using binary search.
  """
  u = jnp.asarray(u)
  is_scalar = u.ndim == 0
  if is_scalar:
    u = u.reshape(1)

  lows = jnp.full_like(u, low)
  highs = jnp.full_like(u, high)
  def body_fn(_, state):
    lows, highs = state

    mids = 0.5 * (lows + highs)
    cdf_mids = cdf(mids)

    new_lows = jnp.where(
      cdf_mids < u,
      mids,
      lows,
    )

    new_highs = jnp.where(
      cdf_mids < u,
      highs,
      mids,
    )

    return new_lows, new_highs

  lows, highs = jax.lax.fori_loop(
    0,
    iterations,
    body_fn,
    (lows, highs),
  )

  result = 0.5 * (lows + highs)

  if is_scalar:
    return result[0]

  return result

# ============================================================
# Uniform
# ============================================================

class Uniform:

  def __init__(self, low, high):
    if high <= low:
      raise ValueError(
        "high must be greater than low"
      )

    self.low = jnp.asarray(low)
    self.high = jnp.asarray(high)

  def pdf(self, x):
    x = jnp.asarray(x)

    return jnp.where(
      (x >= self.low) & (x <= self.high),
      1.0 / (self.high - self.low),
      0.0,
    )

  def cdf(self, x):
    x = jnp.asarray(x)

    return jnp.where(
      x <= self.low,
      0.0,
      jnp.where(
        x >= self.high,
        1.0,
        (x - self.low)
        / (self.high - self.low),
      ),
    )

  def ppf(self, u):
    return (
      self.low
      + u * (self.high - self.low)
    )

  def sample(self, seed, sample_shape=()):
    u = jax.random.uniform(
      key=seed,
      shape=sample_shape,
    )

    return self.ppf(u)


# ============================================================
# LogUniform
# ============================================================

class LogUniform:

  def __init__(self, low, high):
    if low <= 0:
      raise ValueError(
        "LogUniform requires low > 0"
      )

    if high <= low:
      raise ValueError(
        "high must be greater than low"
      )

    self.low = jnp.asarray(low)
    self.high = jnp.asarray(high)

    self._log_ratio = jnp.log(
      self.high / self.low
    )

  def pdf(self, x):
    x = jnp.asarray(x)

    return jnp.where(
      (x >= self.low) & (x <= self.high),
      1.0 / (x * self._log_ratio),
      0.0,
    )

  def cdf(self, x):
    x = jnp.asarray(x)

    safe_x = jnp.maximum(
      x,
      self.low,
    )

    result = (
      jnp.log(safe_x / self.low)
      / self._log_ratio
    )

    return jnp.where(
      x <= self.low,
      0.0,
      jnp.where(
        x >= self.high,
        1.0,
        result,
      ),
    )

  def ppf(self, u):
    return (
      self.low
      * jnp.exp(u * self._log_ratio)
    )

  def sample(self, seed, sample_shape=()):
    u = jax.random.uniform(
      key=seed,
      shape=sample_shape,
    )

    return self.ppf(u)


# ============================================================
# Normal
# ============================================================

class Normal:

  def __init__(self, loc, scale):
    if scale <= 0:
      raise ValueError(
        "scale must be > 0"
      )

    self.loc = jnp.asarray(loc)
    self.scale = jnp.asarray(scale)

  def pdf(self, x):
    x = jnp.asarray(x)

    z = (
      (x - self.loc)
      / self.scale
    )

    return (
      jnp.exp(-0.5 * z**2)
      / (
        self.scale
        * jnp.sqrt(2.0 * jnp.pi)
      )
    )

  def cdf(self, x):
    x = jnp.asarray(x)

    z = (
      (x - self.loc)
      / (self.scale * jnp.sqrt(2.0))
    )

    return 0.5 * (
      1.0
      + jax.scipy.special.erf(z)
    )

  def ppf(self, u):
    return (
      self.loc
      + self.scale
      * jnp.sqrt(2.0)
      * erfinv(2.0 * u - 1.0)
    )

  def sample(self, seed, sample_shape=()):
    z = jax.random.normal(
      key=seed,
      shape=sample_shape,
    )

    return self.loc + self.scale * z


# ============================================================
# Beta
# ============================================================

class Beta:

  def __init__(self, alpha, beta):
    if alpha <= 0:
      raise ValueError(
        "alpha must be > 0"
      )

    if beta <= 0:
      raise ValueError(
        "beta must be > 0"
      )

    self.alpha = jnp.asarray(alpha)
    self.beta = jnp.asarray(beta)

    self._log_beta = betaln(
      self.alpha,
      self.beta,
    )

  def pdf(self, x):
    x = jnp.asarray(x)

    safe_x = jnp.clip(
      x,
      1e-30,
      1.0 - 1e-30,
    )

    log_pdf = (
      (self.alpha - 1.0)
      * jnp.log(safe_x)
      + (self.beta - 1.0)
      * jnp.log1p(-safe_x)
      - self._log_beta
    )

    return jnp.where(
      (x >= 0.0) & (x <= 1.0),
      jnp.exp(log_pdf),
      0.0,
    )

  def cdf(self, x):
    x = jnp.asarray(x)

    safe_x = jnp.clip(
      x,
      0.0,
      1.0,
    )

    result = betainc(
      self.alpha,
      self.beta,
      safe_x,
    )

    return jnp.where(
      x <= 0.0,
      0.0,
      jnp.where(
        x >= 1.0,
        1.0,
        result,
      ),
    )

  def ppf(self, u):
    return generic_ppf(
      self.cdf,
      u,
      low=0.0,
      high=1.0,
      iterations=80,
    )

  def sample(self, seed, sample_shape=()):
    return jax.random.beta(
      key=seed,
      a=self.alpha,
      b=self.beta,
      shape=sample_shape,
    )


# ============================================================
# Truncated Normal
# ============================================================

class TruncatedNormal:

  def __init__(
    self,
    loc,
    scale,
    low,
    high,
  ):
    if scale <= 0:
      raise ValueError(
        "scale must be > 0"
      )

    if high <= low:
      raise ValueError(
        "high must be greater than low"
      )

    self.loc = jnp.asarray(loc)
    self.scale = jnp.asarray(scale)
    self.low = jnp.asarray(low)
    self.high = jnp.asarray(high)

    self._normal = Normal(
      loc=self.loc,
      scale=self.scale,
    )

    self._cdf_low = self._normal.cdf(
      self.low
    )

    self._cdf_high = self._normal.cdf(
      self.high
    )

    self._normalization = (
      self._cdf_high
      - self._cdf_low
    )

  def pdf(self, x):
    x = jnp.asarray(x)

    return jnp.where(
      (x >= self.low) & (x <= self.high),
      self._normal.pdf(x)
      / self._normalization,
      0.0,
    )

  def cdf(self, x):
    x = jnp.asarray(x)

    result = (
      self._normal.cdf(x)
      - self._cdf_low
    ) / self._normalization

    return jnp.where(
      x <= self.low,
      0.0,
      jnp.where(
        x >= self.high,
        1.0,
        result,
      ),
    )

  def ppf(self, u):
    u = jnp.asarray(u)

    normal_u = (
      self._cdf_low
      + u * self._normalization
    )

    return self._normal.ppf(
      normal_u
    )

  def sample(self, seed, sample_shape=()):
    u = jax.random.uniform(
      key=seed,
      shape=sample_shape,
    )

    return self.ppf(u)
