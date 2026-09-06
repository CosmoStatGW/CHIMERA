import jax.numpy as jnp
import jax
import equinox as eqx
from typing import List
from plum import dispatch
from ....utils.math import trapz, interp
from ....data import theta_src

################
# MASS PYTREES #
################

class base_mass_paired_struct(eqx.Module):
  """Base class for the paired (symmetric) 2D mass model.

  Models the joint distribution as:

    p(m1, m2) = p̃(m1) * p̃(m2) * f(m1, m2) / Z,   for m2 <= m1

  where:
    - p̃(m)  is an un-normalised marginal mass PDF (same function for m1 and m2),
    - f(m1, m2) is a pairing function (e.g. a power law in q = m2/m1),
    - Z is the 2D normalisation constant,

        Z = ∫∫_{m2<=m1} p̃(m1) p̃(m2) f(m1,m2) dm1 dm2,

      computed numerically at construction time.

  The support constraint m2 <= m1 is enforced both inside the pairing
  function (which should return 0 for q > 1) and explicitly in p_m1m2 and
  _compute_norm_2d for robustness.

  Concrete sub-classes must override ``mass_pdf_notnorm`` and
  ``pairing_function`` via plum dispatch (see ``plp.py``).

  Attributes
  ----------
  m_low, m_high : float
    Mass range.
  m_grid_res : int
    Number of grid points per axis for the 2D normalisation integral.
    Default 500  (→ 500×500 grid).
  m_grid : jnp.ndarray
    Log-spaced 1D grid.
  norm_2d : float
    Precomputed Z.
  """

  m_low: float
  m_high: float
  m_grid_res: int = eqx.field(static=True, default=500)
  m_grid: jnp.ndarray
  norm_2d: float
  default = {'m_low': 4.58, 'm_high': 86.3}
  keys: List[str] = eqx.field(static=True)
  name = 'base_mass_paired_struct'

  def __init__(self, **kwargs):
    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    # 2-D normalisation Z
    setattr(self, 'm_grid', jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res))
    setattr(self, 'norm_2d', _compute_norm_2d(self))

  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    return self.__class__(**fiducials)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_norm_2d(mass: base_mass_paired_struct) -> float:
  """
  Z = ∫∫_{m2<=m1} p̃(m1) p̃(m2) f(m1,m2) dm1 dm2.

  Uses the stored ``m_grid`` (shape N) and evaluates a full N×N 2-D
  integrand, then collapses it with two successive trapz calls.

  Axes convention:
    axis-0  →  m1 (outer integral)
    axis-1  →  m2 (inner integral)
  """
  g  = mass.m_grid                        # (N,)
  pm = mass_pdf_notnorm(mass, g)  # (N,) unnorm
  # Outer products: (N,1) * (1,N) → (N,N)
  pm1_2d = pm[:, None]
  pm2_2d = pm[None, :]
  m1_2d  = g[:, None]
  m2_2d  = g[None, :]
  fval = pairing_function(mass, m1_2d, m2_2d)          # (N, N)
  # Enforce support: m2 <= m1
  fval = jnp.where(m2_2d <= m1_2d, fval, 0.0)
  integrand = pm1_2d * pm2_2d * fval                   # (N, N)

  # ∫ dm2 for each fixed m1  → shape (N,)
  inner = trapz(integrand, x=g, axis=1)
  # ∫ dm1
  return trapz(inner, x=g)                             # scalar

# ---------------------------------------------------------------------------
# Dispatch stubs – MUST be overridden by concrete subclasses
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm(mass: base_mass_paired_struct,  m: jnp.ndarray) -> jnp.ndarray:
  """Un-normalised 1-D marginal mass PDF p̃(m).

  This is the *same* functional form used for both m1 and m2.  Concrete
  subclasses implement this via plum dispatch specialised on their type.
  """
  raise NotImplementedError(
    "mass_pdf_notnorm not implemented for base_mass_paired_struct."
  )


@dispatch
def pairing_function(mass: base_mass_paired_struct,
                     m1: jnp.ndarray,
                     m2: jnp.ndarray) -> jnp.ndarray:
  """Pairing function f(m1, m2).

  Encodes the preference for certain mass-ratio combinations.  Should:
    - be non-negative,
    - return 0 for m2 > m1 (i.e. q = m2/m1 > 1).

  Concrete subclasses implement this via plum dispatch.
  """
  raise NotImplementedError(
    "pairing_function not implemented for base_mass_paired_struct."
  )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dispatch
def p_m1m2(mass: base_mass_paired_struct,
          m1: jnp.ndarray,
          m2: jnp.ndarray) -> jnp.ndarray:
  """Joint paired mass PDF  p(m1, m2) = p̃(m1) p̃(m2) f(m1,m2) / Z.

  Parameters
  ----------
  mass :
    Model instance.
  m1, m2 :
    Primary and secondary source-frame masses (arbitrary broadcastable shapes).

  Returns
  -------
  jnp.ndarray
      Joint PDF; exactly zero where m2 > m1.
  """
  pm1  = mass_pdf_notnorm(mass, m1)
  pm2  = mass_pdf_notnorm(mass, m2)
  fval = pairing_function(mass, m1, m2)
  fval = jnp.where(m2 <= m1, fval, 0.0)           # enforce support
  Z    = jnp.maximum(mass.norm_2d, jnp.finfo(jnp.float_).tiny)
  return pm1 * pm2 * fval / Z


@dispatch
def p_m1m2(mass: base_mass_paired_struct, theta: theta_src) -> jnp.ndarray:
  return p_m1m2(mass, theta.m1src, theta.m2src)


@dispatch
def p_m1(mass: base_mass_paired_struct, m: jnp.ndarray) -> jnp.ndarray:
  """True marginal primary-mass PDF.

  Obtained by analytically marginalising the joint over m2:

    p(m1) = p̃(m1) * [∫_{m_low}^{m1} p̃(m2) f(m1,m2) dm2] / Z

  Note: this is NOT simply p̃(m1) / ∫p̃ dm because the pairing function
  couples m1 and m2.  The inner integral is evaluated by interpolation on
  a precomputed profile to avoid a vmap over the query points (which can be
  expensive inside a JAX-jitted context).

  The inner integral profile I(m1) = ∫_{m_low}^{m1} p̃(m2) f(m1,m2) dm2
  is approximated on ``mass.m_grid`` and then linearly interpolated.
  """
  g  = mass.m_grid                        # (N,)
  pm = mass_pdf_notnorm(mass, g)  # (N,)
  # I(m1_i) = ∫_{m_low}^{m1_i} p̃(m2) f(m1_i, m2) dm2
  # Compute for every grid point m1_i.
  # Shape of fval:  f(m1_grid[:, None], m2_grid[None, :]) → (N, N)
  m1_2d  = g[:, None]                               # (N, 1)  ← m1 axis
  m2_2d  = g[None, :]                               # (1, N)  ← m2 axis
  fval   = pairing_function(mass, m1_2d, m2_2d)    # (N, N)
  fval   = jnp.where(m2_2d <= m1_2d, fval, 0.0)   # m2 <= m1 mask
  # inner integral profile: (N,)
  I_grid = trapz(pm[None, :] * fval, x=g, axis=1)
  # Interpolate I at queried m1 values
  m1_arr = jnp.atleast_1d(jnp.asarray(m, dtype=jnp.float_))
  I_m1   = interp(m1_arr, g, I_grid)
  pm1_unnorm = mass_pdf_notnorm(mass, m1_arr)
  Z          = jnp.maximum(mass.norm_2d, jnp.finfo(jnp.float_).tiny)
  result     = pm1_unnorm * I_m1 / Z
  return result.squeeze() if jnp.ndim(m) == 0 else result


@dispatch
def p_m1(mass: base_mass_paired_struct, theta: theta_src) -> jnp.ndarray:
  return p_m1(mass, theta.m1src)


# ---------------------------------------------------------------------------
# Utility: joint + marginals on a 2D grid (for plotting / sanity checks)
# ---------------------------------------------------------------------------

def pdf_joint_and_marg(mass: base_mass_paired_struct,
                       res=(5000, 2500)) -> dict:
  """Evaluate the joint PDF and its marginals on a dense grid.

  Parameters
  ----------
  mass :
    Model instance.
  res :
    ``(n_m1, n_m2)`` resolution of the output grid.

  Returns
  -------
  dict with keys:
    ``m1``, ``m2``, ``m1mesh``, ``m2mesh``,
    ``p_joint``, ``p_m1_marg``, ``p_m2_marg``.
  """
  m1 = jnp.linspace(mass.m_low, mass.m_high, res[0])
  m2 = jnp.linspace(mass.m_low, mass.m_high, res[1])
  m1mesh, m2mesh = jnp.meshgrid(m1, m2)          # (n_m2, n_m1)
  p_joint = p_m1m2(mass, m1mesh, m2mesh)
  p1_marg = trapz(p_joint, x=m2, axis=0)
  p1_marg = p1_marg / jnp.maximum(trapz(p1_marg, x=m1),
                                  jnp.finfo(jnp.float_).tiny)
  p2_marg = trapz(p_joint, x=m1, axis=1)
  p2_marg = p2_marg / jnp.maximum(trapz(p2_marg, x=m2),
                                  jnp.finfo(jnp.float_).tiny)

  dict_to_ret = {'m1':m1, 'm2':m2, 'm1mesh':m1mesh, 'm2mesh':m2mesh,
    'p_joint': p_joint, 'p_m1_marg': p1_marg, 'p_m2_marg': p2_marg}
  return dict_to_ret
