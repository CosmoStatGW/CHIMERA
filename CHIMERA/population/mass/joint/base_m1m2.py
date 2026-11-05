import jax.numpy as jnp
import equinox as eqx
from typing import List
from plum import dispatch

from ....utils.math import trapz
from ....data import theta_src

################
# MASS PYTREES #
################

class base_mass_joint_m1m2_struct(eqx.Module):
  r"""Base class for a genuinely joint 2-D mass model, parametrized
  directly in (m1, m2) -- unlike ``joint`` (native in chirp mass/mass
  ratio), ``paired`` (``p̃(m1) p̃(m2) f(m1,m2)/Z``), or ``conditioned``
  (``p(m1) p(m2|m1)``).

  .. math::

      p(m_1, m_2) = \frac{\exp(h(m_1,m_2))}{Z}\;
      \mathbb{1}[m_{min} \le m_2 \le m_1 \le m_{max}],
      \qquad
      Z = \int\!\!\int_{m_{min} \le m_2 \le m_1 \le m_{max}}
      \exp(h(m_1,m_2))\, dm_1\, dm_2.

  **Why native (m1, m2) at all, given ``joint`` already exists.** The
  ``joint`` family (Mc, q) gets a mask-free rectangular domain, but at the
  cost of the box not corresponding to a literal ``m_min``/``m_max`` on
  either physical mass -- a rectangle in (Mc, q) does not map onto
  ``[m_min,m_max]^2`` in (m1, m2) (a well-known subtlety: the induced
  (m1,m2) support region has curved edges set by the Mc/q box corners).
  This family trades that mask-free convenience for the standard,
  immediately interpretable mass-model parametrization used throughout the
  GW population literature: a single shared ``m_min``/``m_max`` for *both*
  masses, with the ``m2 <= m1`` ordering enforced by an explicit mask
  (exactly as the ``paired``/``conditioned`` families already do).

  **Why exp-link, not squared-link.** ``joint``/``paired`` CP-style models
  take ``p ~ h^2``, which can represent an exact zero and admits (in
  principle) a closed-form Gram-matrix normalization, but
  ``d(log h^2)/dc = 2 h'/h`` has a genuine mathematical singularity
  wherever ``h`` crosses zero -- a real risk for gradient-based samplers
  (NUTS/HMC) whenever the fit wants a near-zero density region. The
  exp-link used here, ``p ~ exp(h)``, is the standard log-linear density
  construction: ``log p = h - h_shift - log(Z)`` is smooth *everywhere*,
  since ``h`` (a finite sum of bounded B-spline basis functions times
  finite coefficients) is itself bounded and the log-sum-exp-style
  ``h_shift`` subtraction (see ``_compute_shift_and_norm_2d`` below) keeps
  every intermediate exponential well away from overflow. The price: this
  family cannot represent an exact interior zero (``exp`` is strictly
  positive), and ``Z`` has no closed form and must be evaluated
  numerically on a grid, exactly as ``joint``/``paired`` already do for
  their own (different) reasons.

  Concrete subclasses must override ``_h_m1m2`` -- the signed log-density
  kernel ``h(m1, m2)`` -- via ``plum`` dispatch. ``p_m1m2`` (registered on
  the same global dispatcher as the ``paired``/``conditioned``/``joint``
  families -- see ``CHIMERA.population.mass``) and ``pdf_joint_and_marg``
  are implemented once here, generically, in terms of ``_h_m1m2``.

  Attributes
  ----------
  m_min, m_max : float
    Shared mass truncation range, for *both* m1 and m2.
  m_grid_res : int
    Grid resolution per axis for the 2-D normalisation integral and for
    the ``h_shift`` stabiliser.
  m_grid : jnp.ndarray
    1-D grid (log-spaced) spanning ``[m_min, m_max]``, shared by both
    axes since they share the same range.
  h_shift : float
    ``max_{m2<=m1} h(m1,m2)`` over ``m_grid x m_grid`` -- a
    log-sum-exp-style stabiliser subtracted from every exponent so that
    ``notnorm = exp(h - h_shift)`` never overflows; it cancels exactly in
    ``p = notnorm / Z`` since ``Z`` is computed with the same shift.
  norm_2d : float
    Precomputed ``Z = ∫∫_{m2<=m1} exp(h(m1,m2) - h_shift) dm1 dm2``.
  """

  m_min: float
  m_max: float
  m_grid_res: int = eqx.field(static=True, default=800)
  m_grid: jnp.ndarray
  h_shift: float
  norm_2d: float
  default = {'m_min': 3.0, 'm_max': 90.0}
  keys: List[str] = eqx.field(static=True)
  name = 'base_mass_joint_m1m2_struct'

  def __init__(self, **kwargs):
    # m_grid_res is static (controls jit-trace shape, not a sampled
    # hyperparameter) so, like mc_grid_res/q_grid_res in
    # joint/base.py's base_mass_joint_struct, it is read directly from
    # kwargs rather than through the `.default`/`.keys` dynamic-
    # hyperparameter machinery below.
    self.m_grid_res = kwargs.get('m_grid_res', 800)

    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    self.m_grid = jnp.logspace(jnp.log10(self.m_min), jnp.log10(self.m_max), self.m_grid_res)
    self.h_shift, self.norm_2d = _compute_shift_and_norm_2d(self)

  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    # m_grid_res isn't in `.keys` (see __init__), so it must be carried
    # forward explicitly -- otherwise every `.update()` call (i.e. every
    # sampler step) would silently reset a caller's resolution override
    # back to the class default.
    fiducials['m_grid_res'] = self.m_grid_res
    return self.__class__(**fiducials)


# ---------------------------------------------------------------------------
# Dispatch stub -- MUST be overridden by concrete subclasses
# ---------------------------------------------------------------------------

@dispatch
def _h_m1m2(mass: base_mass_joint_m1m2_struct, m1: jnp.ndarray, m2: jnp.ndarray):
  """Signed log-density kernel h(m1, m2). Concrete subclasses implement
  this via plum dispatch specialised on their type. Not itself masked or
  truncated -- ``mass_pdf_notnorm_m1m2`` below applies the
  ``[m_min,m_max]`` x ``m2<=m1`` mask."""
  raise NotImplementedError(
    "_h_m1m2 not implemented for base_mass_joint_m1m2_struct."
  )


# ---------------------------------------------------------------------------
# Internal helper: log-sum-exp-style shift + 2-D normalisation
# ---------------------------------------------------------------------------

def _inside_box(mass: base_mass_joint_m1m2_struct, m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  """Support mask: m2 <= m1, both within [m_min, m_max]."""
  return (
    (m2 <= m1)
    & (m1 >= mass.m_min) & (m1 <= mass.m_max)
    & (m2 >= mass.m_min) & (m2 <= mass.m_max)
  )


def _compute_shift_and_norm_2d(mass: base_mass_joint_m1m2_struct):
  """h_shift = max_{m2<=m1} h(m1,m2) over m_grid x m_grid (a log-sum-exp-
  style stabiliser, analogous to the reference ``logBspline.logZ()``'s
  max-subtraction, kept compatible with this package's established
  "notnorm + norm_2d division" architecture so the shift cancels exactly
  in p = notnorm / Z); norm_2d = ∫∫_{m2<=m1} exp(h - h_shift) dm1 dm2.

  Axes convention: axis-0 -> m1, axis-1 -> m2 (matches ``joint/base.py``'s
  mc/q convention).
  """
  g = mass.m_grid
  m1_2d = g[:, None]
  m2_2d = g[None, :]
  h_grid = _h_m1m2(mass, m1_2d, m2_2d)              # (N, N)
  inside = _inside_box(mass, m1_2d, m2_2d)          # (N, N)
  h_masked = jnp.where(inside, h_grid, -jnp.inf)
  h_shift = jnp.max(h_masked)
  notnorm_grid = jnp.where(inside, jnp.exp(h_grid - h_shift), 0.0)
  inner = trapz(notnorm_grid, x=g, axis=1)          # ∫ dm2 for every m1 -> (N,)
  norm_2d = trapz(inner, x=g)                       # ∫ dm1 -> scalar
  return h_shift, norm_2d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm_m1m2(mass: base_mass_joint_m1m2_struct, m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  """Un-normalised joint density ``exp(h(m1,m2) - h_shift)``, hard-masked
  to zero where ``m2 > m1`` or outside ``[m_min, m_max]``."""
  h = _h_m1m2(mass, m1, m2)
  inside = _inside_box(mass, m1, m2)
  return jnp.where(inside, jnp.exp(h - mass.h_shift), 0.0)


@dispatch
def p_m1m2(mass: base_mass_joint_m1m2_struct, m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  r"""Joint PDF, already native in (m1, m2) -- no Jacobian needed (unlike
  the ``joint`` (Mc,q)-native family)."""
  notnorm = mass_pdf_notnorm_m1m2(mass, m1, m2)
  Z = jnp.maximum(mass.norm_2d, jnp.finfo(jnp.float_).tiny)
  return notnorm / Z


@dispatch
def p_m1m2(mass: base_mass_joint_m1m2_struct, theta: theta_src) -> jnp.ndarray:
  return p_m1m2(mass, theta.m1src, theta.m2src)


def pdf_joint_and_marg(mass: base_mass_joint_m1m2_struct, res=(5000, 2500)) -> dict:
  """Evaluate the joint PDF and its marginals on a dense (m1, m2) grid, for
  plotting/sanity checks -- same output keys and axes convention
  (``jnp.meshgrid`` default 'xy' indexing; ``p1_marg`` integrated over
  axis 0, ``p2_marg`` over axis 1) as the ``paired``/``joint`` versions of
  this function.

  Parameters
  ----------
  mass :
    Model instance.
  res :
    ``(n_m1, n_m2)`` resolution of the output grid.
  """
  m1 = jnp.linspace(mass.m_min, mass.m_max, res[0])
  m2 = jnp.linspace(mass.m_min, mass.m_max, res[1])
  m1mesh, m2mesh = jnp.meshgrid(m1, m2)
  p_joint = p_m1m2(mass, m1mesh, m2mesh)
  p1_marg = trapz(p_joint, x=m2, axis=0)
  p1_marg = p1_marg / jnp.maximum(trapz(p1_marg, x=m1), jnp.finfo(jnp.float_).tiny)
  p2_marg = trapz(p_joint, x=m1, axis=1)
  p2_marg = p2_marg / jnp.maximum(trapz(p2_marg, x=m2), jnp.finfo(jnp.float_).tiny)
  return {'m1': m1, 'm2': m2, 'm1mesh': m1mesh, 'm2mesh': m2mesh,
          'p_joint': p_joint, 'p_m1_marg': p1_marg, 'p_m2_marg': p2_marg}
