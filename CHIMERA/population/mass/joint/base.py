import jax.numpy as jnp
import equinox as eqx
from typing import List
from plum import dispatch

from ....utils.math import trapz
from ....data import theta_src

################
# MASS PYTREES #
################

class base_mass_joint_struct(eqx.Module):
  r"""Base class for a genuinely joint 2-D mass model, parametrized
  directly in (chirp mass, mass ratio) rather than in (m1, m2).

  Unlike ``paired`` (``p(m1,m2) = p̃(m1) p̃(m2) f(m1,m2) / Z``) or
  ``conditioned`` (``p(m1,m2) = p(m1) p(m2|m1)``), this family places a
  single joint density directly on

  .. math::

      M_c = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}}
      \qquad\text{(chirp mass)}, \qquad
      q = \frac{m_2}{m_1} \in (0, 1]
      \qquad\text{(mass ratio, }m_2 \leq m_1\text{)},

  and pushes it forward to (m1, m2) via the Jacobian of the
  ``(m1,m2) -> (Mc,q)`` change of variables:

  .. math::

      p(m_1, m_2) = p(M_c, q)\,\left|\frac{\partial(M_c,q)}{\partial(m_1,m_2)}\right|
      = p(M_c, q)\,\frac{M_c}{m_1^2}

  (derived in ``jacobian_mc_q_from_m1m2`` below).

  **Why (Mc, q) and not (m1, m2) directly.** The whole point of using
  (Mc, q) as the *native* coordinates for the basis expansion is that the
  physical domain becomes a plain rectangle
  ``[mc_low, mc_high] x [q_low, 1]`` with **no ordering constraint**:
  every ``(Mc, q)`` pair in that rectangle maps to exactly one valid
  ``m2 <= m1`` pair via ``m1_of_mc_q``/``m2_of_mc_q`` below. ``paired``
  and ``conditioned`` models both have to build the ``m2 <= m1`` boundary
  into the model itself (a hard mask plus, for ``paired`` models, a
  pairing function that must vanish for ``q > 1``); here there is nothing
  to mask *inside* the (Mc, q) box, only the box's own edges (an ordinary
  truncation, same spirit as ``m_low``/``m_high`` elsewhere in this
  package).

  Concrete subclasses must override ``mass_pdf_notnorm_mcq`` -- the
  *unnormalised* joint density on ``(Mc, q)`` -- via ``plum`` dispatch.
  ``p_m1m2`` (registered on the same global dispatcher as the
  ``paired``/``conditioned`` families -- see ``CHIMERA.population.mass``)
  and ``pdf_joint_and_marg`` are implemented once here, generically, in
  terms of ``mass_pdf_notnorm_mcq`` and the change-of-variables helpers.

  Attributes
  ----------
  mc_low, mc_high : float
    Chirp-mass truncation range.
  q_low : float
    Lower bound on the mass ratio. Kept strictly positive to avoid the
    ``m2 -> 0`` degeneracy (the Jacobian ``Mc/m1^2`` itself stays finite
    as ``q -> 0``, but ``m1_of_mc_q`` blows up there, and physically
    ``q = 0`` is a measure-zero, unobservable edge). ``q_high`` is fixed
    at exactly ``1.0`` and is *not* a free hyperparameter: ``q > 1`` is
    excluded by the ``m2 <= m1`` convention, not by a truncation choice.
  mc_grid_res, q_grid_res : int
    Grid resolution per axis for the 2-D normalisation integral.
  mc_grid, q_grid : jnp.ndarray
    1-D grids (log-spaced in ``Mc``, linear in ``q``) spanning the
    truncation box, used both for the normalisation integral and as the
    default resolution for ``p_m1`` and ``pdf_joint_and_marg``.
  norm_2d : float
    Precomputed ``Z = ∫∫ p̃(Mc,q) dMc dq``.
  """

  mc_low: float
  mc_high: float
  q_low: float
  mc_grid_res: int = eqx.field(static=True, default=800)
  q_grid_res: int = eqx.field(static=True, default=800)
  mc_grid: jnp.ndarray
  q_grid: jnp.ndarray
  norm_2d: float
  default = {'mc_low': 3.0, 'mc_high': 90.0, 'q_low': 0.05}
  keys: List[str] = eqx.field(static=True)
  name = 'base_mass_joint_struct'

  def __init__(self, **kwargs):
    # mc_grid_res/q_grid_res are static (control jit-trace shape, not a sampled
    # hyperparameter) so, like `degree`/`num_inner_knots` in pls_free_knots.py, they
    # are read directly from kwargs rather than through the `.default`/`.keys` dynamic-
    # hyperparameter machinery below -- otherwise a caller-supplied override would be
    # silently ignored (every kwarg not in `.keys` is dropped by the loop below).
    self.mc_grid_res = kwargs.get('mc_grid_res', 800)
    self.q_grid_res = kwargs.get('q_grid_res', 800)

    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)
    self.mc_grid = jnp.logspace(jnp.log10(self.mc_low), jnp.log10(self.mc_high), self.mc_grid_res)
    self.q_grid = jnp.linspace(self.q_low, 1.0, self.q_grid_res)
    self.norm_2d = _compute_norm_2d_mcq(self)

  @property
  def as_dict(self):
    return {k: getattr(self, k) for k in self.keys}

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    # mc_grid_res/q_grid_res aren't in `.keys` (see __init__), so they must be carried
    # forward explicitly -- otherwise every `.update()` call (i.e. every sampler step)
    # would silently reset a caller's resolution override back to the class default.
    fiducials['mc_grid_res'] = self.mc_grid_res
    fiducials['q_grid_res'] = self.q_grid_res
    return self.__class__(**fiducials)


# ---------------------------------------------------------------------------
# Change of variables: (m1, m2) <-> (Mc, q)
# ---------------------------------------------------------------------------

def mc_of_m1m2(m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  r"""Chirp mass :math:`M_c = (m_1 m_2)^{3/5} / (m_1+m_2)^{1/5}`."""
  return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def q_of_m1m2(m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  """Mass ratio q = m2/m1 (<= 1 when m2 <= m1)."""
  return m2 / m1


def m1_of_mc_q(mc: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
  r"""Inverse map: :math:`m_1 = M_c\,(1+q)^{1/5}\,q^{-3/5}`."""
  return mc * (1.0 + q) ** 0.2 / q ** 0.6


def m2_of_mc_q(mc: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
  """Inverse map: m2 = q * m1(Mc, q)."""
  return q * m1_of_mc_q(mc, q)


def jacobian_mc_q_from_m1m2(m1: jnp.ndarray, mc: jnp.ndarray) -> jnp.ndarray:
  r"""``|d(Mc, q)/d(m1, m2)|``, i.e. the Jacobian needed to push a density
  forward from ``(Mc, q)`` to ``(m1, m2)``: ``p(m1,m2) = p(Mc,q) * this``.

  Derivation. Writing :math:`\ln M_c = \tfrac{3}{5}\ln m_1 + \tfrac{3}{5}\ln
  m_2 - \tfrac{1}{5}\ln(m_1+m_2)` and :math:`q = m_2/m_1`:

  .. math::

      \frac{\partial M_c}{\partial m_1} = M_c\left(\frac{3}{5 m_1} -
      \frac{1}{5(m_1+m_2)}\right), \quad
      \frac{\partial M_c}{\partial m_2} = M_c\left(\frac{3}{5 m_2} -
      \frac{1}{5(m_1+m_2)}\right),

      \frac{\partial q}{\partial m_1} = -\frac{q}{m_1}, \quad
      \frac{\partial q}{\partial m_2} = \frac{1}{m_1}.

  Plugging into the 2x2 Jacobian determinant and using ``q = m2/m1`` (so
  ``q/m2 = 1/m1`` and ``1+q = (m1+m2)/m1``) makes every ``m2``-dependent
  term cancel exactly, leaving

  .. math::

      \left|\frac{\partial(M_c,q)}{\partial(m_1,m_2)}\right| = \frac{M_c}{m_1^2}.

  Equivalently, ``dm1 dm2 = (m1^2/Mc) dMc dq`` -- the standard relation
  used throughout the GW population literature to reweight samples
  between ``(m1,m2)`` and ``(Mc,q)``.
  """
  return mc / m1 ** 2


# ---------------------------------------------------------------------------
# Dispatch stub -- MUST be overridden by concrete subclasses
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm_mcq(mass: base_mass_joint_struct, mc: jnp.ndarray, q: jnp.ndarray):
  """Un-normalised joint density p̃(Mc, q). Concrete subclasses implement
  this via plum dispatch specialised on their type, hard-truncated to
  zero outside [mc_low, mc_high] x [q_low, 1]."""
  raise NotImplementedError(
    "mass_pdf_notnorm_mcq not implemented for base_mass_joint_struct."
  )


# ---------------------------------------------------------------------------
# Internal helper: 2-D normalisation
# ---------------------------------------------------------------------------

def _compute_norm_2d_mcq(mass: base_mass_joint_struct) -> float:
  """Z = ∫∫ p̃(Mc,q) dMc dq over [mc_low,mc_high] x [q_low,1].

  Axes convention: axis-0 -> Mc, axis-1 -> q.
  """
  mc_g, q_g = mass.mc_grid, mass.q_grid
  integrand = mass_pdf_notnorm_mcq(mass, mc_g[:, None], q_g[None, :])  # (n_mc, n_q)
  inner = trapz(integrand, x=q_g, axis=1)   # ∫ dq for every Mc -> (n_mc,)
  return trapz(inner, x=mc_g)               # ∫ dMc -> scalar


def _m1m2_plot_bounds(mass: base_mass_joint_struct):
  """Physical (m1, m2) bounding box implied by the (Mc,q) support box.

  m1(Mc,q) is monotonically decreasing in q (for fixed Mc) and increasing
  in Mc (for fixed q); m2(Mc,q) = q*m1 is monotonically increasing in both.
  """
  m1_lo = m1_of_mc_q(mass.mc_low, 1.0)          # smallest m1: min Mc, q=1
  m1_hi = m1_of_mc_q(mass.mc_high, mass.q_low)  # largest m1: max Mc, min q
  m2_lo = m2_of_mc_q(mass.mc_low, mass.q_low)   # smallest m2: min Mc, min q
  m2_hi = m2_of_mc_q(mass.mc_high, 1.0)         # largest m2: max Mc, q=1
  return m1_lo, m1_hi, m2_lo, m2_hi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dispatch
def p_m1m2(mass: base_mass_joint_struct, m1: jnp.ndarray, m2: jnp.ndarray) -> jnp.ndarray:
  r"""Joint PDF in (m1, m2), obtained by pushing the (Mc,q) density forward:

  .. math::

      p(m_1, m_2) = \frac{\tilde p(M_c, q)}{Z}\,\frac{M_c}{m_1^2}.

  Exactly zero where ``m2 > m1`` (``q > 1``), or outside the
  ``[mc_low,mc_high] x [q_low, 1]`` support box -- both enforced by
  ``mass_pdf_notnorm_mcq`` returning 0 there (not by an extra mask here),
  except for the ``q > 1`` half of the (m1,m2) plane, which is masked
  explicitly since ``mass_pdf_notnorm_mcq`` is never even told which of
  its two inputs came from which physical mass.
  """
  mc = mc_of_m1m2(m1, m2)
  q = q_of_m1m2(m1, m2)
  p_tilde = mass_pdf_notnorm_mcq(mass, mc, q)
  jac = jacobian_mc_q_from_m1m2(m1, mc)
  Z = jnp.maximum(mass.norm_2d, jnp.finfo(jnp.float_).tiny)
  return jnp.where(q <= 1.0, p_tilde * jac / Z, 0.0)


@dispatch
def p_m1m2(mass: base_mass_joint_struct, theta: theta_src) -> jnp.ndarray:
  return p_m1m2(mass, theta.m1src, theta.m2src)


def pdf_joint_and_marg(mass: base_mass_joint_struct, res=(5000, 2500)) -> dict:
  """Evaluate the joint PDF and its marginals on a dense (m1, m2) grid, for
  plotting/sanity checks -- same output keys as the paired/conditioned
  versions of this function.

  Parameters
  ----------
  mass :
    Model instance.
  res :
    ``(n_m1, n_m2)`` resolution of the output grid.
  """
  m1_lo, m1_hi, m2_lo, m2_hi = _m1m2_plot_bounds(mass)
  m1 = jnp.linspace(m1_lo, m1_hi, res[0])
  m2 = jnp.linspace(m2_lo, m2_hi, res[1])
  m1mesh, m2mesh = jnp.meshgrid(m1, m2)
  p_joint = p_m1m2(mass, m1mesh, m2mesh)
  p1_marg = trapz(p_joint, x=m2, axis=0)
  p1_marg = p1_marg / jnp.maximum(trapz(p1_marg, x=m1), jnp.finfo(jnp.float_).tiny)
  p2_marg = trapz(p_joint, x=m1, axis=1)
  p2_marg = p2_marg / jnp.maximum(trapz(p2_marg, x=m2), jnp.finfo(jnp.float_).tiny)
  return {'m1': m1, 'm2': m2, 'm1mesh': m1mesh, 'm2mesh': m2mesh,
          'p_joint': p_joint, 'p_m1_marg': p1_marg, 'p_m2_marg': p2_marg}
