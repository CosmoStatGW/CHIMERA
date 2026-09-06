import jax.numpy as jnp
import jax
from typing import List
from plum import dispatch
import equinox as eqx
from .base import base_mass_paired_struct, mass_pdf_notnorm, pairing_function, _compute_norm_2d
from ..core import truncated_pl, high_pass_filter
from ..conditioned.pls import bspline_design_matrix
from ....utils.math import interp

def knot_logits_to_positions(knot_logits, x_low, x_high, spacing='log'):
    """Map unconstrained knot_logits to strictly increasing knot positions
    spanning [x_low, x_high]. Softmax outputs are strictly positive and sum
    to 1, so cumsum is strictly increasing -> knots can never swap order,
    regardless of knot_logits value.

    spacing='linear': cumsum in linear mass space (knot_logits=0 -> uniform
        spacing in solar masses).
    spacing='log':    cumsum in log-mass space (knot_logits=0 -> uniform
        spacing in log-mass, i.e. geometric/log spacing).
    """
    gaps = jax.nn.softmax(knot_logits)
    cum = jnp.cumsum(gaps)

    if spacing == 'linear':
        inner = x_low + cum * (x_high - x_low)
    elif spacing == 'log':
        log_low, log_high = jnp.log(x_low), jnp.log(x_high)
        inner = jnp.exp(log_low + cum * (log_high - log_low))
    else:
        raise ValueError("`spacing` can be only 'linear' or 'log'.")

    knot_list = jnp.concatenate((jnp.array([x_low]), inner))
    return knot_list


# ---------------------------------------------------------------------------
# Paired Power-Law + Spline (free knots) mass model
# ---------------------------------------------------------------------------

class pls_free_knots(base_mass_paired_struct):
  r"""Paired Power-Law + Spline (PLS) mass model, with inferred knot positions.

  Joint distribution:

    p(m1, m2) = p̃(m1) * p̃(m2) * f(m1, m2) / Z,   m2 <= m1

  with the *same* marginal shape for both masses:

    p̃(m) = PL(m) * exp(spline(m)) * S(m; m_low, δ_m)

  where:
    - PL(m) = m^{-α} / tpl_cdf(m_high; -α, m_low) is a truncated power law,
    - spline(m) is a B-spline correction whose knots span [min_knot, max_knot].
      Knot positions are inferred (via knot_logits) rather than fixed.
    - S(m; m_low, δ_m) is the LVK left-side smoothing window,

  and the pairing function is the usual power law in mass ratio:

    f(m1, m2) = (m2/m1)^β * θ(m1 - m2)

  Knot positions are parameterized via `knot_logits` (unconstrained,
  shape (num_inner_knots - 1,)) mapped through softmax + cumsum,
  guaranteeing strictly increasing knots (no swapping) for any value
  of `knot_logits`.

  Parameters
  ----------
  alpha : float
    Spectral index of the power-law component.
  beta : float
    Mass-ratio power-law index of the pairing function.
  delta_m : float
    Smoothing scale at the low-mass end.
  spline_coefficients : jnp.ndarray
    Coefficients of the B-spline correction, shape (num_coeffs,).
  min_knot: float
    Lower bound of knot's positions
  max_knot: float
    Upper bound of knot's positions
  knot_logits : jnp.ndarray
    Unconstrained parameters controlling inner knot spacing within
    [min_knot, max_knot], shape (num_inner_knots - 1,).
    knot_logits = 0 -> uniform spacing (linear or log, per spline_basis_spacing).
  num_inner_knots : int (static)
    Number of knots including the two endpoints (m_low, m_high).
  degree : int (static)
    B-spline degree (default 3, cubic).
  spline_basis_spacing : str (static)
    'linear' or 'log' -- controls both the knot-placement parameterization
    (via knot_logits_to_positions) and the basis-evaluation grid.
  spline_grid_res : int (static)
    Resolution of the spline evaluation grid.
  """
  alpha: float
  beta: float
  delta_m: float
  spline_coefficients: jnp.ndarray
  knot_logits: jnp.ndarray
  min_knot: float
  max_knot: float
  degree: int = eqx.field(static=True)
  num_inner_knots: int = eqx.field(static=True)
  num_coeffs: int = eqx.field(static=True)
  spline_basis_spacing: str = eqx.field(static=True)
  spline_grid_res: int = eqx.field(static=True)
  default = {
    **base_mass_paired_struct.default,
    'alpha':   3.6,
    'beta':    1.08,
    'delta_m': 3.3,
    'min_knot': 0.5,
    'max_knot': 100.0,
    'spline_coefficients': None,
    'knot_logits': None,
  }
  keys: List[str] = eqx.field(static=True)
  name = 'paired_powerlaw_plus_spline_free_knots'

  def __init__(self, **kwargs):
    self.degree = kwargs.get('degree', 3)
    self.num_inner_knots = kwargs['num_inner_knots']  # required
    assert self.num_inner_knots >= 2, "Need at least 2 inner knots (endpoints)."
    self.spline_basis_spacing = kwargs.get('spline_basis_spacing', 'log')
    self.spline_grid_res = kwargs.get('spline_grid_res', 1000)

    # n_basis = len(full_knots) - degree - 1, len(full_knots) = num_inner_knots + 2*degree
    self.num_coeffs = self.num_inner_knots + self.degree - 1

    self.default['spline_coefficients'] = jnp.zeros(self.num_coeffs)
    self.default['knot_logits'] = jnp.zeros(self.num_inner_knots - 1)  # -> uniform spacing

    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)

    # NB: spline_basis_grid is no longer precomputed once at fixed bounds --
    # it now depends on m_low/m_high, which may themselves be inferred, so
    # it's rebuilt inside compute_spline_basis() on every call, alongside
    # the knot positions and design matrix.

    # 2-D normalisation Z
    setattr(self, 'm_grid', jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res))
    setattr(self, 'norm_2d', _compute_norm_2d(self))

  def compute_spline_basis(self):
    """Rebuilds knots, the spline evaluation grid, and the B-spline design
    matrix. All three now depend on m_low/m_high (which may be inferred),
    in addition to knot_logits, so all are recomputed every call."""
    knot_list = knot_logits_to_positions(
        self.knot_logits, self.min_knot, self.max_knot,
        spacing=self.spline_basis_spacing,
    )
    knots = jnp.concatenate((jnp.full(self.degree, knot_list[0]),
                              knot_list,
                              jnp.full(self.degree, knot_list[-1])))

    if self.spline_basis_spacing == 'linear':
      spline_basis_grid = jnp.linspace(self.m_low, self.m_high, self.spline_grid_res)
    elif self.spline_basis_spacing == 'log':
      spline_basis_grid = jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.spline_grid_res)
    else:
      raise ValueError("`spline_basis_spacing` can be only `linear` or 'log'.")

    spline_basis = bspline_design_matrix(spline_basis_grid, knots, self.degree)
    return spline_basis_grid, spline_basis

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    fiducials['degree'] = self.degree
    fiducials['num_inner_knots'] = self.num_inner_knots
    fiducials['spline_basis_spacing'] = self.spline_basis_spacing
    fiducials['spline_grid_res'] = self.spline_grid_res
    return self.__class__(**fiducials)


# ---------------------------------------------------------------------------
# Marginal shape (same for m1 and m2)
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm(mass: pls_free_knots, m: jnp.ndarray) -> jnp.ndarray:
  """p(m) = PL(m) * exp(spline(m)) * S(m)"""
  P = truncated_pl(m, -mass.alpha, mass.m_low, mass.m_high) 
  spline_basis_grid, spline_basis = mass.compute_spline_basis()
  spline_vals_on_grid = jnp.dot(spline_basis, mass.spline_coefficients)
  spline_values = interp(m, spline_basis_grid, spline_vals_on_grid)
  pdf = P * jnp.exp(spline_values)
  pdf *= high_pass_filter(m, mass.delta_m, mass.m_low)
  return pdf


@dispatch
def pairing_function(mass: pls_free_knots,
                     m1: jnp.ndarray,
                     m2: jnp.ndarray) -> jnp.ndarray:
  q = m2 / m1
  return jnp.where(q <= 1.0, jnp.power(q, mass.beta), 0.0)
