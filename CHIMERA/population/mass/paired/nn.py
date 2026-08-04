# neural_density.py
from typing import List

import jax
import jax.numpy as jnp

from plum import dispatch
import equinox as eqx

from .base import base_mass_paired_struct, mass_pdf_notnorm, pairing_function, _compute_norm_2d
from ..core import high_pass_filter, low_pass_filter

# ---------------------------------------------------------------------------
# Neural log-density MLP: weight initialization
# ---------------------------------------------------------------------------

def init_mlp_params(hidden_size, depth, input_size, key):
  """Randomly initialize ``(Ws, bs)`` for a log-density MLP of the given
  shape.

  Uses the same fan-in uniform scheme as ``eqx.nn.Linear``'s default
  init.

  Returns
  -------
  Ws, bs : list[jnp.ndarray], list[jnp.ndarray]
      ``len(Ws) == depth + 1`` (last entry is the bias-free output
      layer), ``len(bs) == depth``.
  """
  sizes = [input_size] + [hidden_size] * depth
  keys = jax.random.split(key, depth + 1)

  Ws, bs = [], []
  for i in range(depth):
    w_key, b_key = jax.random.split(keys[i])
    lim = 1.0 / jnp.sqrt(sizes[i])
    Ws.append(jax.random.uniform(w_key, (sizes[i + 1], sizes[i]), minval=-lim, maxval=lim))
    bs.append(jax.random.uniform(b_key, (sizes[i + 1],), minval=-lim, maxval=lim))

  lim = 1.0 / jnp.sqrt(hidden_size)
  Ws.append(jax.random.uniform(keys[-1], (1, hidden_size), minval=-lim, maxval=lim))

  return Ws, bs

# ---------------------------------------------------------------------------
# Neural log-density forward pass
# ---------------------------------------------------------------------------
#
#     Linear -> Softplus -> ... -> Linear -> Softplus -> Linear(no bias)
#
# Softplus throughout keeps g(x) C^infty in both x and the weights, which
# matters for gradient-based samplers such as NUTS. The output layer has
# no bias because g(x) -> g(x) + c is exactly degenerate with the CHIMERA
# normalization constant, so a learnable output bias is unidentifiable.

def _log_density_single(Ws, bs, x):
  """Evaluate g(x) for one scalar log-mass x."""
  h = jnp.atleast_1d(x)
  for W, b in zip(Ws[:-1], bs):
    h = jax.nn.softplus(W @ h + b)
  return (Ws[-1] @ h)[0]

def log_density(Ws, bs, x):
  """Vectorized g(x) = NN(x); x may be a scalar or an array of *any*
  shape (e.g. the 2D m1/m2 meshgrids used by `p_m1m2`/`pdf_joint_and_marg`).
  """
  x = jnp.asarray(x)
  flat_g = jax.vmap(lambda xi: _log_density_single(Ws, bs, xi))(x.reshape(-1))
  return flat_g.reshape(x.shape)

# ---------------------------------------------------------------------------
# Paired neural density model
# ---------------------------------------------------------------------------

class neural_density(base_mass_paired_struct):
  r"""Paired neural-density mass model.

  The joint distribution is

  .. math::

      p(m_1, m_2) =
      \frac{\tilde p(m_1)\, \tilde p(m_2)\, f(m_1, m_2)}{Z}

  where the marginal shape is described by a neural log-density,

  .. math::

      \tilde p(m) = \frac{\exp(g(\log m))}{m}\, S(m), \qquad
      g(\log m) = \mathrm{NN}(\log m),

  and :math:`S(m)` is the smoothing window applied at the mass edges
  (see :func:`mass_pdf_notnorm` below). The density is not normalized
  internally; the CHIMERA paired-model machinery computes the full 2D
  normalization :math:`Z` via ``_compute_norm_2d``.

  Design
  ------
  The network's *shape* and its *weights* are deliberately kept separate:

  - ``hidden_size``/``depth`` are static fields -- plain shape metadata,
    part of the pytree's structure (treedef), not differentiable leaves.
  - ``Ws``/``bs`` are the true hyperparameters: ordinary top-level pytree
    leaves on ``neural_density`` itself (not nested in a submodule), on
    equal footing with ``beta``, sampled directly by NUTS.

  Because building ``Ws``/``bs`` needs an RNG key, this class *does*
  overload ``__init__`` -- there's no way around touching RNG somewhere
  before the base class's normalization integral can run. The override
  copies the base class's assignment loop and grid/norm setup verbatim
  and inserts exactly one extra step: draw ``Ws``/``bs`` from
  ``hidden_size``/``depth`` if (and only if) they were not already
  supplied. In particular, ``base_mass_paired_struct.update(...)``
  always passes the model's *current* ``Ws``/``bs`` forward, so updating
  e.g. ``beta`` never resets already-trained weights.

  Parameters
  ----------
  hidden_size, depth : int
      Static network shape.
  Ws, bs : list[jnp.ndarray], optional
      Explicit weights/biases. If omitted, they are drawn from
      ``hidden_size``/``depth`` using ``key`` (or a fixed seed, if
      ``key`` is also omitted).
  key : jax.random.PRNGKey, optional
      Constructor-only argument (not stored as a field). Only used to
      initialize ``Ws``/``bs`` when they aren't provided directly.
  beta : float
      Mass-ratio power-law index (see :func:`pairing_function`).
  bottomsmooth, topsmooth : float
      Smoothing scales at the low- and high-mass edges.
  """
  input_size: int = eqx.field(static=True)
  hidden_size: int = eqx.field(static=True)
  depth: int = eqx.field(static=True)

  Ws: List[jnp.ndarray]
  bs: List[jnp.ndarray]
  beta: float
  bottomsmooth: float
  topsmooth: float

  default = {
    **base_mass_paired_struct.default,
    "beta": 1.08,
    "bottomsmooth": 3.3,
    "topsmooth": 3.3,
    "input_size": 1,
    "hidden_size": 4,
    "depth": 2,
    "Ws": None,
    "bs": None,
  }

  name = "paired_neural_density"

  def __init__(self, key=None, **kwargs):
    # --- copied verbatim from base_mass_paired_struct.__init__ ---
    self.keys = list(self.default.keys())
    for k in self.keys:
      setattr(self, k, kwargs.get(k, self.default[k]))

    if self.Ws is None or self.bs is None:
      self.Ws, self.bs = init_mlp_params(
        self.hidden_size, self.depth, self.input_size,
        key if key is not None else jax.random.PRNGKey(0),
      )

    self.m_grid = jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res)
    self.norm_2d = _compute_norm_2d(self)

# ---------------------------------------------------------------------------
# Marginal density
# ---------------------------------------------------------------------------

@dispatch
def mass_pdf_notnorm(mass: neural_density, m: jnp.ndarray):
  """Unnormalized marginal mass density.

  Returns
  -------
  jnp.ndarray
      :math:`\\tilde p(m) = \\exp(g(\\log m)) / m \\cdot S(m)`, zeroed
      outside ``[m_low, m_high]``.
  """
  g = log_density(mass.Ws, mass.bs, jnp.log(m))
  pdf = jnp.exp(g) / m
  pdf *= high_pass_filter(m, mass.bottomsmooth, mass.m_low)
  pdf *= low_pass_filter(m, mass.topsmooth, mass.m_high)
  return jnp.where((m >= mass.m_low) & (m <= mass.m_high), pdf, 0.0)

# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dispatch
def pairing_function(mass: neural_density, m1: jnp.ndarray, m2: jnp.ndarray):
  r"""Mass-ratio pairing function.

  .. math::

      f(m_1, m_2) = (m_2/m_1)^\beta \quad \text{for } m_2 \le m_1,
      \text{ else } 0.

  :math:`\beta > 0` favors equal-mass binaries, :math:`\beta < 0` favors
  unequal-mass binaries, and :math:`\beta = 0` gives a flat mass-ratio
  distribution.
  """
  q = m2 / m1
  return jnp.where(q <= 1.0, jnp.power(q, mass.beta), 0.0)
