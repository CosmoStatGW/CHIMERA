# nn.py

from typing import List

import jax
import jax.numpy as jnp

from plum import dispatch
import equinox as eqx

from .base import (
    base_mass_paired_struct,
    mass_pdf_notnorm,
    pairing_function,
    _compute_norm_2d,
)

# ===========================================================================
# Tanh-MLP log-density
# ===========================================================================
#
# A genuine (small) feedforward network -- every weight/bias is a free,
# jointly-sampled hyperparameter, unlike `random_features_density` (see
# `random_features.py`) where only `w_out`/`feature_scales` are free and
# the nonlinear basis itself is fixed random noise. This trades
# `random_features_density`'s "no pretraining, few free params" property
# for strictly more shape flexibility per parameter (a trained hidden
# layer, not a random one), at the cost of a higher-dimensional, more
# strongly coupled posterior (every weight interacts with every other
# weight through the tanh nonlinearity, whereas `w_out` enters
# `random_features_density` linearly).
#
# Activation: tanh, not cos. `random_features_density.log_density` uses
# `cos(W_hidden . x + b_hidden)` features, which is fine there because
# `feature_scales` keeps the frequencies low/controlled -- but a network
# whose *weights* are free to roam the full posterior has no such
# safeguard, and a periodic activation combined with large weights can
# produce a mass function with genuine periodic structure in log-mass
# (repeated peaks at regular spacing). That is a specific danger for
# cosmological (H0) inference from the mass function: the standard-siren-
# like trick this package relies on identifies H0 by matching a *fixed*,
# non-repeating source-frame mass feature (e.g. one peak) across events at
# different redshifts/detector-frame masses. A periodic mass function
# supplies extra copies of "a feature" at other mass values for free, so a
# wrong H0 can slide a redshifted copy of one peak onto a neighbouring
# repeat of another and still fit -- an extra approximate degeneracy that a
# smooth, non-periodic, saturating nonlinearity like tanh cannot create
# (tanh has no periodic component at any weight scale, so there is no
# spacing for a wrong-H0 fit to exploit).


def _mlp_init(input_size, hidden_sizes, key):
    """Draw an initial (Ws, bs) pair for a tanh-MLP with the given hidden
    layer widths, plus a final bias-free linear output layer (single
    output unit, the log-density itself)."""
    sizes = (input_size,) + tuple(hidden_sizes)
    depth = len(hidden_sizes)
    keys = jax.random.split(key, depth + 1)

    Ws, bs = [], []
    for i in range(depth):
        w_key, b_key = jax.random.split(keys[i])
        lim = 1.0 / jnp.sqrt(sizes[i])
        Ws.append(jax.random.uniform(w_key, (sizes[i + 1], sizes[i]), minval=-lim, maxval=lim))
        bs.append(jax.random.uniform(b_key, (sizes[i + 1],), minval=-lim, maxval=lim))

    # Bias-free output layer (see the class docstring for why).
    lim = 1.0 / jnp.sqrt(hidden_sizes[-1])
    Ws.append(jax.random.uniform(keys[-1], (1, hidden_sizes[-1]), minval=-lim, maxval=lim))

    return Ws, bs


def _mlp_log_density_single(Ws, bs, x):
    h = jnp.atleast_1d(x)
    for W, b in zip(Ws[:-1], bs):
        h = jnp.tanh(W @ h + b)
    return (Ws[-1] @ h)[0]


def mlp_log_density(mass, x):
    flat_g = jax.vmap(lambda xi: _mlp_log_density_single(mass.Ws, mass.bs, xi))(x.reshape(-1))
    return flat_g.reshape(x.shape)


# ===========================================================================
# Paired tanh-MLP density model
# ===========================================================================

class nn_density(base_mass_paired_struct):
    r"""Paired neural-network mass model.

    The joint distribution is

    .. math::

        p(m_1, m_2) =
        \frac{\tilde p(m_1)\,\tilde p(m_2)\,f(m_1,m_2)}{Z}

    where

    .. math::

        \tilde p(m)
        =
        \frac{\exp(g(\hat x))}{m},

    with

    .. math::

        \hat x =
        \frac{\log m-\bar x}{\sigma_x}.

    The log-density ``g`` is a genuine feedforward network: ``len(hidden_sizes)``
    ``tanh``-activated hidden layers followed by a bias-free linear output
    unit, all of whose weights/biases (``Ws``/``bs``) are free hyperparameters
    inferred directly inside the hierarchical Bayesian fit, exactly like
    ``w_out``/``feature_scales`` are for ``random_features_density`` --
    there is no separate pretraining stage.

    See the module-level comment above ``_mlp_init`` for why ``tanh`` (not
    ``cos``, unlike ``random_features_density``) is used here: with every
    weight free to be sampled (rather than fixed random noise scaled by a
    handful of ``feature_scales``), a periodic activation risks a mass
    function with genuine periodic structure in log-mass, which would open
    up a spurious extra degeneracy with H0 in a cosmological fit. ``tanh``
    is smooth and strictly non-periodic at any weight scale, so it cannot
    introduce that degeneracy.

    -----------------------------------------------------------------------
    Free hyperparameters
    -----------------------------------------------------------------------

    ``Ws`` and ``bs`` (lists of weight matrices / bias vectors, one entry
    per hidden layer, plus one bias-free output weight matrix in ``Ws``)
    are all sampled parameters -- unlike ``random_features_density``, there
    is no fixed/never-sampled noise tensor here.

    -----------------------------------------------------------------------
    Standardization
    -----------------------------------------------------------------------

    The network input is

        x = (log(m) - x_mean) / x_std

    where ``x_mean`` and ``x_std`` are fixed constants computed from
    ``m_low = 0.4`` and ``m_high = 200`` at construction time (a fixed,
    wide standardization range -- not the model's own truncation edges
    ``mass.m_low`` / ``mass.m_high``, which are typically narrower).

    The ``1/m`` factor is the Jacobian associated with the transformation
    from log-mass to mass.

    -----------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------

    input_size : int
        Static; dimensionality of the network input (1 for log-mass only).
    hidden_sizes : tuple of int
        Static; width of each hidden layer, in order, e.g. ``(16, 16)``
        (the default) for a 2-hidden-layer, 16-unit-wide network. Flexible
        depth and width: ``(32,)`` gives a single 32-unit hidden layer,
        ``(16, 16, 16)`` gives three. Keep this modest -- every weight is a
        sampled dimension, so a 2x16 network already adds
        ``1*16+16 + 16*16+16 + 16*1 = 320`` free parameters (input_size=1),
        far more than ``random_features_density``'s ``n_features``.
    Ws, bs : list of jnp.ndarray, optional
        Dynamic (sampled); explicit network weights/biases. ``bs`` has one
        entry per hidden layer; ``Ws`` has one additional entry (the final
        bias-free output row). Drawn once at construction from ``key`` if
        not supplied explicitly (only used to seed the *default* value --
        the sampler is free to move all of ``Ws``/``bs`` afterwards).
    key : jax.random.PRNGKey, optional
        Constructor-only key used to draw the initial ``Ws``/``bs`` when
        not supplied explicitly. Irrelevant once they are supplied (e.g.
        when reconstructing a model instance via ``update``).

    beta : float
        Mass-ratio power-law index.

    Notes
    -----
    ``p̃(m)`` is truncated to exactly zero outside ``[m_low, m_high]`` by a
    hard mask, with no separate edge-smoothing scale -- same reasoning as
    ``random_features_density``: ``g`` is already smooth by construction,
    so there is no sharp analytic edge feature to soften.

    No bias/intercept term is added to the output layer: an overall
    additive shift in the log-density is exactly degenerate with the
    model's own 2-D renormalisation (it exponentiates to a constant factor
    that cancels between ``p̃(m1)*p̃(m2)`` and ``Z``), so it would only add
    an unidentifiable direction to the posterior.
    """

    input_size: int = eqx.field(static=True)
    hidden_sizes: tuple = eqx.field(static=True)
    Ws: List[jnp.ndarray]
    bs: List[jnp.ndarray]
    beta: float
    x_mean: float = eqx.field(static=True)
    x_std: float = eqx.field(static=True)

    default = {
        **base_mass_paired_struct.default,
        "beta": 1.08,
        "input_size": 1,
        "hidden_sizes": (16, 16),
        "Ws": None,
        "bs": None,
    }

    name = "paired_nn_density"

    def __init__(self, key=None, **kwargs):
        self.keys = list(self.default.keys())

        for k in self.keys:
            setattr(self, k, kwargs.get(k, self.default[k]))
        self.hidden_sizes = tuple(self.hidden_sizes)

        log_m_low = jnp.log(0.4)
        log_m_high = jnp.log(200.)

        self.x_mean = 0.5 * (log_m_low + log_m_high)
        self.x_std = 0.5 * (log_m_high - log_m_low)

        if self.Ws is None or self.bs is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            self.Ws, self.bs = _mlp_init(self.input_size, self.hidden_sizes, init_key)

        self.m_grid = jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res)
        self.norm_2d = _compute_norm_2d(self)

# ===========================================================================
# Marginal density
# ===========================================================================

@dispatch
def mass_pdf_notnorm(mass: nn_density, m: jnp.ndarray):
    """Unnormalized marginal mass density.

    The MLP receives standardized log-mass,

        x = (log(m) - x_mean) / x_std.

    The permutation-free density is then

        p_tilde(m) = exp(g(x)) / m,

    hard-masked to zero outside [m_low, m_high]. No edge-smoothing scale is
    applied: g is already a smooth function of mass by construction (a
    tanh-MLP), so there is no sharp analytic feature at the truncation
    edges to soften.
    """
    # Standardized log-mass.
    x = (jnp.log(m) - mass.x_mean) / mass.x_std

    # tanh-MLP log-density.
    g = mlp_log_density(mass, x)

    # pdf, hard-truncated to [m_low, m_high]
    pdf = jnp.exp(g) / m
    pdf = jnp.where((m >= mass.m_low) & (m <= mass.m_high), pdf, 0.0)

    return pdf


# ===========================================================================
# Pairing
# ===========================================================================

@dispatch
def pairing_function(
    mass: nn_density,
    m1: jnp.ndarray,
    m2: jnp.ndarray,
):
    r"""Mass-ratio pairing function.

    .. math::

        f(m_1,m_2)
        =
        \left(\frac{m_2}{m_1}\right)^\beta

        \qquad\text{for }m_2\leq m_1,

    and zero otherwise.

    ``beta > 0`` favors equal-mass binaries,
    ``beta < 0`` favors unequal-mass binaries,
    and ``beta = 0`` gives a flat mass-ratio distribution.
    """
    q = m2 / m1

    return jnp.where(
        q <= 1.0,
        jnp.power(q, mass.beta),
        0.0,
    )
