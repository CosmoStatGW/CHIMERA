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
#
# ---------------------------------------------------------------------------
# Whitened (non-centered) parameterization
# ---------------------------------------------------------------------------
#
# `Ws`/`bs` are *not* the physical weights/biases themselves: they are
# standardized ("whitened") i.i.d. N(0, 1) variables ("z"), and the actual,
# physical weight/bias used in the forward pass is `scale * z`, where
# `scale` is a fixed (static, never sampled) per-layer constant derived at
# construction time from `hidden_prior_mult`/`target_std_g`/
# `bias_prior_scale` -- see `hidden_scales`/`out_scale` below, which
# reproduce the fan-in (Xavier/LeCun-style) calibration that used to be
# hand-rolled per-run in `nn_density_run_tempest.ipynb`.
#
# Why: this makes the *sampled* prior over every single weight/bias element
# an exactly isotropic standard Normal, regardless of that element's
# fan-in. A sampler that works best with (close to) isotropic targets --
# e.g. NUTS's default metric, or mini-tempest's pCN/tpCN moves -- mixes
# noticeably better over a whitened parameterization than over the
# fan-in-rescaled-but-still-anisotropic Gaussian prior this class used to
# place directly on the physical weights. The physical model (the function
# class this network can represent, and its prior over functions) is
# unchanged -- only the coordinates the sampler actually moves in are.
#
# ---------------------------------------------------------------------------
# Symmetry breaking (`break_symmetry`)
# ---------------------------------------------------------------------------
#
# A tanh-MLP posterior has an exact combinatorial redundancy: permuting the
# units of a hidden layer (together with their weights) leaves the network
# function unchanged, and -- because tanh is an odd function -- so does
# jointly flipping the sign of a unit's incoming row, its bias, and its
# outgoing weight. For two 16-unit hidden layers that is up to
# `16! * 2**16` per layer of exactly-degenerate copies of the same function
# scattered around parameter space, which forces any sampler (SMC mutation,
# NUTS, pCN, ...) to waste effort hopping between equivalent copies of one
# mode instead of concentrating on it.
#
# `break_symmetry=True` (the default) removes this redundancy without
# reducing the function class at all: each hidden layer's bias is
# reparameterized as a strictly increasing, strictly positive sequence,
#
#     b = cumsum(softplus(bias_prior_scale * z_b)),
#
# instead of a free, unordered real vector. Since both the permutation and
# the sign-flip symmetry always act on a unit's bias (reordering it, or
# flipping its sign along with the rest of that unit), requiring the bias
# vector to be strictly increasing and strictly positive picks out exactly
# one representative per equivalence class -- every representable function
# is still reachable, just through a unique parameterization. Set
# `break_symmetry=False` to recover the old, unconstrained (but redundant)
# parameterization, e.g. for a direct A/B comparison.


def _mlp_init(input_size, hidden_sizes, key):
    """Draw standard-normal "z" variables -- the whitened, unit-scale
    weights/biases this model actually stores -- for a tanh-MLP with the
    given hidden layer widths, plus a bias-free final linear output layer
    (single output unit, the log-density itself).

    All fan-in/fan-out scaling (and, when `break_symmetry` is on, the
    ordering/positivity transform of the biases) is applied separately, at
    call time, from the model's static `hidden_scales`/`out_scale` -- see
    the module-level "Whitened (non-centered) parameterization" note above.
    This is why initialization here is a plain N(0, 1) draw rather than the
    fan-in-scaled uniform init a non-Bayesian MLP would use.
    """
    sizes = (input_size,) + tuple(hidden_sizes)
    depth = len(hidden_sizes)
    keys = jax.random.split(key, 2 * depth + 1)

    Ws, bs = [], []
    for i in range(depth):
        Ws.append(jax.random.normal(keys[2 * i], (sizes[i + 1], sizes[i])))
        bs.append(jax.random.normal(keys[2 * i + 1], (sizes[i + 1],)))

    # Bias-free output layer (see the class docstring for why).
    Ws.append(jax.random.normal(keys[-1], (1, hidden_sizes[-1])))

    return Ws, bs


def _ordered_positive_bias(z_b, bias_scale):
    """Map a standard-normal "z" vector to a strictly increasing, strictly
    positive bias vector -- the symmetry-breaking reparameterization (see
    module-level note above). `cumsum . softplus` is monotonic and always
    positive, so the result is strictly increasing and strictly positive
    for any real `z_b`, with no truncation/rejection needed."""
    return jnp.cumsum(jax.nn.softplus(bias_scale * z_b))


def _mlp_log_density_single(Ws_z, bs_z, hidden_scales, bias_scale, out_scale, break_symmetry, x):
    h = jnp.atleast_1d(x)
    for W_z, b_z, w_scale in zip(Ws_z[:-1], bs_z, hidden_scales):
        W = w_scale * W_z
        if break_symmetry:
            b = _ordered_positive_bias(b_z, bias_scale)
        else:
            b = bias_scale * b_z
        h = jnp.tanh(W @ h + b)
    W_out = out_scale * Ws_z[-1]
    return (W_out @ h)[0]


def mlp_log_density(mass, x):
    flat_g = jax.vmap(
        lambda xi: _mlp_log_density_single(
            mass.Ws, mass.bs, mass.hidden_scales, mass.bias_prior_scale,
            mass.out_scale, mass.break_symmetry, xi,
        )
    )(x.reshape(-1))
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
    unit, all of whose weights/biases are free hyperparameters inferred
    directly inside the hierarchical Bayesian fit, exactly like
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

    Unlike the original version of this class, ``Ws``/``bs`` are **not**
    the physical weights/biases themselves: they are whitened, i.i.d.
    N(0, 1) variables, and the physical weight/bias used in the forward
    pass is a fixed (static) per-layer rescaling of them -- see
    "Whitened (non-centered) parameterization" in the module-level comment
    above, and ``hidden_scales``/``out_scale`` below. Every element of
    ``Ws``/``bs`` therefore has the *same*, simple N(0, 1) prior, regardless
    of its layer's fan-in -- there is nothing left to calibrate per-layer
    in the prior specification itself.

    -----------------------------------------------------------------------
    Symmetry breaking
    -----------------------------------------------------------------------

    See "Symmetry breaking (``break_symmetry``)" in the module-level
    comment above. In short: ``break_symmetry=True`` (the default)
    reparameterizes each hidden layer's bias as a strictly increasing,
    strictly positive sequence, which removes the hidden-unit permutation
    and tanh sign-flip degeneracies (a combinatorial redundancy that
    otherwise forces any sampler to re-explore many copies of the same
    function) without changing the function class at all.

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
    break_symmetry : bool
        Static; whether to apply the ordering/positivity reparameterization
        of each hidden layer's bias (see "Symmetry breaking" above).
        Default ``True``. Set to ``False`` to recover the original,
        unconstrained (but permutation/sign-redundant) parameterization.
    hidden_prior_mult : float
        Static; Xavier/LeCun-style multiplier on ``1/sqrt(fan_in)`` used to
        set each hidden layer's weight scale (see ``hidden_scales``
        below). Default ``1.5``.
    target_std_g : float
        Static; desired standard deviation of ``g(x)`` itself (the
        network's log-density output) under the prior, used to set
        ``out_scale`` below. Default ``2.5``.
    bias_prior_scale : float
        Static; scale applied to each hidden layer's bias "z" before the
        (optional) ordering/positivity transform. Default ``1.0``.
    hidden_scales : tuple of float
        Static (derived, not a constructor argument); per-hidden-layer
        weight scale, ``hidden_prior_mult / sqrt(fan_in)`` -- the physical
        weight used in the forward pass is ``hidden_scales[i] * Ws[i]``.
    out_scale : float
        Static (derived, not a constructor argument); output-layer weight
        scale, ``target_std_g / sqrt(hidden_sizes[-1])`` -- the physical
        output weight is ``out_scale * Ws[-1]``.
    Ws, bs : list of jnp.ndarray, optional
        Dynamic (sampled); whitened N(0, 1) network weights/biases (see
        "Whitened (non-centered) parameterization" above -- these are
        *not* the physical weights/biases). ``bs`` has one entry per
        hidden layer; ``Ws`` has one additional entry (the final
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
    break_symmetry: bool = eqx.field(static=True)
    hidden_prior_mult: float = eqx.field(static=True)
    target_std_g: float = eqx.field(static=True)
    bias_prior_scale: float = eqx.field(static=True)
    hidden_scales: tuple = eqx.field(static=True)
    out_scale: float = eqx.field(static=True)
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
        "break_symmetry": True,
        "hidden_prior_mult": 1.5,
        "target_std_g": 2.5,
        "bias_prior_scale": 1.0,
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

        # Fan-in-calibrated, per-layer weight scale and output-layer scale
        # (see "Whitened (non-centered) parameterization" above) -- derived
        # once here from the static hyperparameters, never sampled.
        sizes = (self.input_size,) + self.hidden_sizes
        fan_ins = sizes[:-1]
        self.hidden_scales = tuple(
            float(self.hidden_prior_mult / jnp.sqrt(fi)) for fi in fan_ins
        )
        self.out_scale = float(self.target_std_g / jnp.sqrt(self.hidden_sizes[-1]))

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
