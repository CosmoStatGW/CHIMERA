# nn.py

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
# Random-features log-density
# ===========================================================================
#
# A fixed, randomly-initialized nonlinear basis (never trained/sampled) plus
# a single trainable linear output layer -- the "random features" / extreme
# learning-machine trick (Rahimi & Recht 2007-style random Fourier
# features). This gives NN-like shape flexibility while keeping only
# `n_features` free hyperparameters: no hidden-layer weights to fit, and
# critically, no pretraining/simulation-matching step needed at all -- the
# basis is just a fixed random nonlinear feature map, and `w_out` is
# sampled directly inside the same hierarchical Bayesian fit as every other
# hyperparameter here.
#
# The fixed random noise and the mixture lengthscales are deliberately kept
# SEPARATE: `raw_W_hidden` is drawn once as plain N(0, 1) noise (never
# touched again), and each block's `feature_scales[i]` is applied as an
# ordinary differentiable division at call time to get the effective
# W_hidden. This is what lets `feature_scales` be a genuine dynamic
# (traced) pytree leaf -- and therefore a live, jointly-sampled
# hyperparameter alongside `w_out` -- instead of a `static` field: a static
# field's *value* (not just shape/dtype) has to be identical across calls
# for JAX's jit cache to reuse a compiled trace, which is unworkable for a
# hyperparameter that a sampler proposes a new value for on every draw (see
# `random_features_density`'s docstring for the full reasoning).


def _block_feature_counts(n_features, n_scales):
    """Split n_features as evenly as possible across `n_scales` blocks (one
    per lengthscale), spreading any remainder over the first few blocks so
    the total is always exactly n_features."""
    base_count, remainder = divmod(n_features, n_scales)
    return [base_count + (1 if i < remainder else 0) for i in range(n_scales)]


def _block_index_array(n_features, n_scales):
    """Static int32 array of shape (n_features,): for each fixed random
    feature row, which `feature_scales` block it belongs to. Depends only
    on the static `n_features`/`n_scales`, never on the lengthscale
    *values*, so it never needs to change across `update()` calls."""
    counts = _block_feature_counts(n_features, n_scales)
    idx = [i for i, count in enumerate(counts) for _ in range(count)]
    return jnp.array(idx, dtype=jnp.int32)


def _effective_W_hidden(raw_W_hidden, feature_scales, block_index):
    """W_hidden_k = raw_W_hidden_k / feature_scales[block_index_k].

    `raw_W_hidden` is fixed i.i.d. N(0, 1) noise; dividing it by the
    lengthscale of its block is what makes row k behave as if it had been
    drawn from N(0, 1/feature_scales[block]**2) -- but, unlike drawing it
    that way directly, this division is an ordinary differentiable
    elementwise op, so gradients/vmap over `feature_scales` work exactly
    like they do for `w_out`.
    """
    scales_per_row = feature_scales[block_index]  # (n_features,)
    return raw_W_hidden / scales_per_row[:, None]


def _random_features(x, W_hidden, b_hidden):
    """phi_k(x) = cos(W_hidden_k . x + b_hidden_k), shape (n_features,)."""
    x = jnp.atleast_1d(x)
    return jnp.cos(W_hidden @ x + b_hidden)


def log_density(mass, x):
    W_hidden = _effective_W_hidden(mass.raw_W_hidden, mass.feature_scales, mass.block_index)

    def single(xi):
        phi = _random_features(xi, W_hidden, mass.b_hidden)
        return jnp.dot(mass.w_out, phi)

    flat_g = jax.vmap(single)(x.reshape(-1))
    return flat_g.reshape(x.shape)


# ===========================================================================
# Paired random-features density model
# ===========================================================================

class random_features_density(base_mass_paired_struct):
    r"""Paired random-features mass model.

    The joint distribution is

    .. math::

        p(m_1, m_2) =
        \frac{\tilde p(m_1)\,\tilde p(m_2)\,f(m_1,m_2)}{Z}

    where

    .. math::

        \tilde p(m)
        =
        \frac{\exp(g(\hat x))}{m}\,S(m),

    with

    .. math::

        \hat x =
        \frac{\log m-\bar x}{\sigma_x}.

    The log-density g is a *random-features* expansion:

    .. math::

        g(\hat x) = \sum_{k=1}^{K} w_{\mathrm{out},k}\,
        \cos\!\left(W_{\mathrm{hidden},k}\,\hat x + b_{\mathrm{hidden},k}\right),

    where ``W_hidden`` is built from a fixed standard-normal noise tensor
    ``raw_W_hidden`` (drawn once at construction and never touched again)
    rescaled block-by-block by ``feature_scales``:
    ``W_hidden[k] = raw_W_hidden[k] / feature_scales[block_of(k)]``. Both
    ``feature_scales`` and the linear output weights ``w_out`` are free
    hyperparameters, inferred directly inside the hierarchical Bayesian fit
    like every other hyperparameter here -- there is no separate
    pretraining stage and no dependence on assumed simulated mass-function
    shapes.

    -----------------------------------------------------------------------
    Free hyperparameters
    -----------------------------------------------------------------------

    ``w_out`` (shape ``(n_features,)``) and ``feature_scales`` (shape
    ``(len(feature_scales),)``, 4 by default) are the sampled parameters.
    ``raw_W_hidden`` and ``b_hidden`` are fixed at construction from
    ``key`` and never touched again (no prior is ever placed on them, no
    sampler ever proposes a new value for them).

    Unlike ``raw_W_hidden``/``b_hidden`` (fixed noise, kept as ordinary
    non-``static`` pytree leaves so jit only needs to compare their
    shape/dtype -- see the note below), ``feature_scales`` is deliberately
    **not** ``eqx.field(static=True)`` either, for a stronger reason: a
    ``static`` field's *value* must be hashable and identical across calls
    for JAX to reuse a compiled trace, which is fundamentally incompatible
    with a sampler proposing a new continuous value for it on every draw
    (as opposed to being fixed once like the noise). Making
    ``feature_scales`` an ordinary dynamic leaf, and applying it to the
    fixed noise via a plain elementwise division at call time (see
    ``_effective_W_hidden``), is what makes it safe to sample: the jit
    trace only depends on ``feature_scales``'s shape (always 4), and
    ``jax.grad``/``vmap`` over its value works exactly like they already do
    for ``w_out``.

    ``raw_W_hidden``/``b_hidden`` themselves are ordinary (non-``static``)
    pytree leaves, not ``eqx.field(static=True)``: JAX's jit-cache needs to
    compare a traced function's *static* metadata via plain ``==`` to
    decide whether to reuse a compiled trace, and comparing two
    different-valued multi-element arrays that way raises exactly the
    "truth value of an array... is ambiguous" error the first time the
    same jitted function sees two model instances whose
    ``raw_W_hidden``/``b_hidden`` values differ (e.g. constructed from
    different ``key``s). Keeping them as regular leaves sidesteps this
    entirely: jit only needs their *shape/dtype* to decide on a retrace,
    the same way it already treats ``w_out``, and ``base_mass_paired_struct``
    already treats its own ``m_grid``/``norm_2d``.

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
    n_features : int
        Static; total number of fixed random features / free ``w_out``
        entries, split as evenly as possible across ``feature_scales``
        (below). Keep this modest (tens, not hundreds) to keep the sampled
        dimensionality low -- see the package README's guidance on
        gradient-based samplers' dimension scaling.
    feature_scales : array-like of float
        Dynamic (sampled); a *mixture* of characteristic lengthscales for
        the fixed random basis, not a single scale. For block ``i``, the
        effective ``W_hidden ~ N(0, 1/feature_scales[i]**2)`` (realized by
        rescaling the fixed ``raw_W_hidden`` noise -- see above): smaller
        values give higher-frequency (more wiggly / narrower-feature)
        components, larger values give smoother/broader ones. Its *length*
        (number of blocks, 4 by default) is fixed at construction and
        should not change across ``update()`` calls; the lengthscale
        *values* are free to vary continuously and can be sampled.

        A single scale is a real limitation, not just a convenience
        default: a typical GW mass function mixes a broad, smooth
        power-law-ish continuum with a few much narrower Gaussian peaks,
        and no number of features fixes a lengthscale that's simply wrong
        for the structure you're trying to resolve -- e.g. one peak with
        `sigma/mu ~ 0.03` standardized needs a much finer scale than the
        smooth body spanning most of the domain, and a single global scale
        can't serve both regimes at once (empirically: with a single
        scale, MSE against `bpl_dip_three_peaks` stayed high regardless
        of how many features were used, because the narrow low-mass peak
        was simply invisible at that scale; splitting the same feature
        budget across several scales fixes this without per-target
        tuning). The default ``(1.0, 0.3, 0.1, 0.03)`` spans broad-body to
        narrow-peak scales in standardized log-mass; widen/narrow it if
        your target has structure outside that range.
    raw_W_hidden, b_hidden : jnp.ndarray, optional
        Static-shaped, fixed (never sampled) i.i.d. N(0, 1) noise and
        phase. Drawn once at construction from ``key`` if not supplied
        explicitly.
    w_out : jnp.ndarray, optional
        Dynamic (sampled); linear combination of the effective random
        features. Bias-free (see note below); defaults to all-zeros (a
        flat log-density, i.e. uninformative starting point).
    key : jax.random.PRNGKey, optional
        Constructor-only key used to draw the fixed random noise/phase.
        Irrelevant once ``raw_W_hidden``/``b_hidden`` are supplied
        explicitly (e.g. when reconstructing a model instance via
        ``update``).

    beta : float
        Mass-ratio power-law index.

    Notes
    -----
    ``p̃(m)`` is truncated to exactly zero outside ``[m_low, m_high]`` by a
    hard mask, with no separate edge-smoothing scale: unlike an analytic
    power-law-ish shape, ``g`` is already a smooth (infinitely
    differentiable) function of mass by construction, so there is no sharp
    edge feature to soften -- a smoothing scale here would only add two
    unnecessary hyperparameters (as ``bottomsmooth``/``topsmooth`` used to
    be) without changing the fitted shape in any meaningful way.

    No bias/intercept term is added to the output layer: an overall
    additive shift in the log-density is exactly degenerate with the
    model's own 2-D renormalisation (it exponentiates to a constant
    factor that cancels between ``p̃(m1)*p̃(m2)`` and ``Z``), so it would
    only add an unidentifiable direction to the posterior.
    """

    input_size: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)

    # Dynamic (sampled) mixture of lengthscales -- see the class docstring
    # for why this is deliberately NOT eqx.field(static=True).
    feature_scales: jnp.ndarray

    # Fixed random noise/phase -- drawn once at construction and never
    # sampled, but deliberately NOT eqx.field(static=True): jit needs to
    # compare static metadata via `==`, and comparing two different-valued
    # arrays that way raises "truth value of an array is ambiguous" (see
    # the class docstring). Regular (non-static) leaves avoid that; jit
    # only needs their shape/dtype to decide on a retrace, values just
    # flow through as normal traced data.
    raw_W_hidden: jnp.ndarray
    b_hidden: jnp.ndarray

    # Static row -> feature_scales-block lookup, derived purely from the
    # static n_features/len(feature_scales); recomputed (cheaply) every
    # __init__ call, same as m_grid/norm_2d below.
    block_index: jnp.ndarray

    # The only other free/sampled network parameter.
    w_out: jnp.ndarray

    beta: float

    # Fixed preprocessing constants. These should not be sampled.
    x_mean: float = eqx.field(static=True)
    x_std: float = eqx.field(static=True)

    default = {
        **base_mass_paired_struct.default,
        "beta": 1.08,
        "input_size": 1,
        "n_features": 64,
        "feature_scales": (1.0, 0.3, 0.1, 0.03),
        "raw_W_hidden": None,
        "b_hidden": None,
        "w_out": None,
    }

    name = "paired_random_features_density"

    def __init__(self, key=None, **kwargs):
        # ------------------------------------------------------------------
        # Copied from base_mass_paired_struct.__init__
        # ------------------------------------------------------------------
        self.keys = list(self.default.keys())

        for k in self.keys:
            setattr(self, k, kwargs.get(k, self.default[k]))

        # ------------------------------------------------------------------
        # `feature_scales` is a dynamic leaf: normalize to a jnp array so
        # it flows through jit/grad/vmap like `w_out` does (accepts a
        # plain python tuple, e.g. the default, or an already-traced array
        # coming back through `update()`).
        # ------------------------------------------------------------------
        self.feature_scales = jnp.asarray(self.feature_scales, dtype=jnp.float_)

        # ------------------------------------------------------------------
        # Standardization constants
        # ------------------------------------------------------------------
        log_m_low = jnp.log(0.4)
        log_m_high = jnp.log(200.)

        self.x_mean = 0.5 * (log_m_low + log_m_high)
        self.x_std = 0.5 * (log_m_high - log_m_low)

        # ------------------------------------------------------------------
        # Draw the fixed random noise/phase if not explicitly supplied.
        # raw_W_hidden_k ~ N(0, 1), b_hidden_k ~ U(0, 2*pi). This noise is
        # independent of `feature_scales`: the lengthscale mixture is
        # applied at call time (see `_effective_W_hidden`), never baked
        # into the draw, precisely so that `feature_scales` can vary
        # continuously (and be sampled) without ever needing to redraw
        # this noise.
        # ------------------------------------------------------------------
        if self.raw_W_hidden is None or self.b_hidden is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            w_key, b_key = jax.random.split(init_key)

            self.raw_W_hidden = jax.random.normal(w_key, (self.n_features, self.input_size))
            self.b_hidden = jax.random.uniform(
                b_key, (self.n_features,), minval=0.0, maxval=2 * jnp.pi
            )

        if self.w_out is None:
            self.w_out = jnp.zeros(self.n_features)

        # ------------------------------------------------------------------
        # Static row -> block lookup (depends only on the static
        # n_features and the static *length* of feature_scales, never on
        # its values) -- cheap to recompute every call, same as m_grid.
        # ------------------------------------------------------------------
        self.block_index = _block_index_array(self.n_features, len(self.feature_scales))

        # ------------------------------------------------------------------
        # Copied from base_mass_paired_struct.__init__
        # ------------------------------------------------------------------
        self.m_grid = jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res)
        self.norm_2d = _compute_norm_2d(self)


# ===========================================================================
# Marginal density
# ===========================================================================

@dispatch
def mass_pdf_notnorm(mass: random_features_density, m: jnp.ndarray):
    """Unnormalized marginal mass density.

    The random-features map receives standardized log-mass,

        x = (log(m) - x_mean) / x_std.

    The permutation-free density is then

        p_tilde(m) = exp(g(x)) / m,

    hard-masked to zero outside [m_low, m_high]. No edge-smoothing scale is
    applied: g is already a smooth function of mass by construction (it's a
    sum of cosines), so there is no sharp analytic feature at the truncation
    edges to soften the way e.g. a power-law's edge needs to be.
    """
    # Standardized log-mass.
    x = (jnp.log(m) - mass.x_mean) / mass.x_std

    # random-features log-density.
    g = log_density(mass, x)

    # pdf, hard-truncated to [m_low, m_high]
    pdf = jnp.exp(g) / m
    pdf = jnp.where((m >= mass.m_low) & (m <= mass.m_high), pdf, 0.0)

    return pdf


# ===========================================================================
# Pairing
# ===========================================================================

@dispatch
def pairing_function(
    mass: random_features_density,
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
