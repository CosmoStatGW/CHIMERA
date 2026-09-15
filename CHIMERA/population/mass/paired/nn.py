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
from ..core import high_pass_filter, low_pass_filter, smooth_step_up, smooth_step_down

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

def _random_features(x, W_hidden, b_hidden):
    """phi_k(x) = cos(W_hidden_k . x + b_hidden_k), shape (n_features,)."""
    x = jnp.atleast_1d(x)
    return jnp.cos(W_hidden @ x + b_hidden)


def log_density(mass, x):
    def single(xi):
        phi = _random_features(xi, mass.W_hidden, mass.b_hidden)
        return jnp.dot(mass.w_out, phi)

    flat_g = jax.vmap(single)(x.reshape(-1))
    return flat_g.reshape(x.shape)


def _block_feature_counts(n_features, feature_scales):
    """Split n_features as evenly as possible across len(feature_scales)
    blocks (one per lengthscale), spreading any remainder over the first
    few blocks so the total is always exactly n_features."""
    n_scales = len(feature_scales)
    base_count, remainder = divmod(n_features, n_scales)
    return [base_count + (1 if i < remainder else 0) for i in range(n_scales)]


# ===========================================================================
# Fitting-free capacity diagnostics
# ===========================================================================
#
# An MSE fit to some target shape (see fit_to_target_p_m1 in
# examples/nn.ipynb) is NOT a reliable way to answer "is n_features enough
# for these feature_scales": the answer it gives is confounded by the
# comparison grid resolution, floating-point precision, and gradient-descent
# local minima (see the notebook's discussion) -- none of which are
# properties of the *basis*. The two functions below answer that question
# directly, with no target shape, no optimizer, and no grid-comparison step
# at all: `suggested_n_features` gives a closed-form starting point from
# domain width and lengthscale alone, and `kernel_approximation_error`
# exactly (deterministically) checks how well the *actual drawn* fixed
# random basis approximates the theoretical GP kernel it's meant to.

def suggested_n_features(feature_scales, m_low, m_high, coverage=8.0):
    """Closed-form, fitting-free estimate of how many random features each
    block of `feature_scales` needs to adequately cover its frequency band
    over ``[m_low, m_high]``.

    This is a standard Monte-Carlo-coverage rule of thumb, not a fit to any
    target: a block at lengthscale ``ell`` needs on the order of
    ``coverage * domain_width / ell`` random draws before it has sampled
    enough of its own frequency band to behave like the lengthscale it's
    supposed to approximate (``domain_width`` measured in the model's own
    standardized log-mass units, i.e. after the fixed ``x = (log m -
    x_mean) / x_std`` transform). It says nothing about whether that
    lengthscale itself is the right choice for your data -- see the class
    docstring / package discussion for how to set `feature_scales` from
    measurement precision and catalog size.

    Parameters
    ----------
    feature_scales : tuple[float, ...]
        The lengthscale mixture to size, e.g. ``(1.0, 0.3, 0.1, 0.03)``.
    m_low, m_high : float
        The mass range to cover (typically the model's own truncation
        edges, not the fixed 0.4-200 standardization range).
    coverage : float
        Draws-per-wavelength constant (default 8; higher is more
        conservative / demands more features per block).

    Returns
    -------
    dict with keys:
      ``domain_width`` -- the standardized-log-mass span of [m_low, m_high].
      ``per_block`` -- list[int], one suggested count per feature_scales entry.
      ``total`` -- int, sum of per_block (a reasonable starting n_features).
    """
    log_m_low_std = jnp.log(0.4)
    log_m_high_std = jnp.log(200.0)
    x_mean = 0.5 * (log_m_low_std + log_m_high_std)
    x_std = 0.5 * (log_m_high_std - log_m_low_std)
    x_low = (jnp.log(m_low) - x_mean) / x_std
    x_high = (jnp.log(m_high) - x_mean) / x_std
    domain_width = float(x_high - x_low)

    per_block = [int(jnp.ceil(coverage * domain_width / scale)) for scale in feature_scales]
    return {
        "domain_width": domain_width,
        "per_block": per_block,
        "total": int(sum(per_block)),
    }


def kernel_approximation_error(mass, w_out_prior_scale=1.0, n_points=200):
    r"""Fitting-free check of how well `mass`'s FIXED random-feature
    realization approximates the theoretical mixture-RBF kernel implied by
    its `feature_scales`, over its own ``[m_low, m_high]`` mass range.

    Draws no samples: `mass.W_hidden`/`mass.b_hidden` were already drawn
    once at construction, so the kernel that *this specific realization*
    implies,

    .. math::

        K_\mathrm{realized}(x, x') = \sigma^2 \sum_k
        \cos(w_k x + b_k)\cos(w_k x' + b_k),

    is computed exactly (a single matrix product over the fixed features),
    and compared against the ensemble-averaged theoretical kernel this
    configuration is *supposed* to approximate (see the class docstring
    for the Bochner's-theorem / mixture-RBF derivation),

    .. math::

        K_\mathrm{theory}(x, x') = \frac{\sigma^2}{2} \sum_i n_i
        \exp\!\left(-\frac{(x-x')^2}{2\,\ell_i^2}\right),

    with one term per ``feature_scales`` entry :math:`\ell_i` and
    :math:`n_i` features in that block (:math:`\sigma^2` =
    ``w_out_prior_scale**2``, the assumed i.i.d. prior variance on
    ``w_out``). A small relative error means `mass.n_features` is enough
    for `mass.feature_scales` to behave like the intended GP kernel
    mixture *for this drawn realization*; a large one -- especially
    concentrated at short ``|x - x'|`` -- means the finer scales need more
    features (see `suggested_n_features` for a starting point).

    Parameters
    ----------
    mass : random_features_density
        The model instance to check (uses its own W_hidden/b_hidden,
        feature_scales, n_features, m_low/m_high/x_mean/x_std).
    w_out_prior_scale : float
        The assumed i.i.d. prior std on each w_out entry -- doesn't need
        to match any specific prior you'll actually use for inference,
        it's just an overall normalization of the comparison.
    n_points : int
        Grid resolution for the (n_points, n_points) kernel matrices.

    Returns
    -------
    dict with keys ``x`` (standardized-log-mass grid, shape (n_points,)),
    ``K_realized``, ``K_theory`` (both (n_points, n_points)), and
    ``relative_frobenius_error`` (scalar summary: ||K_realized -
    K_theory||_F / ||K_theory||_F).
    """
    x_low = (jnp.log(mass.m_low) - mass.x_mean) / mass.x_std
    x_high = (jnp.log(mass.m_high) - mass.x_mean) / mass.x_std
    x = jnp.linspace(x_low, x_high, n_points)

    Phi = jax.vmap(lambda xi: _random_features(xi, mass.W_hidden, mass.b_hidden))(x)
    sigma2 = w_out_prior_scale ** 2
    K_realized = sigma2 * (Phi @ Phi.T)

    counts = _block_feature_counts(mass.n_features, mass.feature_scales)
    dx = x[:, None] - x[None, :]
    K_theory = jnp.zeros_like(K_realized)
    for n_i, ell_i in zip(counts, mass.feature_scales):
        K_theory = K_theory + (sigma2 / 2.0) * n_i * jnp.exp(-(dx ** 2) / (2.0 * ell_i ** 2))

    rel_err = float(jnp.linalg.norm(K_realized - K_theory) / jnp.linalg.norm(K_theory))

    return {
        "x": x,
        "K_realized": K_realized,
        "K_theory": K_theory,
        "relative_frobenius_error": rel_err,
    }


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

    where ``W_hidden`` and ``b_hidden`` are drawn once at construction
    time and then fixed (never sampled) -- only the linear output weights
    ``w_out`` are free hyperparameters. This is deliberately *not* trained
    or tuned against any simulated population: the random basis is a fixed
    nonlinear feature map, and ``w_out`` is inferred directly inside the
    hierarchical Bayesian fit like every other hyperparameter here, so
    there is no separate pretraining stage and no dependence on assumed
    simulated mass-function shapes.

    -----------------------------------------------------------------------
    Parameter count
    -----------------------------------------------------------------------

    Only ``w_out`` (shape ``(n_features,)``) is a free/sampled parameter.
    ``W_hidden`` and ``b_hidden`` are static: fixed at construction from
    ``key`` and excluded from the sampled pytree entirely.

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
    feature_scales : tuple[float, ...]
        Static; a *mixture* of characteristic lengthscales for the fixed
        random basis, not a single scale. For block ``i``,
        ``W_hidden ~ N(0, 1/feature_scales[i]**2)``: smaller values give
        higher-frequency (more wiggly / narrower-feature) components,
        larger values give smoother/broader ones.

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
    W_hidden, b_hidden : jnp.ndarray, optional
        Static, fixed (never sampled) random projection and phase. Drawn
        once at construction from ``key`` if not supplied explicitly.
    w_out : jnp.ndarray, optional
        The only trainable/sampled parameter: linear combination of the
        fixed random features. Bias-free (see note below); defaults to
        all-zeros (a flat log-density, i.e. uninformative starting point).
    key : jax.random.PRNGKey, optional
        Constructor-only key used to draw the fixed random hidden layer.
        Irrelevant once ``W_hidden``/``b_hidden`` are supplied explicitly
        (e.g. when reconstructing a model instance via ``update``).

    beta : float
        Mass-ratio power-law index.

    bottomsmooth, topsmooth : float
        Low- and high-mass edge smoothing scales.

    Notes
    -----
    No bias/intercept term is added to the output layer: an overall
    additive shift in the log-density is exactly degenerate with the
    model's own 2-D renormalisation (it exponentiates to a constant
    factor that cancels between ``p̃(m1)*p̃(m2)`` and ``Z``), so it would
    only add an unidentifiable direction to the posterior.
    """

    input_size: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    feature_scales: tuple = eqx.field(static=True)

    # Fixed (static) random hidden layer -- drawn once at construction,
    # never part of the sampled pytree.
    W_hidden: jnp.ndarray = eqx.field(static=True)
    b_hidden: jnp.ndarray = eqx.field(static=True)

    # The only free/sampled network parameter.
    w_out: jnp.ndarray

    beta: float
    bottomsmooth: float
    topsmooth: float

    # Fixed preprocessing constants. These should not be sampled.
    x_mean: float = eqx.field(static=True)
    x_std: float = eqx.field(static=True)

    default = {
        **base_mass_paired_struct.default,
        "beta": 1.08,
        "bottomsmooth": 3.3,
        "topsmooth": 3.3,
        "input_size": 1,
        "n_features": 64,
        "feature_scales": (1.0, 0.3, 0.1, 0.03),
        "W_hidden": None,
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
        # Standardization constants
        # ------------------------------------------------------------------
        log_m_low = jnp.log(0.4)
        log_m_high = jnp.log(200.)

        self.x_mean = 0.5 * (log_m_low + log_m_high)
        self.x_std = 0.5 * (log_m_high - log_m_low)

        # ------------------------------------------------------------------
        # Draw the fixed random hidden layer if not explicitly supplied.
        # W_hidden_k ~ N(0, 1/feature_scales[i]**2), b_hidden_k ~ U(0, 2*pi):
        # a mixture of random-Fourier-features-style nonlinear bases, one
        # block of features per entry in `feature_scales`, so the fixed
        # basis spans several lengthscales at once (see the class
        # docstring for why a single scale isn't enough in general).
        # ------------------------------------------------------------------
        if self.W_hidden is None or self.b_hidden is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            w_key, b_key = jax.random.split(init_key)

            counts = _block_feature_counts(self.n_features, self.feature_scales)
            w_keys = jax.random.split(w_key, len(self.feature_scales))
            W_blocks = [
                jax.random.normal(wk, (count, self.input_size)) / scale
                for wk, count, scale in zip(w_keys, counts, self.feature_scales)
                if count > 0
            ]
            self.W_hidden = jnp.concatenate(W_blocks, axis=0)
            self.b_hidden = jax.random.uniform(
                b_key, (self.n_features,), minval=0.0, maxval=2 * jnp.pi
            )

        if self.w_out is None:
            self.w_out = jnp.zeros(self.n_features)

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

        p_tilde(m) = exp(g(x)) / m * S(m),

    where S(m) is the product of the low- and high-mass smoothing
    functions.

    The density is zero outside [m_low, m_high].
    """
    # Standardized log-mass.
    x = (jnp.log(m) - mass.x_mean) / mass.x_std

    # random-features log-density.
    g = log_density(mass, x)

    # pdf
    pdf = jnp.exp(g) / m
    # Mass-edge smoothing.
    pdf *= high_pass_filter(m, mass.bottomsmooth, mass.m_low) * smooth_step_up(m, mass.m_low, steepness=200)
    pdf *= low_pass_filter(m, mass.topsmooth, mass.m_high) * smooth_step_down(m, mass.m_high, steepness=200)

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
