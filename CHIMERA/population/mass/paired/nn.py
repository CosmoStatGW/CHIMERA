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
        Static; number of fixed random features / free ``w_out`` entries.
        Keep this modest (tens, not hundreds) to keep the sampled
        dimensionality low -- see the package README's guidance on
        gradient-based samplers' dimension scaling.
    feature_scale : float
        Static; controls the characteristic frequency of the fixed random
        basis (``W_hidden ~ N(0, 1/feature_scale**2)``): smaller
        ``feature_scale`` gives smoother functions, larger gives more
        wiggly ones.
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
    feature_scale: float = eqx.field(static=True)

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
        "n_features": 16,
        "feature_scale": 1.0,
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
        # W_hidden_k ~ N(0, 1/feature_scale**2), b_hidden_k ~ U(0, 2*pi):
        # a standard random-Fourier-features-style nonlinear basis.
        # ------------------------------------------------------------------
        if self.W_hidden is None or self.b_hidden is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            w_key, b_key = jax.random.split(init_key)

            self.W_hidden = jax.random.normal(
                w_key, (self.n_features, self.input_size)
            ) / self.feature_scale
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
