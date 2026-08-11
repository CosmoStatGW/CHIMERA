# neural_density.py

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
from ..core import high_pass_filter, low_pass_filter


# ===========================================================================
# Neural log-density MLP
# ===========================================================================

def _ordered_bias(raw_b):
    """Map unconstrained latent parameters to centered, ordered biases.

    The returned biases satisfy

        b_0 < b_1 < ... < b_(H-1)

    while remaining centered around the first raw parameter and having
    a finite, controlled span.
    """
    if raw_b.size == 0 or raw_b.size == 1:
        return raw_b
    gaps = jax.nn.softplus(raw_b[1:]) + 1e-4
    cumulative = jnp.concatenate(
        [
            jnp.zeros(1, dtype=raw_b.dtype),
            jnp.cumsum(gaps),
        ]
    )
    cumulative = cumulative / cumulative[-1]
    positions = cumulative - 0.5
    center = raw_b[0]

    return center + positions



def init_mlp_params(hidden_size, depth, input_size, key):
    """Randomly initialize the permutation-free MLP parameters.

    The returned biases are *unconstrained latent parameters*. They are
    transformed by `_ordered_bias` at evaluation time into the actual
    hidden-layer biases.

    Parameters
    ----------
    hidden_size : int
        Number of neurons in each hidden layer.
    depth : int
        Number of hidden layers.
    input_size : int
        Input dimensionality.
    key : jax.random.PRNGKey
        Random key.

    Returns
    -------
    Ws : list[jnp.ndarray]
        Network weight matrices. ``len(Ws) == depth + 1``.
        The final matrix is the bias-free output layer.

    bs : list[jnp.ndarray]
        Unconstrained latent bias vectors. ``len(bs) == depth``.
        Actual biases are obtained through `_ordered_bias`.
    """
    sizes = [input_size] + [hidden_size] * depth
    keys = jax.random.split(key, depth + 1)

    Ws = []
    bs = []

    for i in range(depth):
        w_key, b_key = jax.random.split(keys[i])

        lim = 1.0 / jnp.sqrt(sizes[i])

        W = jax.random.uniform(
            w_key,
            (sizes[i + 1], sizes[i]),
            minval=-lim,
            maxval=lim,
        )

        # These are latent/raw biases. `_ordered_bias` converts them
        # into the actual strictly ordered biases.
        b_raw = jax.random.uniform(
            b_key,
            (sizes[i + 1],),
            minval=-lim,
            maxval=lim,
        )

        Ws.append(W)
        bs.append(b_raw)

    # Bias-free output layer.
    lim = 1.0 / jnp.sqrt(hidden_size)

    Ws.append(
        jax.random.uniform(
            keys[-1],
            (1, hidden_size),
            minval=-lim,
            maxval=lim,
        )
    )

    return Ws, bs


# ===========================================================================
# Neural log-density forward pass
# ===========================================================================

def _log_density_single(Ws, bs, x):
    """Evaluate g(x) for one scalar standardized log-mass.

    Every hidden layer uses strictly ordered biases, which removes the
    exact hidden-neuron permutation symmetry of the unconstrained MLP.
    """
    h = jnp.atleast_1d(x)

    for W, b_raw in zip(Ws[:-1], bs):
        b = _ordered_bias(b_raw)
        h = jax.nn.softplus(W @ h + b)

    # Output layer has no bias. A constant output offset is degenerate
    # with the normalization of the density.
    return (Ws[-1] @ h)[0]


def log_density(Ws, bs, x):
    """Vectorized permutation-free neural log-density.

    Parameters
    ----------
    Ws, bs
        MLP parameters. ``bs`` are the unconstrained latent bias
        parameters; `_ordered_bias` is applied internally.

    x : array-like
        Scalar or array of arbitrary shape.

    Returns
    -------
    jnp.ndarray
        g(x), with the same shape as ``x``.
    """
    x = jnp.asarray(x)

    flat_g = jax.vmap(
        lambda xi: _log_density_single(Ws, bs, xi)
    )(x.reshape(-1))

    return flat_g.reshape(x.shape)


# ===========================================================================
# Paired neural density model
# ===========================================================================

class neural_density(base_mass_paired_struct):
    r"""Paired neural-density mass model.

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

    The neural log-density g is represented by a Softplus MLP.

    -----------------------------------------------------------------------
    Permutation-free parameterization
    -----------------------------------------------------------------------

    Hidden-neuron permutation symmetry is removed by parameterizing every
    hidden-layer bias vector through `_ordered_bias`.

    The stored ``bs`` are therefore NOT the physical biases directly.
    They are unconstrained latent parameters whose transformation gives

        b_0 < b_1 < ... < b_(H-1).

    For every ordinary MLP with distinct hidden-layer biases, there is a
    permutation of the neurons that puts those biases into this ordering.
    Thus this removes the redundant neuron-label representations without
    changing the represented function class, apart from the measure-zero
    case of exactly coincident biases.

    No `sort()` is used, so the transformation remains differentiable and
    suitable for NUTS.

    -----------------------------------------------------------------------
    Network shape
    -----------------------------------------------------------------------

    ``input_size``, ``hidden_size`` and ``depth`` are static fields.

    ``Ws`` and ``bs`` are ordinary pytree leaves and can therefore be
    sampled directly by the inference machinery.

    ``Ws`` contain ordinary unconstrained weights.

    ``bs`` contain unconstrained latent variables which are transformed
    into ordered physical biases by `_ordered_bias`.

    -----------------------------------------------------------------------
    Standardization
    -----------------------------------------------------------------------

    The neural network input is

        x = (log(m) - x_mean) / x_std

    where ``x_mean`` and ``x_std`` are computed from ``m_low`` and
    ``m_high`` at construction time.

    The ``1/m`` factor is the Jacobian associated with the transformation
    from log-mass to mass.

    -----------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------

    hidden_size, depth, input_size : int
        Static network shape.

    Ws, bs : list[jnp.ndarray], optional
        Explicit network parameters. ``bs`` are latent/raw bias
        parameters, not the ordered physical biases.

    key : jax.random.PRNGKey, optional
        Constructor-only initialization key.

    beta : float
        Mass-ratio power-law index.

    bottomsmooth, topsmooth : float
        Low- and high-mass edge smoothing scales.
    """

    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    Ws: List[jnp.ndarray]
    bs: List[jnp.ndarray]

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
        "hidden_size": 4,
        "depth": 2,
        "Ws": None,
        "bs": None,
    }

    name = "paired_neural_density"

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
        # Initialize network if parameters were not explicitly supplied
        # ------------------------------------------------------------------
        if self.Ws is None or self.bs is None:
            init_key = (
                key
                if key is not None
                else jax.random.PRNGKey(0)
            )

            self.Ws, self.bs = init_mlp_params(
                self.hidden_size,
                self.depth,
                self.input_size,
                init_key,
            )

        # ------------------------------------------------------------------
        # Copied from base_mass_paired_struct.__init__
        # ------------------------------------------------------------------
        self.m_grid = jnp.logspace(
            jnp.log10(self.m_low),
            jnp.log10(self.m_high),
            self.m_grid_res,
        )

        self.norm_2d = _compute_norm_2d(self)


# ===========================================================================
# Marginal density
# ===========================================================================

@dispatch
def mass_pdf_notnorm(
    mass: neural_density,
    m: jnp.ndarray,
):
    """Unnormalized marginal mass density.

    The neural network receives standardized log-mass,

        x = (log(m) - x_mean) / x_std.

    The permutation-free neural density is then

        p_tilde(m) = exp(g(x)) / m * S(m),

    where S(m) is the product of the low- and high-mass smoothing
    functions.

    The density is zero outside [m_low, m_high].
    """
    m = jnp.asarray(m)

    # Standardized log-mass.
    x = (jnp.log(m) - mass.x_mean) / mass.x_std

    # Permutation-free neural log-density.
    g = log_density(mass.Ws, mass.bs, x)

    # Jacobian from x = log(m) to m.
    pdf = jnp.exp(g) / m

    # Mass-edge smoothing.
    pdf *= high_pass_filter(
        m,
        mass.bottomsmooth,
        mass.m_low,
    )

    pdf *= low_pass_filter(
        m,
        mass.topsmooth,
        mass.m_high,
    )

    return jnp.where(
        (m >= mass.m_low) & (m <= mass.m_high),
        pdf,
        0.0,
    )


# ===========================================================================
# Pairing
# ===========================================================================

@dispatch
def pairing_function(
    mass: neural_density,
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