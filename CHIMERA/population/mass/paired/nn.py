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
from ..core import high_pass_filter, low_pass_filter, smooth_step_up, smooth_step_down, truncated_pl

# ===========================================================================
# Neural log-density MLP
# ===========================================================================

def init_mlp_params(hidden_size, depth, input_size, key):
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
    h = jnp.atleast_1d(x)
    for W, b in zip(Ws[:-1], bs):
        h = jnp.tanh(W @ h + b)
    return (Ws[-1] @ h)[0]

def log_density(Ws, bs, x):
  flat_g = jax.vmap(
      lambda xi: _log_density_single(Ws, bs, xi)
  )(x.reshape(-1))
  return flat_g.reshape(x.shape) 

def log_density_interpolated(Ws, bs, x, x_grid):
    g_grid = jax.vmap(lambda xi: _log_density_single(Ws, bs, xi))(x_grid)
    g_grid = g_grid - jnp.mean(g_grid)
    return jnp.interp(x, x_grid, g_grid)

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

    #alpha: float
    beta: float
    bottomsmooth: float
    topsmooth: float

    # Fixed preprocessing constants. These should not be sampled.
    x_mean: float = eqx.field(static=True)
    x_std: float = eqx.field(static=True)
    x_grid_size: int = eqx.field(static=True)
    x_grid: jnp.ndarray = eqx.field(static=True)

    default = {
        **base_mass_paired_struct.default,
        #"alpha": 2.3,
        "beta": 1.08,
        "bottomsmooth": 3.3,
        "topsmooth": 3.3,
        "input_size": 1,
        "hidden_size": 8,
        "depth": 2,
        "x_grid_size": 5000,
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
        log_m_grid = jnp.linspace(log_m_low, log_m_high, self.x_grid_size)
        self.x_grid = (log_m_grid - self.x_mean) / self.x_std

        # ------------------------------------------------------------------
        # Initialize network if parameters were not explicitly supplied
        # ------------------------------------------------------------------
        if self.Ws is None or self.bs is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            
            self.Ws, self.bs = init_mlp_params(
                self.hidden_size,
                self.depth,
                self.input_size,
                init_key,
            )

        # ------------------------------------------------------------------
        # Copied from base_mass_paired_struct.__init__
        # ------------------------------------------------------------------
        self.m_grid = jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res)
        self.norm_2d = _compute_norm_2d(self)


# ===========================================================================
# Marginal density
# ===========================================================================

@dispatch
def mass_pdf_notnorm(mass: neural_density, m: jnp.ndarray):
    """Unnormalized marginal mass density.

    The neural network receives standardized log-mass,

        x = (log(m) - x_mean) / x_std.

    The permutation-free neural density is then

        p_tilde(m) = exp(g(x)) / m * S(m),

    where S(m) is the product of the low- and high-mass smoothing
    functions.

    The density is zero outside [m_low, m_high].
    """
    # Standardized log-mass.
    x = (jnp.log(m) - mass.x_mean) / mass.x_std

    # neural log-density.
    g = log_density(mass.Ws, mass.bs, x) # log_density_interpolated(mass.Ws, mass.bs, x, mass.x_grid) # 

    # pdf
    pdf = jnp.exp(g) / m  #  truncated_pl(m, -mass.alpha, mass.m_low, mass.m_high) 
    # Mass-edge smoothing.
    pdf *= high_pass_filter(m, mass.bottomsmooth, mass.m_low) * smooth_step_up(m, mass.m_low, steepness=200)
    pdf *= low_pass_filter(m, mass.topsmooth, mass.m_high) * smooth_step_down(m, mass.m_high, steepness=200)

    return pdf


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