# random_features.py (joint)

import jax
import jax.numpy as jnp

from plum import dispatch
import equinox as eqx

from .base import (
    base_mass_joint_struct,
    mass_pdf_notnorm_mcq,
    _compute_norm_2d_mcq,
)
from ..paired.random_features import (
    _block_index_array,
    _effective_W_hidden,
    _random_features,
)

# ===========================================================================
# Joint (Mc, q) random-features log-density
# ===========================================================================

def _standardize(mass, mc, q):
    """Standardized 2-D input x = (x1, x2), broadcasting mc/q against each
    other first (they may arrive as e.g. (N,1) and (1,N) grid axes)."""
    mc_b, q_b = jnp.broadcast_arrays(mc, q)
    x1 = (jnp.log(mc_b) - mass.mc_mean) / mass.mc_std
    x2 = 2.0 * q_b - 1.0                      # q in (0,1] -> x2 in (-1,1], no logit needed
    return x1, x2


def joint_log_density(mass, mc, q):
    x1, x2 = _standardize(mass, mc, q)
    out_shape = x1.shape
    flat_x = jnp.stack([x1.reshape(-1), x2.reshape(-1)], axis=-1)  # (N, 2)

    W_hidden = _effective_W_hidden(mass.raw_W_hidden, mass.feature_scales, mass.block_index)

    def single(xi):
        phi = _random_features(xi, W_hidden, mass.b_hidden, mass.activation)
        return jnp.dot(mass.w_out, phi)

    flat_g = jax.vmap(single)(flat_x)
    return flat_g.reshape(out_shape)

# ===========================================================================
# Joint random-features density model
# ===========================================================================

class random_features_joint_density(base_mass_joint_struct):
    r"""Genuinely joint 2-D random-features mass model, parametrized
    directly in (chirp mass ``Mc``, mass ratio ``q``) -- neither
    ``paired`` (``p̃(m1) p̃(m2) f(m1,m2)/Z``) nor ``conditioned``
    (``p(m1) p(m2|m1)``).

    The joint distribution on the native coordinates is

    .. math::

        p(M_c, q) = \frac{\exp(g(\hat x))}{Z_{Mc,q}},

    with ``g`` a random-features expansion (same recipe as
    ``CHIMERA.population.mass.paired.random_features_density``, generalized
    to a 2-D input):

    .. math::

        g(\hat x) = \sum_{k=1}^{K} w_{\mathrm{out},k}\,
        \phi\!\left(W_{\mathrm{hidden},k}\cdot\hat x + b_{\mathrm{hidden},k}\right),
        \qquad \hat x = (\hat x_1, \hat x_2),

    where ``phi`` is ``cos`` (default) or ``tanh`` (see ``activation`` in
    ``random_features_density``'s docstring for the trade-off -- it applies
    identically here), and

    .. math::

        \hat x_1 = \frac{\log M_c - \overline{\log M_c}}{\sigma_{\log M_c}},
        \qquad
        \hat x_2 = 2q - 1,

    i.e. standardized log-chirp-mass and mass ratio linearly rescaled from
    ``(0, 1]`` to ``(-1, 1]`` (no logit transform: ``q``'s domain is already
    bounded, unlike a mass spanning orders of magnitude, so there is no
    "wide range" to compress -- and *not* transforming it is precisely
    what keeps the map to ``(m1, m2)`` free of any extra boundary to guard
    against, besides the plain box truncation).

    ``p(m1, m2)`` is obtained by the base class's generic ``p_m1m2``, which
    pushes ``p(Mc, q)`` forward through the Jacobian
    ``|d(Mc,q)/d(m1,m2)| = Mc/m1^2`` (see
    ``CHIMERA.population.mass.joint.base.jacobian_mc_q_from_m1m2``).

    -----------------------------------------------------------------------
    Free hyperparameters
    -----------------------------------------------------------------------

    ``w_out`` (shape ``(n_features,)``) and ``feature_scales`` are the
    sampled parameters, exactly as in ``random_features_density``.
    ``raw_W_hidden`` (shape ``(n_features, 2)`` -- one column per input
    dimension) and ``b_hidden`` are fixed at construction from ``key`` and
    never touched again. ``feature_scales`` rescales *both* input
    dimensions of a given block's rows by the same scalar (isotropic in
    ``(x1, x2)``); it does not give ``Mc`` and ``q`` independent
    lengthscales -- a natural follow-up would be a per-axis rescaling, not
    implemented here to keep this a direct, minimal-changes generalization
    of ``random_features_density``.

    -----------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------

    mc_low, mc_high : float
        Dynamic (sampled); chirp-mass truncation range (see base class).
    q_low : float
        Dynamic (sampled); lower bound on the mass ratio (see base class).
        ``q_high`` is fixed at exactly ``1.0``, not a free hyperparameter.
    n_features : int
        Static; total number of fixed random features / free ``w_out``
        entries. Keep modest (tens, not hundreds).
    activation : str
        Static; ``'cos'`` (default) or ``'tanh'``.
    feature_scales : array-like of float
        Dynamic (sampled); mixture of lengthscales, same role as in
        ``random_features_density``.
    raw_W_hidden, b_hidden : jnp.ndarray, optional
        Static-shaped, fixed (never sampled) noise/phase. Drawn once at
        construction from ``key`` if not supplied explicitly.
    w_out : jnp.ndarray, optional
        Dynamic (sampled); linear combination of the effective random
        features. Defaults to all-zeros.
    key : jax.random.PRNGKey, optional
        Constructor-only key.
    mc_grid_res, q_grid_res : int, optional
        Static; resolution of the ``(Mc, q)`` grid used for ``norm_2d``
        (default 800 each -- deliberately higher than the 1-D paired
        models' 500-point default: the *same* number of points per axis
        resolves a 2-D oscillatory function far less accurately than a 1-D
        one, since both axes can wiggle independently). Read directly
        from kwargs (not through ``.default``/``.keys``) and explicitly
        carried across ``.update()`` calls -- see ``__init__`` and
        ``base_mass_joint_struct.update``.

        **This is a real accuracy knob, not just a speed one.** ``norm_2d``
        is a finite grid trapezoid estimate, and for a sufficiently wiggly
        draw (small ``feature_scales`` combined with a large ``w_out``)
        800x800 can still under-count the true 2-D integral by a
        percent-level amount -- verified directly: for one moderately
        wiggly draw, the estimated ``Z`` grew from 100.12 (400x400) to
        100.48 (2500x2500) and was still not fully converged, a systematic
        ~0.4% bias at the old 400x400 default, confirmed independently
        with unbiased Monte-Carlo integration of the resulting ``p(m1,m2)``
        (>7 sigma from 1.0 at 40M samples). If you widen ``w_out``'s prior
        or shrink ``feature_scales`` beyond the defaults, increase
        ``mc_grid_res``/``q_grid_res`` and re-run a Monte-Carlo check like
        the one above -- or, better, calibrate ``w_out``'s prior scale the
        same way ``random_features_blocks.ipynb`` calibrates
        ``w_out_prior_scale`` from ``n_features``, which keeps the induced
        density smooth enough that a modest grid already resolves it well.

    Notes
    -----
    No bias/intercept term is added to the output layer, for the same
    reason as ``random_features_density``: an overall additive shift in
    the log-density is exactly degenerate with the model's own 2-D
    renormalisation.
    """

    n_features: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    feature_scales: jnp.ndarray
    raw_W_hidden: jnp.ndarray
    b_hidden: jnp.ndarray
    block_index: jnp.ndarray
    w_out: jnp.ndarray
    mc_mean: float = eqx.field(static=True)
    mc_std: float = eqx.field(static=True)

    default = {
        **base_mass_joint_struct.default,
        "n_features": 64,
        "activation": "cos",
        "feature_scales": (1.0, 0.3, 0.1, 0.03),
        "raw_W_hidden": None,
        "b_hidden": None,
        "w_out": None,
    }

    name = "joint_random_features_density"

    def __init__(self, key=None, **kwargs):
        # See base_mass_joint_struct.__init__ for why these are read directly from
        # kwargs rather than through `.default`/`.keys`.
        self.mc_grid_res = kwargs.get('mc_grid_res', 800)
        self.q_grid_res = kwargs.get('q_grid_res', 800)

        self.keys = list(self.default.keys())

        for k in self.keys:
            setattr(self, k, kwargs.get(k, self.default[k]))
        self.feature_scales = jnp.asarray(self.feature_scales, dtype=jnp.float_)

        if self.activation not in ('cos', 'tanh'):
            raise ValueError(f"Unknown activation {self.activation!r}, must be 'cos' or 'tanh'.")

        # Fixed, wide standardization range for log(Mc) -- independent of mc_low/mc_high,
        # which may themselves be sampled/narrower (same convention as random_features_density).
        log_mc_low_std = jnp.log(0.5)
        log_mc_high_std = jnp.log(150.)
        self.mc_mean = 0.5 * (log_mc_low_std + log_mc_high_std)
        self.mc_std = 0.5 * (log_mc_high_std - log_mc_low_std)

        if self.raw_W_hidden is None or self.b_hidden is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            w_key, b_key = jax.random.split(init_key)

            self.raw_W_hidden = jax.random.normal(w_key, (self.n_features, 2))  # input_size=2
            if self.activation == 'cos':
                self.b_hidden = jax.random.uniform(
                    b_key, (self.n_features,), minval=0.0, maxval=2 * jnp.pi
                )
            else:  # 'tanh'
                self.b_hidden = jax.random.normal(b_key, (self.n_features,))

        if self.w_out is None:
            self.w_out = jnp.zeros(self.n_features)
        self.block_index = _block_index_array(self.n_features, len(self.feature_scales))
        self.mc_grid = jnp.logspace(jnp.log10(self.mc_low), jnp.log10(self.mc_high), self.mc_grid_res)
        self.q_grid = jnp.linspace(self.q_low, 1.0, self.q_grid_res)
        self.norm_2d = _compute_norm_2d_mcq(self)

# ===========================================================================
# Joint (Mc, q) density
# ===========================================================================

@dispatch
def mass_pdf_notnorm_mcq(mass: random_features_joint_density, mc: jnp.ndarray, q: jnp.ndarray):
    """Unnormalized joint density p̃(Mc, q) = exp(g(Mc, q)), hard-masked to
    zero outside [mc_low, mc_high] x [q_low, 1]. No edge-smoothing scale:
    g is already smooth by construction (a random-features expansion), so
    there is no sharp analytic feature at the truncation edges to soften.
    """
    g = joint_log_density(mass, mc, q)
    pdf = jnp.exp(g)
    mc_b, q_b = jnp.broadcast_arrays(mc, q)
    inside = (mc_b >= mass.mc_low) & (mc_b <= mass.mc_high) & (q_b >= mass.q_low) & (q_b <= 1.0)
    return jnp.where(inside, pdf, 0.0)
