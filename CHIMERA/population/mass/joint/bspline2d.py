# bspline2d.py (joint, 2-D dense tensor-product B-spline, native m1, m2)

import jax
import jax.numpy as jnp
from typing import List
from plum import dispatch
import equinox as eqx

from .base_m1m2 import base_mass_joint_m1m2_struct, _h_m1m2, _compute_shift_and_norm_2d
from ..bspline_utils import (
    _axis_grid_design, n_basis_to_num_inner_knots,
    band_indices, n_band_coeffs,
)
from ....utils.math import interp2d

# ===========================================================================
# Dense tensor-product B-spline log-density kernel, native (m1, m2)
# ===========================================================================


def _dense_coeffs(mass):
    """Scatter the free, "live-band" coefficient vector ``mass.coeffs``
    (shape ``(n_band_coeffs(n_basis, degree),)``) into the full, dense
    ``(n_basis, n_basis)`` matrix ``h`` is built from, filling every
    structurally non-physical entry (``bspline_utils.band_mask``, ``j > i
    + degree``) with 0.

    Any fixed value would do identically here, not just 0 -- those entries
    are provably inert (exactly zero gradient w.r.t. h in the physical
    ``m2<=m1`` region, for *any* coefficient values elsewhere), so this
    scatter changes nothing about the represented density; it only
    reconstructs the shape ``bspline2d_h_m1m2``'s matmul needs from the
    minimal free parametrization actually being sampled/optimized.
    """
    row_idx, col_idx = band_indices(mass.n_basis, mass.degree)
    return jnp.zeros((mass.n_basis, mass.n_basis)).at[row_idx, col_idx].set(mass.coeffs)


def bspline2d_h_m1m2(mass, m1, m2):
    """h(m1, m2) = B(m1)^T C B(m2), a full tensor-product B-spline surface.
    ``C`` is the dense, *square* ``(n_basis, n_basis)`` matrix obtained by
    scattering the free "live-band" vector ``mass.coeffs`` via
    ``_dense_coeffs`` (see that function and ``bspline_utils.band_mask``
    for why only a band, not the full square, is ever sampled). Both axes
    reuse the exact same knots/basis (built once from the single shared
    ``m_min``/``m_max``/``n_basis``/``degree``/``knot_spacing``), since m1
    and m2 are draws from the same underlying astrophysical mass
    distribution -- a single shared basis is the natural, more
    parameter-economical choice here (unlike ``random_features_joint_density``'s
    (Mc, q), which are genuinely different quantities and keep independent
    per-axis bases).
    """
    m1_b, m2_b = jnp.broadcast_arrays(m1, m2)

    grid, design = _axis_grid_design(
        mass.m_min, mass.m_max, mass.num_inner_knots, mass.degree,
        mass.knot_spacing, mass.spline_grid_res,
    )

    # h on the tensor grid: (G, K) @ (K, K) @ (K, G) -> (G, G)
    h_grid = design @ _dense_coeffs(mass) @ design.T

    return interp2d(m1_b, m2_b, grid, grid, h_grid)


# ===========================================================================
# Dense tensor-product B-spline joint density model, native (m1, m2)
# ===========================================================================

class bspline2d_m1m2_density(base_mass_joint_m1m2_struct):
    r"""Genuinely joint 2-D dense tensor-product B-spline mass model, native
    directly in (m1, m2), with a standard shared ``m_min``/``m_max`` for
    *both* masses and an exp-link density.

    .. math::

        p(m_1, m_2) = \frac{\exp(h(m_1, m_2))}{Z}\,
        \mathbb{1}[m_{min} \le m_2 \le m_1 \le m_{max}], \qquad
        h(m_1, m_2) = \sum_{i=1}^{n} \sum_{j=1}^{n}
        c_{ij}\, B_i(m_1)\, B_j(m_2),

    i.e. ``h = B(m_1)^T\,C\,B(m_2)`` with ``C`` a *dense, square*
    ``(n_basis, n_basis)`` coefficient matrix -- a single shared
    ``n_basis``/``degree``/``knot_spacing`` basis is used for *both* axes
    (see ``bspline2d_h_m1m2``'s docstring for why).

    This is the sole dense tensor-product B-spline mass model in this
    package (an earlier (Mc,q)-native, squared-link sibling,
    ``bspline2d_joint_density``, was retired in favor of this one -- native
    (m1,m2) coordinates plus an exp-link density, described next).

    **Design: native (m1, m2), with an exp-link density**

    Two central design choices:

    1. **Native coordinates**: directly in (m1, m2), with the
       ``m2 <= m1`` ordering enforced by an explicit mask (as in
       ``paired``/``conditioned``), rather than via a coordinate change to
       a mask-free rectangular domain. This buys a literal, standard,
       immediately-interpretable ``m_min``/``m_max`` shared by both
       masses -- a rectangle in (Mc, q), by contrast, induces an
       (m1,m2) support region that does *not* coincide with
       ``[m_min,m_max]^2`` (a known subtlety), so this is the
       construction to reach for when a textbook shared mass range is
       wanted.
    2. **exp-link, not squared-link**: ``p ~ exp(h)`` rather than
       ``p ~ h^2``. ``d(log h^2)/dc = 2h'/h`` has a genuine singularity at
       ``h=0``; ``log p = h - h_shift - log(Z)`` is smooth everywhere
       since ``h`` is bounded. The price is that this model cannot
       represent an exact interior zero, and ``Z`` needs the
       log-sum-exp-stabilised numerical grid integral implemented in
       ``base_m1m2.py``'s ``_compute_shift_and_norm_2d`` rather than (in
       principle) a closed form.

    Dimensionality: ``n_band_coeffs(n_basis, degree) + 2`` (only two
    boundary scalars -- ``m_min``, ``m_max``).
    ``n_band_coeffs(n_basis, degree) < n_basis**2`` strictly whenever
    ``degree < n_basis - 1`` -- see "Live-band parametrization" below.

    **Live-band parametrization: not the full ``n_basis**2`` dense matrix**

    ``coeffs`` is **not** the full dense ``(n_basis, n_basis)`` matrix --
    it is a flat vector over only the "live band" ``{(i,j) : j <= i +
    degree}`` (``bspline_utils.band_mask``/``band_indices``/
    ``n_band_coeffs``). ``_dense_coeffs`` scatters it back into the dense
    square (filling every other entry with 0) before ``bspline2d_h_m1m2``
    evaluates ``h``.

    This is not an approximation: entries with ``j > i + degree`` are
    *exactly* non-physical. Because both axes share one knot sequence,
    ``B_i(m1)`` and ``B_j(m2)`` can be simultaneously active at some
    physical point (``m2 <= m1``) only if their compact,
    ``degree+1``-knot-wide supports can "reach across" the ``m1=m2``
    boundary, which reduces to the index condition ``j <= i + degree`` --
    a pure consequence of B-spline support, independent of
    ``m_min``/``m_max``/``knot_spacing``. For every ``(i,j)`` outside that
    band, the gradient of ``h`` (and hence of the likelihood) with
    respect to ``C[i,j]`` is identically zero everywhere, not just small
    (verified numerically: ``jax.grad`` of ``h`` summed over a dense
    physical grid reproduces exactly the ``band_mask`` pattern). Exposing
    those entries as free sampled parameters would therefore waste
    sampler dimensions on directions the likelihood can never constrain
    -- the posterior there would just equal the prior, exactly. They are
    not simply set to 0 as a modeling choice; 0 is used in
    ``_dense_coeffs`` only because *any* fixed value is equally and
    provably inert.

    **Parametrization: ``n_basis``, not knot count**

    Configured directly by how many coefficients ("basis functions") the
    (shared) basis has -- ``n_basis`` -- rather than by the
    knot-construction detail (``num_inner_knots``) that implies it, via
    the same ``bspline_utils.n_basis_to_num_inner_knots`` inversion used
    throughout this package (mirroring the reference ``logBspline`` 1-D
    implementation's ``n_basis``-first construction pattern).

    Unlike that reference, this class does not pin any boundary
    coefficients to exactly zero (the reference's 1-D
    ``coeffs[0]=coeffs[-1]=0``, forcing the spline to vanish at the domain
    edges): the 2-D generalization would require zeroing an entire edge
    row/column of ``C`` and is a genuine, optional modeling choice, left
    out here.

    **Parameters**

    m_min, m_max : float
        Dynamic (sampled); shared mass truncation range for both m1 and
        m2; see base class.
    n_basis : int
        Static; number of B-spline coefficients ("basis functions") along
        each axis (default 12). Both axes share this same count and the
        same knots.
    degree : int
        Static; B-spline degree (default 3, cubic).
    knot_spacing : str
        Static; ``'log'``/``'linear'`` fixed knot placement, shared by
        both axes (default ``'log'`` -- avoids the representational
        inefficiency of a linear grid wasting resolution on the long
        high-mass tail). Knots are fixed, not inferred.
    spline_grid_res : int
        Static; resolution of the grid the (shared) B-spline design
        matrix is built on; arbitrary-point evaluation is by 2-D
        (separable-bilinear) interpolation on this grid.
    coeffs : jnp.ndarray, optional
        Dynamic (sampled); flat shape ``(n_band_coeffs(n_basis, degree),)``
        over the live band only (see above), whitened N(0,1) prior per
        element. Drawn once from ``key`` if not supplied.
    m_grid_res : int, optional
        Static; see base class for the accuracy-vs-cost discussion.
    key : jax.random.PRNGKey, optional
        Constructor-only key for the default coefficient draw.
    """

    n_basis: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    knot_spacing: str = eqx.field(static=True)
    spline_grid_res: int = eqx.field(static=True)
    num_inner_knots: int = eqx.field(static=True)
    coeffs: jnp.ndarray

    default = {
        **base_mass_joint_m1m2_struct.default,
        'n_basis': 12,
        'degree': 3,
        'knot_spacing': 'log',
        'spline_grid_res': 1000,
        'coeffs': None,
    }
    keys: List[str] = eqx.field(static=True)
    name = 'joint_m1m2_bspline2d_density'

    def __init__(self, key=None, **kwargs):
        # See base_mass_joint_m1m2_struct.__init__ for why this is read
        # directly from kwargs rather than through `.default`/`.keys`.
        self.m_grid_res = kwargs.get('m_grid_res', 800)

        self.keys = list(self.default.keys())
        for k in self.keys:
            setattr(self, k, kwargs.get(k, self.default[k]))

        if self.knot_spacing not in ('log', 'linear'):
            raise ValueError(f"Unknown knot_spacing {self.knot_spacing!r}, must be 'log' or 'linear'.")
        if self.n_basis <= self.degree:
            raise ValueError(f"n_basis ({self.n_basis}) must be > degree ({self.degree}).")

        # Knots are derived from n_basis, not set directly -- see class docstring.
        self.num_inner_knots = n_basis_to_num_inner_knots(self.n_basis, self.degree)

        n_free = n_band_coeffs(self.n_basis, self.degree)
        if self.coeffs is None:
            init_key = key if key is not None else jax.random.PRNGKey(0)
            self.coeffs = jax.random.normal(init_key, (n_free,))
        else:
            self.coeffs = jnp.asarray(self.coeffs, dtype=jnp.float_)
            if self.coeffs.shape != (n_free,):
                raise ValueError(
                    f"coeffs has shape {self.coeffs.shape}, expected a flat "
                    f"(n_band_coeffs,) = ({n_free},) vector over the live "
                    f"band {{(i,j) : j <= i + degree}} (see "
                    f"bspline_utils.band_mask) -- not the full dense "
                    f"(n_basis, n_basis) = ({self.n_basis}, {self.n_basis}) matrix."
                )

        self.m_grid = jnp.logspace(jnp.log10(self.m_min), jnp.log10(self.m_max), self.m_grid_res)
        self.h_shift, self.norm_2d = _compute_shift_and_norm_2d(self)


# ===========================================================================
# Log-density kernel dispatch
# ===========================================================================

@dispatch
def _h_m1m2(mass: bspline2d_m1m2_density, m1: jnp.ndarray, m2: jnp.ndarray):
    """h(m1, m2) = B(m1)^T C B(m2) -- not itself masked/truncated; the
    base class's ``mass_pdf_notnorm_m1m2``/``_compute_shift_and_norm_2d``
    apply the ``[m_min,m_max]`` x ``m2<=m1`` mask."""
    return bspline2d_h_m1m2(mass, m1, m2)
