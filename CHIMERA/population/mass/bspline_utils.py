"""Shared helpers for building dense tensor-product B-spline densities,
configured directly by ``n_basis`` (how many coefficients/basis functions an
axis has) rather than by knot count. Used by
``CHIMERA.population.mass.joint.bspline2d_m1m2_density`` (native directly
in ``m1``, ``m2``) -- the only dense-B-spline model in this package;
``joint``'s other model, ``random_features_joint_density``, is a different,
non-B-spline construction and does not use these helpers.
"""

import numpy as np
import jax.numpy as jnp

from .conditioned.pls import bspline_design_matrix


def n_basis_to_num_inner_knots(n_basis, degree):
    """Invert ``n_basis = num_inner_knots + degree - 1`` (the standard
    open/clamped B-spline basis-count formula) to get the
    ``num_inner_knots`` a knot vector needs to produce exactly ``n_basis``
    basis functions of the given ``degree``.

    This is the key move that lets a model be configured directly by its
    actual parameter count (``n_basis``, i.e. how many coefficients one
    axis has) rather than by the knot-construction detail
    (``num_inner_knots``) that implies it -- ``num_inner_knots`` is then a
    derived quantity, computed once at construction time and never set
    directly. Mirrors e.g. ``logBspline._setup_grid_and_knots``'s
    ``interior = linspace(..., n_basis - degree + 1)`` construction.
    """
    return n_basis - degree + 1


def fixed_knots(x_low, x_high, num_inner_knots, degree, spacing):
    """Build a clamped B-spline knot vector spanning [x_low, x_high].

    Knots are NOT free/inferred in this first implementation (unlike
    ``paired.pls_free_knots``) -- ``num_inner_knots`` interior knots are
    placed uniformly in log- or linear-space, then clamped by repeating
    the first/last knot ``degree`` times, exactly as
    ``pls_free_knots.compute_spline_basis`` and
    ``conditioned.pls.setup_spline`` do. Free knots are a natural future
    extension if a fixed grid turns out to need too many knots for a given
    target shape.
    """
    if spacing == 'log':
        knot_list = jnp.logspace(jnp.log10(x_low), jnp.log10(x_high), num_inner_knots)
    elif spacing == 'linear':
        knot_list = jnp.linspace(x_low, x_high, num_inner_knots)
    else:
        raise ValueError(f"Unknown spacing {spacing!r}, must be 'log' or 'linear'.")
    return jnp.concatenate((jnp.full(degree, knot_list[0]),
                             knot_list,
                             jnp.full(degree, knot_list[-1])))


def _axis_grid_design(x_low, x_high, num_inner_knots, degree, spacing, grid_res):
    """Fixed knots + B-spline design matrix on a dedicated grid for one
    axis. Arbitrary-point evaluation always goes through this grid plus
    interpolation, rather than calling ``bspline_design_matrix`` directly
    on event-level values (a scipy-consistency hack in
    ``bspline_design_matrix`` is only correct when the *last* input point
    is exactly the grid's upper edge).

    Used by the sole dense tensor-product B-spline model in this package,
    ``joint.bspline2d_m1m2_density`` (``joint.bspline2d``).
    """
    knots = fixed_knots(x_low, x_high, num_inner_knots, degree, spacing)
    if spacing == 'log':
        grid = jnp.logspace(jnp.log10(x_low), jnp.log10(x_high), grid_res)
    else:
        grid = jnp.linspace(x_low, x_high, grid_res)
    design = bspline_design_matrix(grid, knots, degree)  # (G, K)
    return grid, design


# ===========================================================================
# "Live band" of a *shared-basis* tensor-product coefficient matrix
# ===========================================================================
#
# Used by joint.bspline2d_m1m2_density, where both axes (m1, m2) reuse the
# exact same knots/basis and the physical density is masked to m2<=m1. Not
# meaningful for joint's other model, random_features_joint_density (a
# different, non-B-spline, non-shared-basis construction).


def band_mask(n_basis, degree):
    """Boolean ``(n_basis, n_basis)`` mask, ``True`` at ``(i, j)`` iff
    coefficient ``C[i, j]`` of a *shared-basis* tensor-product B-spline
    (same knot sequence reused for both m1 and m2) can have any effect at
    all on a density masked to ``m2 <= m1``.

    Derivation: for a clamped B-spline, ``supp(B_i) = [t_i,
    t_{i+degree+1}]`` (Cox-de Boor). Since here both axes share the exact
    same knot sequence, ``B_i(m1)`` and ``B_j(m2)`` can be simultaneously
    nonzero at some physical point (``m2 <= m1``) iff ``t_j <=
    t_{i+degree+1}``, which collapses to the purely combinatorial
    condition ``j <= i + degree`` -- it depends only on ``n_basis`` and
    ``degree``, never on ``m_min``/``m_max``/``knot_spacing``, since it's a
    direct consequence of each basis function's compact,
    ``degree+1``-knot-wide support.

    Entries with ``j > i + degree`` are *exactly*, not approximately,
    non-physical: the gradient of ``h`` (and so of the likelihood) with
    respect to them is identically zero everywhere, verified numerically
    by evaluating ``jax.grad`` of ``h`` summed over a dense physical grid
    and checking it reproduces this exact mask. A dense ``(n_basis,
    n_basis)`` parametrization therefore wastes
    ``n_basis**2 - n_band_coeffs(n_basis, degree)`` sampler dimensions on
    directions the likelihood can never constrain (posterior = prior,
    exactly) -- see ``n_band_coeffs``/``band_indices`` below, used by
    ``joint.bspline2d_m1m2_density`` to parametrize only the live band.
    """
    i = np.arange(n_basis)[:, None]
    j = np.arange(n_basis)[None, :]
    return j <= i + degree


def band_indices(n_basis, degree):
    """``(row_idx, col_idx)`` int arrays listing every "live" ``(i, j)``
    pair of ``band_mask(n_basis, degree)``, in a fixed, deterministic
    (row-major, i.e. ``np.nonzero``) order.

    This exact order is the contract between every piece of code that
    iterates the band: the model's own coefficient vector
    (``bspline2d_m1m2_density.coeffs``), the dense-matrix reconstruction
    it is scattered into (``joint.bspline2d._dense_coeffs``), and
    anything external that must agree on what a flat free-coefficient
    vector means element-by-element (e.g. a sampler's per-element prior
    registration/unpacking, as in the run-script adapter's
    ``BSpline2DM1M2Spec``).
    """
    return np.nonzero(band_mask(n_basis, degree))


def n_band_coeffs(n_basis, degree):
    """Number of "live" coefficients in ``band_mask(n_basis, degree)`` --
    the free parameter count of a shared-basis tensor-product B-spline
    density's coefficient vector, strictly less than the full
    ``n_basis**2`` dense count whenever ``degree < n_basis - 1``.
    """
    return int(band_mask(n_basis, degree).sum())
