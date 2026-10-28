"""
mass.joint
==========
Genuinely joint 2-D mass models -- neither ``paired``
(p̃(m1) p̃(m2) f(m1,m2)/Z) nor ``conditioned`` (p(m1) p(m2|m1)). This
package holds two sibling families that differ in which coordinates the
basis expansion is native in:

(Mc, q)-native (``base.py``)
  p(m1, m2) = p(Mc, q) * |d(Mc,q)/d(m1,m2)|

  Using (Mc, q) as the native coordinates means the physical domain is a
  plain rectangle [mc_low, mc_high] x [q_low, 1] with no m2 <= m1
  ordering constraint to build into the model -- every point in that
  rectangle maps to exactly one valid m2 <= m1 pair.

  Models: random_features_joint_density -- fixed 2-D random nonlinear
  basis (never trained) plus a single trainable linear output layer,
  directly on (Mc, q) -- the 2-D-input analogue of
  ``mass.paired.random_features_density``.

(m1, m2)-native (``base_m1m2.py``)
  p(m1, m2) = exp(h(m1,m2)) / Z,   for m_min <= m2 <= m1 <= m_max

  A single shared ``m_min``/``m_max`` truncates *both* masses (the
  standard mass-model parametrization used throughout the GW population
  literature), with the ``m2 <= m1`` ordering enforced by an explicit
  mask -- the price paid for a literal shared mass range, since (unlike
  the (Mc,q) box) there is no coordinate change that turns
  ``[m_min,m_max]^2`` minus the ``m2>m1`` triangle into a mask-free
  rectangle. Densities here use an exp-link (``p ~ exp(h)``), not the
  squared-link (``p ~ h^2``) the (Mc,q)-native family above uses: smooth
  everywhere (no ``2h'/h``-type gradient singularity), at the cost of
  never being able to represent an exact interior zero and needing a
  numerical, log-sum-exp-stabilised normalisation integral (see
  ``base_m1m2.py``).

  Models: bspline2d_m1m2_density -- dense tensor-product B-spline,
  p(m1,m2) ~ exp(h(m1,m2)) with h = B(m1)^T C B(m2) and C a square
  (n_basis, n_basis) coefficient matrix (only its "live band" is ever
  sampled -- see that class's docstring and ``bspline_utils.band_mask``).
  A single shared basis for both axes, since m1 and m2 are drawn from the
  same underlying astrophysical population.

(An earlier (Mc,q)-native, squared-link dense B-spline model,
``bspline2d_joint_density``, has been retired in favor of
``bspline2d_m1m2_density`` -- see that class's docstring for the design
rationale.)

Dispatch functions (also accessible via ``mass.p_m1m2``)
----------------------------------------------------------
p_m1m2(mass, m1, m2)  or  p_m1m2(mass, theta)   -- shared across both families
pdf_joint_and_marg(mass, res)                    -- (Mc,q)-native family
pdf_joint_and_marg_m1m2(mass, res)               -- (m1,m2)-native family
mass_pdf_notnorm_mcq(mass, mc, q)                -- (Mc,q)-native family
mass_pdf_notnorm_m1m2(mass, m1, m2)              -- (m1,m2)-native family

Change-of-variables helpers ((Mc,q)-native family only)
--------------------------------------------------------
mc_of_m1m2(m1, m2), q_of_m1m2(m1, m2)
m1_of_mc_q(mc, q), m2_of_mc_q(mc, q)
jacobian_mc_q_from_m1m2(m1, mc)
"""

from .base import (
  base_mass_joint_struct,
  mass_pdf_notnorm_mcq,
  p_m1m2,
  pdf_joint_and_marg,
  mc_of_m1m2,
  q_of_m1m2,
  m1_of_mc_q,
  m2_of_mc_q,
  jacobian_mc_q_from_m1m2,
)
from .random_features import random_features_joint_density

# `base_m1m2`'s p_m1m2 is the *same* global plum generic as `base`'s above
# (plum generics are module-level singletons identified by the function
# object -- see CHIMERA.population.mass's own import block for the same
# pattern) -- importing it here just registers its (m1,m2)-native overload
# onto that shared object; the `as` alias only avoids rebinding the name
# `p_m1m2` already bound above. `pdf_joint_and_marg`, by contrast, is a
# plain (non-dispatched) function in both base modules, so it genuinely
# needs a distinct name here to avoid the second import silently shadowing
# the first.
from .base_m1m2 import (
  base_mass_joint_m1m2_struct,
  mass_pdf_notnorm_m1m2,
  p_m1m2 as _m1m2_p_m1m2,  # noqa: F401  (side-effect: registers overload)
  pdf_joint_and_marg as pdf_joint_and_marg_m1m2,
)
from .bspline2d import bspline2d_m1m2_density

__all__ = [
  'base_mass_joint_struct',
  'mass_pdf_notnorm_mcq',
  'p_m1m2',
  'pdf_joint_and_marg',
  'mc_of_m1m2',
  'q_of_m1m2',
  'm1_of_mc_q',
  'm2_of_mc_q',
  'jacobian_mc_q_from_m1m2',
  'random_features_joint_density',
  'base_mass_joint_m1m2_struct',
  'mass_pdf_notnorm_m1m2',
  'pdf_joint_and_marg_m1m2',
  'bspline2d_m1m2_density',
]
