"""Joint mass-distribution models for compact binary populations.

Sub-packages::

  conditioned
    p(m1, m2) = p(m1) * p(m2|m1)
    Models: bpl, plp, pl2p, pls

  paired
    p(m1, m2) = p̃(m1) * p̃(m2) * f(m1,m2) / Z
    Models: plp, pls_free_knots, bpl_dip_two_peaks, bpl_dip_three_peaks,
            random_features_density, nn_density

  joint
    Two genuinely-joint 2-D sibling families, sharing one package:
    - (Mc, q)-native: p(m1, m2) = p(Mc, q) * |d(Mc,q)/d(m1,m2)|
      Models: random_features_joint_density
    - (m1, m2)-native: p(m1, m2) = exp(h(m1,m2)) / Z, shared m_min/m_max,
      explicit m2 <= m1 mask
      Models: bspline2d_m1m2_density

Usage
-----
Import models by namespace, use dispatch functions from the top level::

  from CHIMERA.mass import p_m1m2, p_m1, pdf_joint_and_marg
  from CHIMERA.mass.conditioned import plp as plp_cond
  from CHIMERA.mass.paired     import plp as plp_paired
  from CHIMERA.mass.joint      import random_features_joint_density, bspline2d_m1m2_density

  model_c = plp_cond()
  model_p = plp_paired()
  model_j = random_features_joint_density()
  model_jm = bspline2d_m1m2_density()

  p_m1m2(model_c, m1, m2)   # conditioned joint PDF
  p_m1m2(model_p, m1, m2)   # paired joint PDF
  p_m1m2(model_j, m1, m2)   # joint (Mc,q)-native PDF
  p_m1m2(model_jm, m1, m2)  # joint (m1,m2)-native PDF
"""

# ── 1. Import the plum generic functions from conditioned (they are created
#       there first with their initial overloads).
from .conditioned.base import (
  p_m1m2,
  p_m1,
  pdf_joint_and_marg,
)
from .conditioned.pls import setup_spline

# ── 2. Import paired dispatch functions.  Because plum generics are module-
#       level singletons identified by the function object, importing these
#       registers the paired overloads onto the *same* generic objects that
#       conditioned already exported — so p_m1m2 above now also dispatches
#       on paired structs.
from .paired.base import (  # noqa: F401  (side-effect: registers overloads)
  p_m1m2 as _paired_p_m1m2,
  p_m1   as _paired_p_m1,
  pdf_joint_and_marg as _paired_pdf_joint_and_marg,
)

# ── 2b. Same for `joint`'s two sibling families -- registers both of their
#        p_m1m2 overloads onto the same global dispatcher too. Neither has
#        a p_m1 (see their base modules). `joint.base_m1m2`'s overload is
#        imported directly (not via `joint`'s own `__init__`, which already
#        aliases it as `_m1m2_p_m1m2` for its own namespace's sake) so this
#        file doesn't need to know that alias name.
from .joint.base import (  # noqa: F401  (side-effect: registers overload)
  p_m1m2 as _joint_p_m1m2,
)
from .joint.base_m1m2 import (  # noqa: F401  (side-effect: registers overload)
  p_m1m2 as _joint_m1m2_p_m1m2,
)

# ── 3. Make sub-packages accessible as mass.conditioned / mass.paired / mass.joint
from . import conditioned  # noqa: F401
from . import paired       # noqa: F401
from . import joint        # noqa: F401

__all__ = [
  # top-level dispatch functions
  'p_m1m2',
  'p_m1',
  'pdf_joint_and_marg',
  'setup_spline',
  # sub-packages
  'conditioned',
  'paired',
  'joint',
]
