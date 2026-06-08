"""
mass
====
Joint mass-distribution models for compact binary populations.

Sub-packages
------------
conditioned
  p(m1, m2) = p(m1) * p(m2|m1)
  Models: bpl, plp, pl2p, pls

paired
  p(m1, m2) = p̃(m1) * p̃(m2) * f(m1,m2) / Z
  Models: plp

Usage
-----
Import models by namespace, use dispatch functions from the top level::

  from CHIMERA.mass import p_m1m2, p_m1, pdf_joint_and_marg
  from CHIMERA.mass.conditioned import plp as plp_cond
  from CHIMERA.mass.paired     import plp as plp_paired

  model_c = plp_cond()
  model_p = plp_paired()

  p_m1m2(model_c, m1, m2)   # conditioned joint PDF
  p_m1m2(model_p, m1, m2)   # paired joint PDF
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

# ── 3. Make sub-packages accessible as mass.conditioned / mass.paired
from . import conditioned  # noqa: F401
from . import paired       # noqa: F401

__all__ = [
  # top-level dispatch functions
  'p_m1m2',
  'p_m1',
  'pdf_joint_and_marg',
  'setup_spline',
  # sub-packages
  'conditioned',
  'paired',
]
