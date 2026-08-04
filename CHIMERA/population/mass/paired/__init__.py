"""
mass.paired
===========
Paired symmetric mass models:

  p(m1, m2) = p̃(m1) * p̃(m2) * f(m1,m2) / Z,   m2 <= m1

Available models
----------------
plp : Power-Law + Gaussian Peak with power-law-in-q pairing function.

Dispatch functions (also accessible via ``mass.p_m1m2`` etc.)
-------------------------------------------------------------
p_m1m2(mass, m1, m2)  or  p_m1m2(mass, theta)
p_m1(mass, m)          or  p_m1(mass, theta)
pdf_joint_and_marg(mass, res)
mass_pdf_notnorm(mass, m)
pairing_function(mass, m1, m2)
"""

from .base import (
  base_mass_paired_struct,
  mass_pdf_notnorm,
  pairing_function,
  p_m1m2,
  p_m1,
  pdf_joint_and_marg,
)
from .plp import plp
from .pls_free_knots import pls_free_knots
from .bpl_dip_three_peaks import bpl_dip_three_peaks
from .bpl_dip_two_peaks import bpl_dip_two_peaks
from .nn import neural_density

__all__ = [
  'base_mass_paired_struct',
  'mass_pdf_notnorm',
  'pairing_function',
  'p_m1m2',
  'p_m1',
  'pdf_joint_and_marg',
  'plp',
  'pls_free_knots',
  'bpl_dip_three_peaks',
  'bpl_dip_two_peaks',
  'neural_density',
]
