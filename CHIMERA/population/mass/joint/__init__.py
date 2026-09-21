"""
mass.joint
==========
Genuinely joint 2-D mass models, parametrized directly in
(chirp mass Mc, mass ratio q) rather than in (m1, m2) -- neither
``paired`` (p̃(m1) p̃(m2) f(m1,m2)/Z) nor ``conditioned`` (p(m1) p(m2|m1)).

  p(m1, m2) = p(Mc, q) * |d(Mc,q)/d(m1,m2)| = p(Mc, q) * Mc/m1^2

Using (Mc, q) as the native coordinates means the physical domain is a
plain rectangle [mc_low, mc_high] x [q_low, 1] with no m2 <= m1 ordering
constraint to build into the model -- every point in that rectangle maps
to exactly one valid m2 <= m1 pair.

Available models
----------------
random_features_joint_density : Fixed 2-D random nonlinear basis (never
    trained) plus a single trainable linear output layer, directly on
    (Mc, q) -- the 2-D-input analogue of
    ``mass.paired.random_features_density``.

Dispatch functions (also accessible via ``mass.p_m1m2``)
----------------------------------------------------------
p_m1m2(mass, m1, m2)  or  p_m1m2(mass, theta)
pdf_joint_and_marg(mass, res)
mass_pdf_notnorm_mcq(mass, mc, q)

Change-of-variables helpers
----------------------------
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
]
