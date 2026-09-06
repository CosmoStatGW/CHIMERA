import jax.numpy as jnp
import jax
from functools import partial
import equinox as eqx
from plum import dispatch
from typing import List
from .base import base_mass_conditioned_struct, secondary_mass_conditioned_pdf_notnorm
from ..core import truncated_pl, high_pass_filter
from ....utils.math import trapz, cumtrapz

# Semi-paramtrice PL+spline model:
# slightly modified w.r.t base_struct_mass to handle spline_basis and spline_coefficients
class pls(base_mass_conditioned_struct):
  alpha: float
  delta_m: float
  beta: float
  spline_coefficients: jnp.ndarray
  spline_basis_grid: jnp.ndarray  # output of setup_spline() routine
  spline_basis: jnp.ndarray # output of setup_spline() routine
  num_coeffs: int = eqx.field(static=True)
  default = {
    **base_mass_conditioned_struct.default,
    'alpha': 3.4,
    'delta_m': 4.8,
    'beta': 1.1,
    'spline_coefficients': None
  }
  keys: List[str] = eqx.field(static=True)
  name = 'powerlaw_plus_spline'

  def __init__(self, **kwargs): # overload
    self.spline_basis_grid = kwargs['spline_basis_grid'] # needed
    self.spline_basis = kwargs['spline_basis'] # needed
    self.num_coeffs = self.spline_basis.shape[1]
    self.default['spline_coefficients'] = jnp.zeros(self.num_coeffs)
    self.keys = list(self.default.keys())
    for key in self.keys:
      value = kwargs.get(key, self.default[key])
      setattr(self, key, value)

    # normalizations
    setattr(self, 'm_grid', jnp.logspace(jnp.log10(self.m_low), jnp.log10(self.m_high), self.m_grid_res))
    p_values = secondary_mass_conditioned_pdf_notnorm(self, self.m_grid, self.m_high)
    cdf_values = cumtrapz(p_values, self.m_grid)
    setattr(self, 'cdf_m2_conditioned', cdf_values)
    integrand_values = primary_mass_pdf_notnorm(self, self.m_grid)
    norm = trapz(integrand_values, x=self.m_grid)
    setattr(self, 'norm_p_m1', norm)

  # ad_dict method inherited from base_mass_conditioned_struct

  def update(self, **kwargs):
    keys_to_update = {k: v for k, v in kwargs.items() if k in self.keys}
    if keys_to_update == {}:
      return self
    fiducials = self.as_dict
    fiducials.update(keys_to_update)
    fiducials['spline_basis'] = self.spline_basis
    fiducials['spline_basis_grid'] = self.spline_basis_grid
    return self.__class__(**fiducials)

# Useful functions

def bspline_basis(i, x, knots, degree):
  # Compute i-th B-spline basis function of given degree
  if degree == 0:
    return jnp.where((knots[i] <= x) & (x < knots[i+1]), 1.0, 0.0)
  else:
    # Cox-de Boor recursion formula
    left = jnp.where(knots[i+degree] != knots[i],
                    (x - knots[i]) / (knots[i+degree] - knots[i]) * bspline_basis(i, x, knots, degree-1),
                    0.0)
    right = jnp.where(knots[i+degree+1] != knots[i+1],
                      (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1]) * bspline_basis(i+1, x, knots, degree-1),
                      0.0)
    return left + right

@partial(jax.jit, static_argnums=(2,))
def bspline_design_matrix(x, knots, degree):
  # JAX-compatible B-spline design matrix
  n_basis = len(knots) - degree - 1
  matrix = jax.vmap(bspline_basis, in_axes=(0, None, None, None))(jnp.arange(n_basis), x, knots, degree)
  matrix = matrix.at[-1,-1].set(1.0) # consistency with scipy
  return matrix.T

@partial(jax.jit, static_argnums=(2,3,4,5,6))
def setup_spline(x_low_prior,
                 x_high_prior,
                 num_inner_knots,
                 degree = 3,
                 knots_method = 'linear',
                 spline_basis_spacing = 'linear',
                 spline_grid_res=1000,
                 xdata=None,
                 inner_knots=None):
  # compute knots
  if inner_knots is not None:
    assert len(inner_knots) == num_inner_knots, "The length of the provided knots must match `num_inner_knots`."
    knot_list = inner_knots
  else:
    if knots_method == 'linear':
      assert x_low_prior is not None, "`xmin` must be a number if `knots_method='linear'`."
      assert x_high_prior is not None, "`xmax` must be a number if `knots_method='linear'`."
      knot_list = jnp.linspace(x_low_prior, x_high_prior, num_inner_knots)
    elif knots_method=='logspace':
      assert x_low_prior is not None, "`xmin` must be a number if `knots_method='linear'`."
      assert x_high_prior is not None, "`xmax` must be a number if `knots_method='linear'`."
      knot_list = jnp.logspace(jnp.log10(x_low_prior), jnp.log10(x_high_prior), num_inner_knots)
    elif knots_method == 'quantile':
      assert xdata is not None, "`xdata` must be an array of numbers if `knots_method='quantile'`."
      # this does not account for x_low/high priors, unless xdata is properly modelled
      knot_list = jnp.quantile(xdata, jnp.linspace(0, 1, num_inner_knots))
    else:
      raise ValueError("`knots_method` can be only `linear`, 'logspace' or `quantile`.")
  knots = jnp.concatenate((jnp.full(degree, knot_list[0]),
                            knot_list,
                            jnp.full(degree, knot_list[-1])))

  # compute spline basis
  if spline_basis_spacing == 'linear':
    spline_basis_grid = jnp.linspace(x_low_prior, x_high_prior, spline_grid_res)
  elif spline_basis_spacing == 'log':
    spline_basis_grid = jnp.logspace(jnp.log10(x_low_prior), jnp.log10(x_high_prior), spline_grid_res)
  else:
    raise ValueError("`spline_basis_spacing` can be only `linear` or 'log'.")
  spline_basis = bspline_design_matrix(spline_basis_grid, knots, degree)

  return knots, spline_basis_grid, spline_basis


# Main routine

@dispatch
def primary_mass_pdf_notnorm(mass:pls, m: jnp.ndarray):
  P = truncated_pl(m, -mass.alpha, mass.m_low, mass.m_high)
  spline_vals_on_grid = jnp.dot(mass.spline_basis, mass.spline_coefficients)
  spline_values = jnp.interp(m, mass.spline_basis_grid, spline_vals_on_grid)
  pdf = P*jnp.exp(spline_values)
  pdf *= high_pass_filter(m, mass.delta_m, mass.m_low)
  return pdf
