from .utils.config import jax, jnp, logger
from .utils.math import trapz
import equinox as eqx
from typing import Callable, Optional, Dict
from numbers import Number
from .population import pop_rate_det
from .data import theta_inj_det
from functools import partial

class selection_function(object):
  r"""A class to estimate GW selection effects, that is the fraction :math:`\xi(\lambda)` of detected injections drawn from a population model.

  Args:
    inj_data (Dict[str, jnp.ndarray]): Injected data, typically contains the parameters used for injection (e.g., luminosity distances, masses).
    inj_prior (jnp.ndarray): Prior probabilities for the injected data.
    N_inj (int): Total number of generated injections.
    population (object): A CHIMERA.population object.
    neff_inj_min (Optional[Number], default=5): Minimum effective number of injections, used for filtering the completeness factor.

  Class Attributes:
    - check_Neff (bool): A flag indicating whether the minimum effective number of injections (`neff_inj_min`) is specified.
    - Tobs (int): Observation time.
  """
  def __init__(self,
    theta_inj_det: theta_inj_det,
    N_inj: int,
    N_eff: Optional[Number] = 5.,
  ):
    # ininitialize
    self.theta_inj_det = theta_inj_det
    self.N_inj = N_inj
    self.N_eff = N_eff

  @partial(jax.jit, static_argnums=(0,)) # may be useful to jit this for fast bias checking
  def N_exp(self, pop_lambdas):
      r"""Estimates the number of expected detected events of the population."""
      dNdtheta = pop_rate_det(pop_lambdas, self.theta_inj_det)
      dNdtheta /= self.theta_inj_det.p_draw # importance sampling from p_draw
      xi = jnp.nansum(dNdtheta, axis = -1) / self.N_inj
      # N_exp
      Nexp = pop_lambdas.Tobs * xi
      # Check Neff
      if self.N_eff is not None:
        variance2 = jnp.sum((dNdtheta)**2, axis = -1) / self.N_inj**2  - xi**2 / self.N_inj
        neff = xi**2 / variance2
        neff_cond = neff < self.N_eff
        Nexp = jnp.where(neff_cond, 0.0, Nexp)
      return Nexp

  @partial(jax.jit, static_argnums=(0,)) # may be useful to jit this for debugging
  def __call__(self, pop_lambdas):
    """Call the `N_exp` method"""
    return self.N_exp(pop_lambdas)
