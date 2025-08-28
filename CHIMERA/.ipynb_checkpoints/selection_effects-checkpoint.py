from .utils.config import jax, jnp, logger, trapz
import equinox as eqx
from typing import Callable, Optional, Dict
from numbers import Number

class selection_effects(object):
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
    inj_data: Dict[str, jnp.ndarray],
    inj_prior: jnp.ndarray,
    N_inj: int,
    population: object,
    neff_inj_min: Optional[Number] = 5.,
  ):
    # ininitialize
    self.inj_data = inj_data
    self.inj_prior = inj_prior
    self.N_inj = N_inj
    self.population = population
    self.neff_inj_min = neff_inj_min

    # Get some useful attributes
    self.Tobs = self.population.Tobs
    self.check_Neff = self.neff_inj_min is not None

  def compute_Nexp(self, **hyper_params):
    r"""Estimates the number of expected detected events of the population."""
    dNdtheta = self.population.compute_pop_rate_inj(self.inj_data, self.inj_prior, **hyper_params)
    xi = jnp.sum(dNdtheta, axis = -1) / self.N_inj
    # N_exp
    Nexp = self.Tobs * xi
    # Manually check neff
    variance2 = jnp.sum((dNdtheta)**2, axis = -1) / self.N_inj**2  - xi**2 / self.N_inj
    neff = xi**2 / variance2
    if self.check_Neff:
      neff_cond = neff < self.neff_inj_min
      Nexp = jnp.where(neff_cond, 0.0, Nexp)
    return Nexp

  def __call__(self, **hyper_params):
    """Call the `compute_Ndet_exp` method"""
    return self.compute_Nexp(**hyper_params)
