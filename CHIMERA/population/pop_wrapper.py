import equinox as eqx
from typing import Optional, Union, List, Dict
from numbers import Number
from plum import dispatch
from ..utils.config import jax, jnp, logger
from ..utils.math import trapz
from ..utils import angles
from ..population.cosmo import dVcdz_at_z, Vc_at_z, z_from_dGW, ddLdz_at_z
from ..population.mass import p_m1m2
from ..population.rate import merger_rate
from ..catalog.catalog import empty_catalog
from ..data import theta_src, theta_pe_det, theta_inj_det, theta_generic

class population(eqx.Module):
  cosmo: eqx.Module
  mass: eqx.Module
  rate: eqx.Module
  R0: Number = 1.
  gal_cat: object = eqx.field(static=True)
  Tobs: Number = eqx.field(static=True)
  scale_free: bool = eqx.field(static=True)

  def __init__(
    self,
    cosmo: eqx.Module,
    mass: eqx.Module,
    rate: eqx.Module,
    R0: Number = 1.,
    gal_cat: object = None,  # Allow None as default
    Tobs: Number = 1,
    scale_free: bool = True,
  ):
    # If gal_cat is None, initialize it with empty_catalog(cosmo)
    self.cosmo = cosmo
    self.mass = mass
    self.rate = rate
    self.R0 = R0
    if gal_cat is None:
      gal_cat = empty_catalog(p_bkg='dVdz')
    self.gal_cat = gal_cat

    self.Tobs = Tobs
    self.scale_free = scale_free

  def __repr__(self):
    return (
      f"cosmo = {self.cosmo},\n"
      f"mass = {self.mass},\n"
      f"rate = {self.rate},\n"
      f"R0 = {self.R0},\n"
      f"galcat_obj = {self.gal_cat},\n"
      f"Tobs = {self.Tobs},\n"
      f"scale_free = {self.scale_free}"
    )

  def update(self, **hyper_lambdas):
    cosmo_lambdas = self.cosmo.update(**hyper_lambdas)
    mass_lambdas = self.mass.update(**hyper_lambdas)
    rate_lambdas = self.rate.update(**hyper_lambdas)
    R0 = hyper_lambdas.get('R0', self.R0)
    return self.__class__(
      cosmo_lambdas, mass_lambdas, rate_lambdas, R0,
      self.gal_cat, self.Tobs, self.scale_free
    )

# Functions
def theta_det2src(cosmo_lambdas, theta_det, include_original_distances=False):
  # convert from det to source
  z = z_from_dGW(cosmo_lambdas, theta_det.dL)
  m1s, m2s = theta_det.m1det/(1.+z), theta_det.m2det/(1.+z)
  if include_original_distances:
    return theta_src(m1src=m1s, m2src=m2s, z=z,
      original_distances=theta_det.dL)
  else:
    return theta_src(m1src=m1s, m2src=m2s, z=z)

def get_theta_src_and_weights(pop_lambdas:population, theta_det:theta_pe_det):
  th_src = theta_det2src(pop_lambdas.cosmo, theta_det)
  weights = p_m1m2(pop_lambdas.mass, th_src)/theta_det.pe_prior
  return th_src, weights

def p_cbc(pop_lambdas:population, z:jnp.ndarray):
  """Computes redshift prior"""
  p_gal = pop_lambdas.gal_cat.p_gal(pop_lambdas.cosmo, z)
  p_rate = merger_rate(pop_lambdas.rate, z) / (1+z)
  if p_gal.ndim > p_rate.ndim:
    p_z = jnp.where(p_gal != -100, p_gal * p_rate[:, None, :], -100)  # pixelated
  else:
    p_z = p_gal * p_rate # no pixels
  return p_z

@dispatch
def pop_rate_det(pop_lambdas:population, th_det:theta_pe_det):
  #Computes the population rate in detector frame for GW events. May be useful but is not used
  theta_pe_src = theta_det2src(pop_lambdas.cosmo, th_det)
  p_z = p_cbc(pop_lambdas, theta_pe_src.z)
  dNdtheta = pop_lambdas.R0*p_m1m2(pop_lambdas.mass, theta_pe_src)*p_z
  jacobian = jnp.abs(ddLdz_at_z(pop_lambdas.cosmo, theta_pe_src))*(1.+theta_pe_src.z)**2
  dNdtheta /= jacobian
  return dNdtheta

@dispatch
def pop_rate_det(pop_lambdas:population, th_det:theta_inj_det):
  """Computes the population rate in detector frame for injections."""
  theta_inj_src = theta_det2src(pop_lambdas.cosmo, th_det, include_original_distances=True)
  p_z = pop_lambdas.gal_cat.p_bkg(pop_lambdas.cosmo, theta_inj_src)
  p_z *= merger_rate(pop_lambdas.rate, theta_inj_src)/(1.+theta_inj_src.z)
  dNdtheta = pop_lambdas.R0*p_m1m2(pop_lambdas.mass, theta_inj_src)*p_z
  jacobian = jnp.abs(ddLdz_at_z(pop_lambdas.cosmo, theta_inj_src))*(1.+theta_inj_src.z)**2
  dNdtheta /= jacobian
  return dNdtheta

@dispatch
def pop_rate_det(pop_lambdas:population, th_src:theta_src):
  """Computes the population rate in detector frame for mock data."""
  p_z = pop_lambdas.gal_cat.p_bkg(pop_lambdas.cosmo, th_src)
  p_z *= merger_rate(pop_lambdas.rate, th_src)/(1.+th_src.z)
  dNdtheta = pop_lambdas.R0*p_m1m2(pop_lambdas.mass, th_src)*p_z
  jacobian = jnp.abs(ddLdz_at_z(pop_lambdas.cosmo, th_src))*(1.+th_src.z)**2
  dNdtheta /= jacobian
  return dNdtheta

def N_cbc_1yr(pop_lambdas:population):
  """Computes the total number of expected CBC sources in 1 year."""
  zz    = jnp.linspace(*[0.001,pop_lambdas.cosmo.z_max], 10_000)
  dN_dz = merger_rate(pop_lambdas.rate, zz)/(1.+zz) * pop_lambdas.gal_cat.p_bkg(pop_lambdas.cosmo, zz)
  dN_dz *= pop_lambdas.R0
  N = trapz(dN_dz, x = zz, axis = -1)
  return N

# compute z_grids

def compute_z_grids(cosmo: eqx.Module,
  theta_det: theta_pe_det,
  cosmo_prior: Optional[Dict[str, List[Number]]] = None,
  z_int_res: int = 300,
  z_conf_range: Optional[Union[Number, list]] = None,
):
  """Computes the redshift grids on which GW events have support, given some cosmological priors.

  Args:
    cosmo (eqx.Module): a CHIMERA.cosmo object describing cosmological params.
    events_dL (jnp.ndarray): GW events luminosity distance samples
    cosmo_prior (optional, Dict[str, List[Numbers]]): dict containing tuple of parameters, describing the priors over cosmological parameters.
    z_int_res (int): number fo points of the redshift grid.
    z_conf_range (Number or List): sigma or quantiles defining the redshift grid extenstion.

  Returns:
    jnp.ndarray: the redshift grids.
  """
  events_dL = theta_det.dL
  if isinstance(z_conf_range, list):
    dL_min, dL_max = jnp.percentile(events_dL, z_conf_range, axis=1)
  elif isinstance(z_conf_range, Number):
    mu     = jnp.mean(events_dL, axis=1)
    sig    = jnp.std(events_dL, axis=1)
    dL_min = mu - z_conf_range * sig
    dL_max = mu + z_conf_range * sig
  else:
    dL_max = jnp.max(events_dL, axis = 1)*2
    dL_min = jnp.min(events_dL, axis = 1)*0.5
    dL_min = jnp.where(dL_min < 1.e-8, 1.e-8, dL_min)

  # update cosmo prior
  cp = {k:[v, v] for k,v in cosmo.as_dict.items()}
  if cosmo_prior is not None:
    cp.update(cosmo_prior)
  if cosmo.name=='flrw':
    lc_low = {"H0" :cp['H0'][0],
      "Om0":cp['Om0'][0],
      "Ok0":cp['Ok0'][0],
      "Or0":cp['Or0'][0],
      "w0" :cp['w0'][0],
      "wa" :cp['wa'][0]
    }
    lc_high = {"H0" :cp['H0'][1],
      "Om0":cp['Om0'][1],
      "Ok0":cp['Ok0'][1],
      "Or0":cp['Or0'][1],
      "w0" :cp['w0'][1],
      "wa" :cp['wa'][1]
    }
  else:
    lc_low  = {"H0" :cp['H0'][0],
      "Om0":cp['Om0'][0],
      "Ok0":cp['Ok0'][0],
      "Or0":cp['Or0'][0],
      "w0" :cp['w0'][0],
      "wa" :cp['wa'][0],
      "Xi0":cp['Xi0'][1],
      "n"  :cp['n'][1]
    }
    lc_high = {"H0" :cp['H0'][1],
      "Om0":cp['Om0'][1],
      "Ok0":cp['Ok0'][1],
      "Or0":cp['Or0'][1],
      "w0" :cp['w0'][1],
      "wa" :cp['wa'][1],
      "Xi0":cp['Xi0'][0],
      "n"  :cp['n'][1] # true if cp['Xi0'][0] < 1, otherwise should be cp['n'][0]
    }
  cosmo1 = cosmo.update(**lc_low, z_grid_res = 10_000)
  cosmo2 = cosmo.update(**lc_high, z_grid_res = 10_000)

  z_min  = z_from_dGW(cosmo1, dL_min)
  z_max  = z_from_dGW(cosmo2, dL_max)
  z_grids = jnp.linspace(z_min, z_max, z_int_res, axis=1)
  return z_grids
