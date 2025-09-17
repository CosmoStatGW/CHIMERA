import os, sys
os.environ["CHIMERA_ENABLE_GPU"] = "True"
sys.path.append(os.getcwd()+'/../../')

from CHIMERA.utils.config import jax, jnp, logger
import CHIMERA.utils.emcee_utils as eu
from CHIMERA import data
from CHIMERA.cosmo import mg_flrw, flrw
from CHIMERA.mass import plp
from CHIMERA.rate import madau_dickinson
from CHIMERA.catalog import empty_catalog
from CHIMERA import population
from CHIMERA import selection_effects
from CHIMERA import hyperlikelihood

import emcee

################################################################

mcmc_settings = {
  # chain output settings
  'output_dir': '/path/to/outdir/',
  'chain_prefix': 'mcmc_test',
  'restart_chain' : False,
  # emcee settings
  'nsteps': 10000,
  'nwalkers' : 100
}

dir_data = "/path/to/data/"
file_inj = dir_data+"injections.h5"
file_ev  = dir_data+"gwPE.h5"

#################################################################

# 1. Load data (non pixelated gw catalog)
events_pe = load_data_gw(file_ev)
pe_priors = events_pe['dL']**2
inj_data, inj_prior = load_data_injection(file_inj, snr_cut=20)
# inj_prior *= 1000 needed for LVK injection drawn from \pi_{draw} in which ddL/dz is in Mpc instead of Gpc

# 2. Instantiate population, bias and like objs
cosmo_obj = flrw()
mass_obj  = plp()
rate_obj  = madau_dickinson()
cat_obj   = empty_catalog(cosmo_obj)
population = population(cosmo_obj, mass_obj, rate_obj, cat_obj)

sel_eff = selection_effects(inj_data,
  inj_prior,
  N_inj=20*1e6,
  population=population
)

hyperlike = hyperlikelihood(
  # data
  events_pe,
  pe_priors,
  # pop model
  population=population,
  # bias model
  sel_eff=sel_eff,
  # KDE settings
  kernel    ='epan',
  bw_method = None,
  cut_grid  = 2,
  binning   = True,
  num_bins  = 200,
  # z_grids settings,
  cosmo_prior={'H0':[10.,200.]},
  z_conf_range = 5,
  z_int_res    = 300
)

# 3. Priors
priors = {"H0":  [10.,200.],
  "lambda_peak": [0.01,0.99],
  "alpha":       [1.5,12.],
  "beta":        [-4,12.],
  "delta_m":     [0.01,10.],
  "m_low":       [2.,50.],
  "m_high":      [50.,200.],
  "mu_g":        [2.,50.,],
  "sigma_g":     [0.4,10.],
  "gamma":       [0., 12.],
  "kappa":       [0., 6.],
  "zp":          [0., 4.]}

# 4. Define log_prior and log_prob_fn functions
def log_flat_prior(params):
  if len(params.shape)>1:
    # vectorized case
    within_bounds = jnp.logical_and(hyperspace[:, 0]<=params, params<=hyperspace[:, 1])
    condition     = jnp.all(within_bounds, axis = 1)
    return jnp.where(condition, 0., -jnp.inf)
  else:
    # non vectorized case
    if all(hyperspace[:, 0] <= params) and all(hyperspace[:, 1] >= params):
      return 0.
    return -jnp.inf

def log_prob_fn(params):

  log_prob = jnp.zeros_like(params)

  max_size = 15 ## to tune depending on the number of events
  for start in range(0, params.size, max_size):
    params_chunk = params[start:start + max_size]

    log_prior_chunk = log_flat_prior(params_chunk)

    to_calc  = jnp.isfinite(log_prior_chunk)
    hp_chunk = eu.generate_dict(params_chunk, params_keys, to_calc)

    log_like_chunk = jnp.zeros_like(log_prior_chunk)
    log_like_chunk = log_like_chunk.at[to_calc].set(jax.vmap(hyperlike)(**hp_chunk))

    log_prob = log_prob.at[start:start + max_size].set(log_like_chunk + log_prior_chunk)

  return log_prob

# 6. Generate IC and filename
trues = {"H0":   70.,
  "lambda_peak": 0.039,
  "alpha":       3.4,
  "beta":        1.1,
  "delta_m":     4.8,
  "m_low":       5.1,
  "m_high":      87.,
  "mu_g":        34.,
  "sigma_g":     3.6,
  "gamma":       2.7,
  "kappa":       3.,
  "zp":          2.}

params_keys = priors.keys()

bests      = jnp.array([trues[x] for x in params_keys])
hyperspace = jnp.array([priors[x] for x in params_keys])
sigmas     = 0.05 * jnp.diff(hyperspace).flatten()

logger.info("Generating initial conditions...")
initial_state = eu.get_initial_state(mcmc_settings['nwalkers'],
  len(params_keys),
  log_flat_prior,
  distribution = 'truncgauss',
  priors = hyperspace,
  gaussian_bests = bests,
  gaussian_sigmas = sigmas,
  restart_chain = mcmc_settings['restart_chain'],
  output_dir = mcmc_settings['output_dir'],
  chain_prefix = mcmc_settings['chain_prefix']
)

filename = eu.generate_chain_filename(mcmc_settings['output_dir'],
  mcmc_settings['chain_prefix'],
  mcmc_settings['restart_chain']
)
backend = emcee.backends.HDFBackend(filename)
backend.reset(mcmc_settings['nwalkers'], len(params_keys))

# 7. Setup ensemble sampler
sampler = emcee.EnsembleSampler(mcmc_settings['nwalkers'],
  len(params_keys),
  log_prob_fn,
  moves     = [emcee.moves.StretchMove()],
  backend   = backend,
  vectorize = True,
)

# 8. Run MCMC
logger.info("Running MCMC...")
sampler.run_mcmc(initial_state,
  mcmc_settings['nsteps'],
  store    = True,
  progress = True,
  progress_kwargs = {'file':sys.stdout}
)
