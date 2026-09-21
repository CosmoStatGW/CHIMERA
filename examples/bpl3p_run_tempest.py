import os, sys
os.environ["CHIMERA_ENABLE_GPU"] = "True"
sys.path.append('/leonardo/home/userexternal/mtagliaz/softwares/CHIMERA/')
from CHIMERA import data
from CHIMERA.data import theta_inj_det
from CHIMERA.cosmo import flrw
from CHIMERA.mass.paired import bpl_dip_three_peaks
from CHIMERA.rate import madau_dickinson
from CHIMERA import population, selection_function, hyperlikelihood
from CHIMERA.utils import emcee_utils as eu

import h5py
import tempest as tp
import numpy as np
import xarray as xr
import arviz as az
import scipy
import jax
import jax.numpy as jnp
import glob
import argparse

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def load_data_from_json(data):
  if  isinstance(data, list):
    return jnp.asarray(data)
  return data

# Set up device visibility - IMPORTANT for multi-GPU

devices = jax.devices('gpu')
print(f"Available devices: {devices}")

parser = argparse.ArgumentParser(description="gwtc5-bpl3p")
parser.add_argument("--n_particles", type=int, required=False, default=512)
parser.add_argument("--n_steps", type=int, required=False, default=None)  # default -> 5
parser.add_argument("--n_total", type=int, required=False, default=8192)
parser.add_argument("--prng_key", type=int, required=False, default=0)
parser.add_argument("--resample", type=str, required=False, default='syst', choices=['mult', 'syst'])
parser.add_argument("--restart", type=str2bool, nargs='?', const=True, default=False)

args = parser.parse_args()

# Setup devices BEFORE any JAX operations
print(f"tempest using {len(devices)} GPU(s)")
if args.restart:
  print("Restarting run")

dir_data = '/leonardo_work/IscrC_MLGW/gwtc5/data/'
dir_chain =  '/leonardo_work/IscrC_MLGW/gwtc5/res/tempest/'

chain_label = f'gwtc5_bpl3p_scaleFree_tempest'

def uniform(low, high):
 return scipy.stats.uniform(loc=low, scale=high-low)

def loguniform(low, high):
  return scipy.stats.loguniform(a=low, b=high)

def norm(loc, scale):
  return scipy.stats.norm(loc=loc, scale=scale)

def truncnorm(loc, scale, low, high):
  return scipy.stats.truncnorm(low, high, loc=loc, scale=scale)

def beta(alpha, beta, low=0., high=1.):
    return scipy.stats.beta(alpha, beta, loc=low, scale=high-low)

# Priors
priors = {"H0": uniform(10.,200.),
  "alpha_1": uniform(-4,12.),
  "alpha_2": uniform(-4,12.),
  "beta_bottom": uniform(-4.,12.),
  "beta_top": uniform(-4.,12.),
  "mu_g1": uniform(5.,150.),       
  "sigma_g1": uniform(0.4,10.),               
  "mu_g2":uniform(5.0, 150.),
  "sigma_g2":uniform(0.4, 15.),
  "mu_g3":uniform(5.0, 150.),
  "sigma_g3":uniform(0.4, 15.),
  "lambda_g":uniform(0,1),
  "lambda_1":beta(alpha=1, beta=2, low=0, high=1),
  "lambda_2":uniform(0,1),
  "bottomsmooth": loguniform(0.01,1.),
  "topsmooth": loguniform(0.001,1.),
  "leftdip": uniform (1.5,3.0),
  "rightdip": uniform(5.0, 9.0),
  "leftdipsmooth": loguniform(0.01,2.0),
  "rightdipsmooth": loguniform(0.01,2.0),
  "deep": uniform(0., 1.),  
  "m_low": uniform(0.4,1.4),
  "m_high": uniform(50.,200.),
  "gamma": uniform(0.,12.),
  "kappa": uniform(0.,6.),
  "zp": uniform(0.,4.)
}

params_keys = list(priors.keys())
n_dim = len(params_keys)
prior_dists = [priors[k] for k in params_keys]

def prior_transform(u):
  return np.array([dist.ppf(u[i]) for i, dist in enumerate(prior_dists)])

n_steps = args.n_steps if args.n_steps is not None else 5

## Load PE data
file_ev = dir_data+'GWTC5_242CBC_FAR0.25_PE2048.h5'
pe_gw = data.load_gw_pe_samples(file_ev, parameters=['m1det', 'm2det', 'dL'], return_struct=True)
pe_prior = jnp.load(dir_data+'GWTC5_242CBC_FAR0.25_PE2048prior.npy')
pe_gw = pe_gw.update(pe_prior=pe_prior)
Nev = len(pe_gw.dL)

## Load injectiond data and defined selection functon
file_inj = dir_data+"GWTC5_injections_far0.25.h5"
f_inj = h5py.File(file_inj)
Ngen = float(f_inj.attrs['Ngen'])
Tobs = float(f_inj.attrs['Tobs'])
theta_inj_det = data.load_injection_data(file_inj, snr_cut=None, key_mapping={'snr':'snr', 'log_pdraw':'log_pdraw'}, frame='detector')

sel_fcn = selection_function(theta_inj_det, N_inj=Ngen)

# Instantiate hyperikelihood object
cosmo = flrw(H0 = 67.9, Om0 = 0.3065, z_max = 5.)
mass = bpl_dip_three_peaks()
rate = madau_dickinson()
population = population(cosmo, mass, rate, scale_free=True, Tobs=Tobs, R0=1)

hyperlike = hyperlikelihood(  # data
  theta_gw_det = pe_gw,
  # population 
  population=population,
  # integration grid resolution
  z_grids_res = 300,  
  # selection function
  selection_function=sel_fcn,
  # numerical stability
  pe_neff = 2.0,
  inj_neff = None, # default to 5*Nev  
  # KDE settings
  kind_kde = 'fft',
  kernel = 'epan',
  kde_bw = None, # default to scott
  #num_bins = 200,
)

# Define loglike fcn
def log_likelihood(params):
  hyperparams = eu.generate_dict(params, params_keys)
  log_like_evs, N_exp, neff_inj, log_like = hyperlike.compute_all(**hyperparams)
  return log_like, log_like_evs, N_exp, neff_inj

# Setup sampler
sampler = tp.Sampler(
  prior_transform=prior_transform,
  log_likelihood=log_likelihood,
  n_dim=n_dim,
  n_particles=args.n_particles,
  n_steps=n_steps,
  resample=args.resample,
  random_state=args.prng_key,
  output_dir=dir_chain,
  output_label=chain_label,
)


# Run sampler
if args.restart:
  state_files = glob.glob(f"{dir_chain}/{chain_label}_*.state")
  latest_state = None
  final_state = f"{dir_chain}/{chain_label}_final.state"
  if os.path.exists(final_state):
    latest_state = final_state
  elif state_files:
    numeric_state_files = [f for f in state_files if not f.endswith('_final.state')]
    if numeric_state_files:
      latest_state = max(numeric_state_files,
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
else:
  latest_state=None

if latest_state is not None:
  print("restarting from", latest_state)

sampler.run(n_total=args.n_total, save_every=1, resume_state_path=latest_state)

# Get posterior samples (importance-weighted; trim outliers) and blobs
samples, weights, logl, blobs = sampler.posterior(resample=True, return_blobs=True)
samples_dict = {k: samples[:, i] for i, k in enumerate(priors)}

log_like_evs_blob = blobs[:,0]
N_exp_blob = blobs[:,1]
neff_inj_blob = blobs[:,2]

logZ, logZerr = sampler.evidence()
logZerr = str('none') if logZerr is None else logZerr

"""
with h5py.File(f"{dir_chain}/{chain_label}_final.h5", 'w') as f:
  for k, v in samples_dict.items():
      f.create_dataset(k, data=v)
  f.create_dataset('weights', data=weights)
  f.create_dataset('log_likelihood', data=logl)
  f.create_dataset('log_like_evs', data=np.stack(log_like_evs_blob))
  f.create_dataset('N_exp', data=np.asarray(N_exp_blob, dtype=np.float64))
  f.create_dataset('Neff_inj', data=np.asarray(neff_inj_blob, dtype=np.float64))
  f.attrs['logZ'] = logZ
  f.attrs['logZerr'] = logZerr
"""

n_chains = 1
n_draws = len(samples)
n_events = np.stack(log_like_evs_blob).shape[-1]
posterior_ds = xr.Dataset(
  {k: (['chain', 'draw'], v.reshape(n_chains, n_draws)) for k, v in samples_dict.items()}
)
sample_stats_ds = xr.Dataset({
  'log_likelihood': (['chain', 'draw'], logl.reshape(n_chains, n_draws)),
  'weights': (['chain', 'draw'], weights.reshape(n_chains, n_draws)),
  'log_like_evs': (['chain', 'draw', 'event'], np.stack(log_like_evs_blob).reshape(n_chains, n_draws, n_events)),
  'N_exp': (['chain', 'draw'], np.asarray(N_exp_blob).reshape(n_chains, n_draws)),
  'Neff_inj': (['chain', 'draw'], np.asarray(neff_inj_blob).reshape(n_chains, n_draws))
})

idata = az.InferenceData(
  posterior=posterior_ds,
  sample_stats=sample_stats_ds
)
idata.attrs['log_evidence'] = float(logZ)
idata.attrs['log_evidence_err'] = logZerr
file = f"{dir_chain}/{chain_label}_final.state"
idata.to_netcdf(f"{file.replace('.state', '.nc')}")

