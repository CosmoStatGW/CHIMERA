from .core.jax_config import jax, jnp, logger
from .likelihood import hyperlikelihood
from .bias import bias

from functools import partial
import numpy as np
import h5py
import os, re, sys
import emcee

from mpi4py import MPI
import mpi4jax

# FUNCTION TO GENERATE A LIKELIHOOD CLASS FOR EACH MPI PROCESS

def instantiate_models(# Data
                       events_pe, 
                       pe_prior, 
                       inj_data, 
                       inj_prior,
                       # Models
                       cosmo_model,
                       mass_model, 
                       rate_model,
                       cat_model,
                       # Bias settings
                       p_bkg          = None,
                       N_inj          = None,
                       z_det_range    = None,
                       z_int_res_bias = 1000,
                       neff_inj_min   = 5,
                       Tobs           = 1, 
                       # Like settings
                       kernel    ='epan', 
                       bw_method = None, 
                       cut_grid  = 2, 
                       binning   = True, 
                       num_bins  = 200,
                       positive_weights_thresh = 5.,  
                       data_Neff    = 5.,
                       cosmo_prior  = None,
                       z_conf_range = 5,
                       z_int_res    = 300
                       ):

  proc_bias = bias(inj_data       = inj_data,
                   inj_prior      = inj_prior,
                   cosmo_model    = cosmo_model, 
                   mass_model     = mass_model,
                   rate_model     = rate_model,
                   p_bkg          = p_bkg,
                   N_inj          = N_inj,
                   z_det_range    = z_det_range,
                   z_int_res_bias = z_int_res_bias,
                   neff_inj_min   = neff_inj_min,
                   Tobs           = Tobs)

  proc_like = hyperlikelihood(events_pe    = events_pe,
                              pe_prior     = pe_prior,
                              cosmo_model  = cosmo_model, 
                              mass_model   = mass_model,
                              rate_model   = rate_model,
                              galcat_model = cat_model, 
                              bias_model   = None, 
                              kernel       = kernel, 
                              bw_method    = bw_method, 
                              cut_grid     = cut_grid, 
                              binning      = binning, 
                              num_bins     = num_bins, 
                              cosmo_prior  = cosmo_prior,
                              z_conf_range = z_conf_range,
                              z_int_res    = z_int_res)

  return proc_like, proc_bias

"""
@partial(jax.jit, static_argnums=(0,1,))
def compute_bias(proc_bias, commhandle, hyper_params=None):

  # PARALLELIZED OVER INJECTION CHUNKS

  comm = MPI.Intracomm.fromhandle(commhandle)
  rank = comm.Get_rank()

  proc_pop_rate = proc_bias.get_rate(hyper_params)

  dNdtheta, _ = mpi4jax.gather(proc_pop_rate, root = 0, comm=comm)

  if rank == 0:

    shape = dNdtheta.shape
    dNdtheta_exp = jnp.expand_dims(dNdtheta, axis=1) if len(shape) == 2 else dNdtheta
    dNdtheta = jnp.concatenate([dNdtheta_exp[i,:,:] for i in range(dNdtheta_exp.shape[0])], axis=-1)

    if dNdtheta.shape[0] == 1:
      dNdtheta = dNdtheta[0]

    xi   = jnp.sum(dNdtheta, axis = -1) / proc_bias.N_inj
    # manually check neff
    s2   = jnp.sum(dNdtheta**2, axis = -1) / proc_bias.N_inj**2  - xi**2 / proc_bias.N_inj
    neff = xi**2 / s2
    if proc_bias.check_Neff:
        neff_cond = jnp.atleast_1d(neff) < proc_bias.neff_inj_min
        xi = jnp.where(neff_cond, 0.0, xi)       
  else:
    xi = jnp.zeros(proc_pop_rate.shape[0])

  xi, _ = mpi4jax.bcast(xi, root = 0)
  return xi
"""

# FUNCTIONS TO COMPUTE THE BIAS AND THE LIKE ACROSS SEVERAL PROCESSES.

def compute_bias(proc_bias, commhandle, **hyper_params):

  # PARALLELIZED OVER THE NUMBER OF HYPERPARAMS, NOT INJECTION CHUNKS
  comm = MPI.Intracomm.fromhandle(commhandle)
  rank = comm.Get_rank()
  size = comm.Get_size()

  if not hyper_params:
    proc_xi = proc_bias.compute()
  else:
    first_value = next(iter(hyper_params.values()))  
    nvec = jnp.atleast_1d(first_value).shape[0]

    nvec_proc = nvec // size  
    remainder = nvec % size   

    start_idx = rank * nvec_proc + min(rank, remainder)
    end_idx   = start_idx + nvec_proc + (1 if rank < remainder else 0)

    proc_hyperparams = {}
    for key, value in hyper_params.items():
      if end_idx - start_idx == 1:
        proc_hyperparams[key] = jnp.atleast_1d(value)[start_idx].astype(float)
      else:
        proc_hyperparams[key] = jnp.atleast_1d(value)[start_idx:end_idx]
  
    proc_xi_local = jnp.zeros(nvec)
    if start_idx < end_idx:  
      proc_xi_local = proc_xi_local.at[start_idx:end_idx].set(proc_bias.compute(**proc_hyperparams))
  
    proc_xi, _ = mpi4jax.allreduce(proc_xi_local, op=MPI.SUM, comm=comm)
  return proc_xi


def compute_log_hyperlike(proc_like, proc_bias, commhandle, **hyper_params):

  loglike_num_partial, neff_ev_partial = proc_like.compute_loglike_num(**hyper_params)
  bias = compute_bias(proc_bias, commhandle, **hyper_params)

  loglike_num, _ = mpi4jax.allreduce(loglike_num_partial, op=MPI.SUM)
  neff_ev, _ = mpi4jax.allreduce(neff_ev_partial, op=MPI.SUM)

  return loglike_num - neff_ev*jnp.log(bias)  

# FUNCTIONS TO GENERATE INITIAL WALKER POSITIONS

def get_initial_state(nwalkers,
                      ndim,
                      log_prior,
                      distribution='gaussian',
                      priors=None,
                      gaussian_bests=None,
                      gaussian_sigmas=None,
                      restart_chain = False,
                      output_dir = None,
                      chain_prefix = None):

  if not restart_chain:
    if priors is None:
      priors = jnp.tile([-jnp.inf, jnp.inf])
    if gaussian_bests is None:
      gaussian_bests = jnp.ones(ndim)
    if gaussian_sigmas is None:
      gaussian_sigmas = jnp.full(ndim, 0.2)
    start = jnp.zeros((nwalkers, ndim))
    if distribution == 'gaussian':
      for i in range(nwalkers):
        tmp = jnp.array(np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(1, ndim)))
        while not check_initials(tmp):
          tmp = jnp.array(np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(1, ndim)))
        start = start.at[i].set(tmp)
        
    elif distribution == 'truncgauss':
      start  = jnp.array(np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(nwalkers, ndim)))
      outside_indices = jnp.logical_or(start < priors[:, 0], start > priors[:, 1])
      for i in range(ndim):
        start = start.at[outside_indices[:, i], i].set(jnp.array(np.random.uniform(low=priors[i, 0], 
                                                                                   high=priors[i, 1], 
                                                                                   size=np.sum(outside_indices[:, i]))))
      
    elif distribution == 'uniform':
      for i in range(nwalkers):
        tmp = jnp.array(np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim)))
        while not check_initials(tmp, log_prior):
          tmp = jnp.array(np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim)))
          while not check_initials(tmp, log_prior):
            tmp = jnp.array(np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim)))
          start = start.at[i].set(tmp)
    else:
      print("Only admitted distributions are 'gaussian', 'uniform', and 'truncgauss'.")
      return None
    
    return start

  else: # start from last point
    try:       
      # get last chain to restart in the directory where 'chain_prefix' is present
      directory = output_dir 
      prefix = chain_prefix
      pattern = rf'{prefix}_(\d+)\.h5$'
      highest_number = float('-inf')  # Initialize with negative infinity to ensure any found number will be greater
      for filename in os.listdir(directory):
          if filename.endswith(".h5"):
              match = re.search(pattern, filename)
              if match:
                  number = int(match.group(1))  # Extract the matched number
                  highest_number = max(highest_number, number)  # Update highest number if necessary
      
      if highest_number == float('-inf'):
          print("No files found matching the pattern.")
                      
      chain_to_restart = directory + prefix + '_' +str(highest_number) + '.h5'
                      
      reader = emcee.backends.HDFBackend(chain_to_restart, read_only=True)
      starting_state = reader.get_last_sample() 
      return starting_state
    except (IOError, KeyError):
        print(f"Error opening file {chr(chain_to_restart)}. Return None.")
        return None  

def check_initials(initial_values, log_prior):
  for i in range(initial_values.shape[0]):
    if log_prior(initial_values[i,:])==-jnp.inf:
      return False
  return True

# FUNCTION TO GENERATE THE FILENAME

def generate_chain_filename(output_dir, chain_prefix, restart_chain):

  directory = output_dir
  prefix = chain_prefix
  pattern = rf'{prefix}_(\d+)\.h5$'
  highest_number = float('-inf')  
  for filename in os.listdir(directory):
    if filename.endswith(".h5"):
      match = re.search(pattern, filename)
      if match:
        number = int(match.group(1))  # Extract the matched number
        highest_number = max(highest_number, number)  # Update highest number if necessary

  if restart_chain:

    if highest_number == float('-inf'):
      raise ValueError("No files found matching the prefix requested.")
    
    chain_to_restart = directory + prefix + '_' + str(highest_number) + '.h5'
    filename = directory + prefix + '_' + str(highest_number+1) + '.h5'

    print(f"Restarting last chain with prefix {prefix}, that is {chain_to_restart}")
    print(f"Samples of the restarded chain written in the file {filename}")

  else:
    filename = directory + prefix + '_0.h5'
    print(f"Writing samples in the file {filename}")     
    
    if highest_number == 0:
      raise ValueError(f"WARNING: there are already some chains with prefix {prefix}. Change prefix or delete {filename} if you want to overwrite it.")

  return filename

# FUNCTION THAT GENERATE A DICT OF ARRAYS GIVEN A NDARRAY AND SOME KEYS

def generate_dict(params, params_keys, to_calc=None):
  if len(params.shape)>1:
    # vectorized case
    if to_calc is None:
      hyperparams = {k:jnp.array(params[:,i]) for i,k in enumerate(params_keys)}
    else:
      hyperparams = {k:jnp.array(params[to_calc,i]) for i,k in enumerate(params_keys)}
  else:
    # non vectorized case
    hyperparams = {k:params[i] for i,k in enumerate(params_keys)}
  return hyperparams
