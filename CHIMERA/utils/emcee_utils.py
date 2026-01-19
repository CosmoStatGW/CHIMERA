"""Utilities for MCMC sampling with emcee.

This module provides helper functions and custom classes for running MCMC chains
with the emcee package, including:
- Chain file management and restart functionality
- Initial walker position generation from various distributions
- Custom sampler classes for specialized parallelization needs
- Parameter dictionary conversion utilities
"""

from .config import jnp, logger
import os, re, sys
from typing import List, Union, Dict, Optional
import numpy as np
import h5py
import warnings

import emcee

def generate_chain_filename(output_dir, chain_prefix, restart_chain):
  """
  Generates the filename for an MCMC chain, either restarting from the last chain or creating a new chain file.
  Args:
    output_dir (str): directory where chain files are stored.
    chain_prefix (str): prefix for the chain filenames. Files are expected to follow the format `<chain_prefix>_<number>.h5`, where `<number>` is an integer.
    restart_chain (bool): if True, restart from the last available chain. Otherwise, create a new chain.
  Returns:
    str: the filename for the chain file to be written.
  """
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

    chain_to_restart = os.path.join(directory, f'{prefix}_{highest_number}.h5')
    filename = os.path.join(directory, f'{prefix}_{highest_number+1}.h5')

    logger.info(f"Restarting last chain with prefix {prefix}, that is {chain_to_restart}")
    logger.info(f"Samples of the restarted chain written in the file {filename}")

  else:
    filename = os.path.join(directory, f'{prefix}_0.h5')
    logger.info(f"Writing samples in the file {filename}")

    if highest_number == 0:
      raise ValueError(f"Chains with prefix {prefix} already exist. Change prefix or delete {filename} to overwrite.")

  return filename

def generate_dict(params, params_keys, to_calc=None):
  """Converts parameter array to dictionary with named keys.
  
  Args:
    params (array-like): Parameter array of shape (n_samples, n_params) or (n_params,)
    params_keys (list[str]): Names for each parameter dimension
    to_calc (array-like, optional): Indices to extract. If None, uses all samples.
  
  Returns:
    dict[str, jnp.ndarray]: Dictionary mapping parameter names to values
  
  Example:
    >>> params = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> generate_dict(params, ['alpha', 'beta'])
    {'alpha': array([1., 3.]), 'beta': array([2., 4.])}
  """
  if len(params.shape) > 1:
    if to_calc is None:
      hyperparams = {k: jnp.array(params[:, i]) for i, k in enumerate(params_keys)}
    else:
      hyperparams = {k: jnp.array(params[to_calc, i]) for i, k in enumerate(params_keys)}
  else:
    hyperparams = {k: params[i] for i, k in enumerate(params_keys)}
  return hyperparams

def get_initial_state(nwalkers,
  ndim,
  log_prior,
  distribution='gaussian',
  priors=None,
  gaussian_bests=None,
  gaussian_sigmas=None,
  restart_chain=False,
  output_dir=None,
  chain_prefix=None
):
  """
  Generates initial walker positions for an emcee sampler.
  Args:
    nwalkers (int): number of walkers in the sampler.
    ndim (int): number of dimensions for each walker.
    log_prior (callable): a function to evaluate the log prior for a given parameter set.
    distribution (str, optional): the distribution to sample initial positions from. Options are  'gaussian', 'uniform', and 'truncgauss'. Defaults to 'gaussian'.
    priors (jnp.ndarray, optional): array of shape (ndim, 2) specifying lower and upper bounds for each dimension. Used for 'uniform' and 'truncgauss' distributions. Defaults to `[-inf, inf]` for all dimensions.
    gaussian_bests (jnp.ndarray, optional): array of shape (ndim,) specifying the mean values for the Gaussian distribution. Defaults to 1 for all dimensions.
    gaussian_sigmas (jnp.ndarray, optional): array of shape (ndim,) specifying the standard deviations for the Gaussian distribution. Defaults to 0.2 for all dimensions.
    restart_chain (bool, optional): if True, initializes the sampler from the last available chain file. Defaults to False.
    output_dir (str, optional): directory containing chain files, required if `restart_chain=True`.
    chain_prefix (str, optional): prefix of chain filenames, required if `restart_chain=True`.
  Returns:
    jnp.ndarray: array of shape (nwalkers, ndim) containing initial positions of the walkers.
  """
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
        while not _check_initials(tmp, log_prior):
          tmp = jnp.array(np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(1, ndim)))
        start = start.at[i].set(tmp)

    elif distribution == 'truncgauss':
      start  = jnp.array(np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(nwalkers, ndim)))
      outside_indices = jnp.logical_or(start < priors[:, 0], start > priors[:, 1])
      for i in range(ndim):
        start = start.at[outside_indices[:, i], i].set(
          jnp.array(np.random.uniform(low=priors[i, 0], high=priors[i, 1], size=np.sum(outside_indices[:, i]))))

    elif distribution == 'uniform':
      for i in range(nwalkers):
        tmp = jnp.array(np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim)))
        while not check_initials(tmp, log_prior):
          tmp = jnp.array(np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim)))
        start = start.at[i].set(tmp)

    else:
      raise ValueError("Only admitted distributions are 'gaussian', 'uniform', and 'truncgauss'.")

    return start

  else: # start from last point

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
      raise ValueError("No files found matching the pattern.")

    chain_to_restart = os.path.join(directory, f'{prefix}_{highest_number}.h5')

    reader = emcee.backends.HDFBackend(chain_to_restart, read_only=True)
    starting_state = reader.get_last_sample()

    return starting_state

def _check_initials(initial_values, log_prior):
  """Validates initial walker positions against prior.
  
  Args:
    initial_values (jnp.ndarray): Proposed initial positions
    log_prior (callable): Log prior function
  
  Returns:
    bool: True if all positions have finite log prior
  """
  for i in range(initial_values.shape[0]):
    if log_prior(initial_values[i, :]) == -jnp.inf:
      return False
  return True

class NotMove(emcee.moves.Move):
  """Custom emcee move for MPI parallelization without state updates.
  
  This move mimics the RedBlueMove structure but sets proposed parameters to -inf,
  allowing them to be overwritten by rank 0 in MPI contexts. Used for specialized
  parallelization schemes where parameter proposals are handled externally.
  
  Args:
    nsplits (int, optional): Number of walker splits. Defaults to 2.
    randomize_split (bool, optional): Randomize split assignments. Defaults to True.
  """

  def __init__(self, nsplits = 2, randomize_split=True):
    self.nsplits = nsplits
    self.randomize_split = randomize_split

  def setup(self, coords):
    pass

  def propose(self, model, state):

    nwalkers, ndim = state.coords.shape

    if nwalkers < 2 * ndim and not self.live_dangerously:
      raise RuntimeError(
        "It is unadvisable to use a red-blue move "
        "with fewer walkers than twice the number of "
        "dimensions."
      )

    self.setup(state.coords)

    accepted = np.zeros(nwalkers, dtype=bool)
    all_inds = np.arange(nwalkers)
    inds = all_inds % self.nsplits
    if self.randomize_split:
      model.random.shuffle(inds)

    for split in range(self.nsplits):
      S1 = inds == split

      sets = [state.coords[inds == j] for j in range(self.nsplits)]
      s = sets[split]
      c = sets[:split] + sets[split + 1 :]

      # params (q) are not obtained using not using "get_proposal" but they will always be a ndarray of zeros.
      # In `log_prob_fn` they will be overwritten by the "master" (rank 0) params
      q       = np.full((len(s), ndim), -np.inf)
      factors = np.full((len(s),), -np.inf)

      # It is important to call "model.compute_log_prob_fn" so that the like can be MPI parallelized.
      new_log_probs, new_blobs = model.compute_log_prob_fn(q)

      # Loop over the walkers and update them accordingly.
      for i, (j, f, nlp) in enumerate(
        zip(all_inds[S1], factors, new_log_probs)
      ):
        lnpdiff = f + nlp - state.log_prob[j]
        if lnpdiff > np.log(model.random.rand()):
          accepted[j] = True

      new_state = emcee.State(q, log_prob=new_log_probs, blobs=new_blobs)
      state = self.update(state, new_state, accepted, S1)

    return state, accepted

class CustomEnsembleSampler(emcee.EnsembleSampler):
  """Modified emcee sampler that skips parameter validity checks.
  
  Extends emcee.EnsembleSampler by removing checks for inf/nan parameter values
  in compute_log_prob. This allows for MPI parallelization schemes where
  parameters may be intentionally set to sentinel values before being
  overwritten by the master process.
  
  All other functionality remains identical to the standard EnsembleSampler.
  See emcee documentation for full parameter descriptions.
  """

  def __init__(
    self,
    nwalkers,
    ndim,
    log_prob_fn,
    pool=None,
    moves=None,
    args=None,
    kwargs=None,
    backend=None,
    vectorize=False,
    blobs_dtype=None,
    parameter_names: Optional[Union[Dict[str, int], List[str]]] = None,
    a=None,
    postargs=None,
    threads=None,
    live_dangerously=None,
    runtime_sortingfn=None,
  ):
    super().__init__(
      nwalkers,
      ndim,
      log_prob_fn,
      pool,
      moves,
      args,
      kwargs,
      backend,
      vectorize,
      blobs_dtype,
      parameter_names,
      # Deprecated...
      a,
      postargs,
      threads,
      live_dangerously,
      runtime_sortingfn
    )

  def compute_log_prob(self, coords):
    """Compute log probability without checking for inf/nan parameters.
    
    Identical to emcee.EnsembleSampler.compute_log_prob but skips the
    validation checks for infinite or NaN parameter values.
    
    Args:
      coords (array-like): Walker coordinates
    
    Returns:
      tuple: (log_prob, blob) arrays
    """
    p = coords

    if self.params_are_named:
      p = emcee.ensemble.ndarray_to_list_of_dicts(p, self.parameter_names)

    if self.vectorize:
      results = self.log_prob_fn(p)
    else:
      if self.pool is not None:
        map_func = self.pool.map
      else:
        map_func = map
      results = list(map_func(self.log_prob_fn, p))

    try:
      blob = [l[1:] for l in results if len(l) > 1]
      if not len(blob):
        raise IndexError
      log_prob = np.array([emcee.ensemble._scalar(l[0]) for l in results])
    except (IndexError, TypeError):
      log_prob = np.array([emcee.ensemble._scalar(l) for l in results])
      blob = None
    else:
      if self.blobs_dtype is not None:
        dt = self.blobs_dtype
      else:
        try:
          with warnings.catch_warnings(record=True):
            warnings.simplefilter(
              "error", VisibleDeprecationWarning
            )
            try:
              dt = np.atleast_1d(blob[0]).dtype
            except Warning:
              warnings.warn(
                "You have provided blobs that are not all the "
                "same shape or size. This means they must be "
                "placed in an object array. Numpy has "
                "deprecated this automatic detection, so "
                "please specify blobs_dtype=np.dtype('object')",
                DeprecationWarning,
                stacklevel=2
              )
              dt = np.dtype("object")
        except ValueError:
          dt = np.dtype("object")
        if dt.kind in "US":
          dt = np.dtype("object")
      blob = np.array(blob, dtype=dt)

      shape = blob.shape[1:]
      if len(shape):
        axes = np.arange(len(shape))[np.array(shape) == 1] + 1
        if len(axes):
          blob = np.squeeze(blob, tuple(axes))

    if np.any(np.isnan(log_prob)):
      raise ValueError("Probability function returned NaN")

    return log_prob, blob
