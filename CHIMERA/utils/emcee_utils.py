from .config import jnp, logger
import os, re, sys
from typing import List, Union, Dict, Optional
import numpy as np
import h5py

import emcee

# FUNCTION TO GENERATE THE FILENAME OF THE CHAIN

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

# FUNCTIONS TO GENERATE INITIAL WALKER POSITIONS FOR EMCEE SAMPLER

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
        while not _check_initials(tmp):
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

    chain_to_restart = directory + prefix + '_' +str(highest_number) + '.h5'

    reader = emcee.backends.HDFBackend(chain_to_restart, read_only=True)
    starting_state = reader.get_last_sample()

    return starting_state

def _check_initials(initial_values, log_prior):
  for i in range(initial_values.shape[0]):
    if log_prior(initial_values[i,:])==-jnp.inf:
      return False
  return True

# CUSTOM emcee.moves.Move THAT DOES NOT UPDATE THE SRARE

class NotMove(emcee.moves.Move):

  # Same structure of the "RedBlueMove" class, but the new params are not proposed using the "stretch" or "walk" algorithm.
  # New params are always zeros, so that they will pass each check in "compute_log_prob_fn"
  # In the user function "log_prob_fn" such params will be overwritten by rank 0 params.

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

# CUSTOM emcee.EnsembleSampler THAT DOES NOT CHECK IF THE PARAMS ARE "inf" OR "nan".
# Everything else is equal to emcee.EnsembleSampler

class CustomEnsembleSampler(emcee.EnsembleSampler):

  def __init__(self,
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
    # Deprecated...
    a=None,
    postargs=None,
    threads=None,
    live_dangerously=None,
    runtime_sortingfn=None,
  ):
    super().__init__(nwalkers,
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

    # same as in emcee.EnsembleSampler but it does not check if params are "inf" or "nan"

    p = coords

    # Check that the parameters are in physical ranges.
    # ---> removed
    #if np.any(np.isinf(p)):
    #    raise ValueError("At least one parameter value was infinite")
    #if np.any(np.isnan(p)):
    #    raise ValueError("At least one parameter value was NaN")

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
              deprecation_warning(
                "You have provided blobs that are not all the "
                "same shape or size. This means they must be "
                "placed in an object array. Numpy has "
                "deprecated this automatic detection, so "
                "please specify "
                "blobs_dtype=np.dtype('object')"
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
