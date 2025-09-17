from .utils.config import jax, jnp, logger
from .likelihood import hyperlikelihood
from .bias import bias

from mpi4py import MPI
import mpi4jax
import gc

class MPIHyperLike(object):
  r"""A class for MPI-distributed computation of log-likelihood and bias.

  Args:
    comm (MPI.Comm): MPI communicator used for communication between ranks.
    hyperlike_obj (CHIMERA.hyperlikelihood): An instance of the CHIMERA hyperlikelihood object.
    parallelization_scheme (str): Scheme for parallelization. Options are 'params', 'data', or 'both'.
    num_param_batches (int): Number of batches for parallelizing the computation over parameters.

  Class Attributes:
    - rank (int): The rank of the current MPI process.
    - size (int): Total number of MPI processes.
    - main_like (object): The main likelihood object that holds data and settings.
    - main_bias (object): The main bias object used in bias computations.
    - params_keys (list of str): List of parameter keys used for the model.
    - nkeys (int): The number of parameter keys.
    - rank_like (object): The likelihood model assigned to the current rank.
    - rank_bias (object): The bias model assigned to the current rank.
  """
  def __init__(self,
    comm,
    hyperlike_obj,
    params_keys,
    parallelization_scheme='params',
    num_param_batches = None):
    # num_param_batches = how many batches of MPI processes use to parallelize the likelihood over params.
    # The number of MPI process in each batch is determined from the toal size.
    # Process in each batch are used to parallelize the likelihood over params

    self.comm = comm
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    self.main_like = hyperlike_obj
    self.main_bias = self.main_like.bias_obj

    self.params_keys = params_keys
    self.nkeys       = len(self.params_keys)

    self.parallelization_scheme = parallelization_scheme
    self.num_param_batches      = num_param_batches

    self.instantiate_model()

  def instantiate_model(self):

    if self.parallelization_scheme == 'params':
      # repeat like/bias model for each rank
      self.rank_like = self.main_like
      self.rank_bias = self.main_bias

      self.compute_log_likenum  = self._map_log_likenum_over_params
      self.compute_bias = self._map_bias_over_params

    elif self.parallelization_scheme == 'data':

      # divide inj in chunks and initialize new bias model for each rank
      self.tot_inj = self.main_bias.tot_inj

      bias_chunk_size = self.tot_inj // self.size
      bias_remainder  = self.tot_inj % self.size

      start_inj_idx = self.rank * bias_chunk_size + min(self.rank, bias_remainder)
      end_inj_idx   = start_inj_idx + bias_chunk_size + (1 if self.rank < bias_remainder else 0)
      self.local_inj_idx = jnp.arange(start_inj_idx, end_inj_idx)

      local_inj_data  = {k : v[self.local_inj_idx] for k,v in self.main_bias.inj_data.items()}
      local_inj_prior = self.main_bias.inj_prior[self.local_inj_idx]

      self.rank_bias = bias(local_inj_data, local_inj_prior,
        N_inj = self.main_bias.N_inj,
        # model
        cosmo_obj = self.main_bias.cosmo_obj,
        mass_obj  = self.main_bias.mass_obj,
        rate_obj  = self.main_bias.rate_obj,
        p_bkg = self.main_bias.p_bkg,
        # settings
        neff_inj_min   = self.main_bias.neff_inj_min,
        z_det_range    = self.main_bias.z_det_range,
        z_int_res_bias = self.main_bias.z_int_res_bias,
        Tobs           = self.main_bias.Tobs,
        max_z_inj      = self.main_bias.max_z_inj
      )

      # divide events in chunks and initialize new like model for each rank
      like_chunk_size = self.main_like.nevents // self.size
      like_remainder  = self.main_like.nevents % self.size

      start_ev_idx = self.rank * like_chunk_size + min(self.rank, like_remainder)
      end_ev_idx   = start_ev_idx + like_chunk_size + (1 if self.rank < like_remainder else 0)
      local_ev_idx = jnp.arange(start_ev_idx, end_ev_idx)

      local_ev_data  = {k : v[local_ev_idx] for k,v in self.main_like.events_pe.items()}
      local_ev_prior = self.main_like.pe_prior[local_ev_idx]

      self.rank_like = hyperlikelihood(local_ev_data, local_ev_prior,
        # model
        cosmo_obj = self.main_like.cosmo_obj,
        mass_obj  = self.main_like.mass_obj,
        rate_obj  = self.main_like.rate_obj,
        galcat_obj= self.main_like.galcat_obj,
        bias_obj  = None,
        # KDE settings
        kernel    = self.main_like.kernel,
        bw_method = self.main_like.bw_method,
        cut_grid  = self.main_like.cut_grid,
        binning   = self.main_like.binning,
        num_bins  = self.main_like.num_bins,
        # z_grids settings,
        cosmo_prior  = self.main_like.cosmo_prior,
        z_conf_range = self.main_like.z_conf_range,
        z_int_res    = self.main_like.z_int_res
      )

      # free some memory
      del self.main_like
      del self.main_bias
      gc.collect()

      # methods
      self.compute_log_likenum  = self._map_log_likenum_over_events
      self.compute_bias = self._map_bias_over_inj

    elif self.parallelization_scheme == 'both':
      assert type(self.num_param_batches)==int, "`num_param_batches` should be an int."
      assert self.size >= self.num_param_batches, "MPI size should be at least equal to `num_param_batches`."

      # Determine how many and which MPI processes belong to each batch
      # Split them in groups with sub-communicator.
      mean_processes_per_batch = self.size // self.num_param_batches
      remainder = self.size % self.num_param_batches

      process_per_batch = [mean_processes_per_batch + 1 if i < remainder else mean_processes_per_batch for i in range(self.num_param_batches)]
      cumulative_ranks = 0
      self.batch_id = None
      for i, batch_size in enumerate(process_per_batch):
        if cumulative_ranks <= self.rank < cumulative_ranks + batch_size:
            self.batch_id = i
            break
        cumulative_ranks += batch_size

      # Split the communicator based on `batch_id`
      self.batch_comm = self.comm.Split(color=self.batch_id, key=self.rank)

      # Now each batch has its own `batch_comm`
      # Get the rank within the new communicator
      self.rank_in_batch = self.batch_comm.Get_rank()
      self.batch_size = self.batch_comm.Get_size()

      # For each process in each batch, divide injection and generate various bias_obj with a subset of inj
      self.tot_inj = self.main_bias.tot_inj

      bias_chunk_size = self.tot_inj // self.batch_size
      bias_remainder  = self.tot_inj % self.batch_size

      start_inj_idx = self.rank_in_batch * bias_chunk_size + min(self.rank_in_batch, bias_remainder)
      end_inj_idx   = start_inj_idx + bias_chunk_size + (1 if self.rank_in_batch < bias_remainder else 0)
      self.local_inj_idx = jnp.arange(start_inj_idx, end_inj_idx)

      local_inj_data  = {k : v[self.local_inj_idx] for k,v in self.main_bias.inj_data.items()}
      local_inj_prior = self.main_bias.inj_prior[self.local_inj_idx]

      self.rank_in_batch_bias = bias(local_inj_data, local_inj_prior,
        N_inj = self.main_bias.N_inj,
        # model
        cosmo_obj = self.main_bias.cosmo_obj,
        mass_obj  = self.main_bias.mass_obj,
        rate_obj  = self.main_bias.rate_obj,
        p_bkg = self.main_bias.p_bkg,
        # settings
        neff_inj_min   = self.main_bias.neff_inj_min,
        z_det_range    = self.main_bias.z_det_range,
        z_int_res_bias = self.main_bias.z_int_res_bias,
        Tobs           = self.main_bias.Tobs,
        max_z_inj      = self.main_bias.max_z_inj
      )

      # For each process in each batch, divide events and generate various like_obj with a subset of events
      # divide events in chunks and initialize new like model for each rank
      like_chunk_size = self.main_like.nevents // self.batch_size
      like_remainder  = self.main_like.nevents % self.batch_size

      start_ev_idx = self.rank_in_batch * like_chunk_size + min(self.rank_in_batch, like_remainder)
      end_ev_idx   = start_ev_idx + like_chunk_size + (1 if self.rank_in_batch < like_remainder else 0)
      local_ev_idx = jnp.arange(start_ev_idx, end_ev_idx)

      local_ev_data  = {k : v[local_ev_idx] for k,v in self.main_like.events_pe.items()}
      local_ev_prior = self.main_like.pe_prior[local_ev_idx]

      self.rank_in_batch_like = hyperlikelihood(local_ev_data, local_ev_prior,
        # model
        cosmo_obj = self.main_like.cosmo_obj,
        mass_obj  = self.main_like.mass_obj,
        rate_obj  = self.main_like.rate_obj,
        galcat_obj= self.main_like.galcat_obj,
        bias_obj  = None,
        # KDE settings
        kernel    = self.main_like.kernel,
        bw_method = self.main_like.bw_method,
        cut_grid  = self.main_like.cut_grid,
        binning   = self.main_like.binning,
        num_bins  = self.main_like.num_bins,
        # z_grids settings,
        cosmo_prior  = self.main_like.cosmo_prior,
        z_conf_range = self.main_like.z_conf_range,
        z_int_res    = self.main_like.z_int_res
      )

      # free some memory
      del self.main_like
      del self.main_bias
      gc.collect()

      # methods
      self.compute_log_likenum  = self._map_log_likenum_over_both
      self.compute_bias = self._map_bias_over_both

      # check print
      print(f"Global rank {self.rank} is in batch {self.batch_id}, with local rank {self.rank_in_batch} in a group of size {self.batch_size}.\n",
            f"This rank has injection from {start_inj_idx} to {end_inj_idx} for a total of {self.rank_in_batch_bias.tot_inj}.\n",
            f"This rank has events from {start_ev_idx} to {end_ev_idx} for a total of {self.rank_in_batch_like.nevents}.\n")

    else:
      raise ValueError("`parallelization_scheme` can be only 'params', 'data' or 'both'.")

  def compute_log_likelihood(self, **hyper_params):
    """Computes the log hyper-likelihood for the given hyperparameters, using MPI"""
    # Broadcast params from root rank (0) to all other ranks
    if self.rank == 0:
      params = jnp.asarray([hyper_params[k] for k in self.params_keys], dtype=jnp.float64)
    else:
      nparams = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]
      params = jnp.zeros((self.nkeys, nparams), dtype=jnp.float64)
    params, _ = mpi4jax.bcast(params, root=0, comm=self.comm)

    hp = {k: params[i] for i, k in enumerate(self.params_keys)}

    loglike_num, neff_ev = self.compute_log_likenum(**hp)
    bias = self.compute_bias(**hp)

    self.comm.Barrier()
    return loglike_num - neff_ev * jnp.log(bias)

  def __call__(self, **hyper_params):
    """Calls compute_log_likelihood"""
    return self.compute_log_likelihood(**hyper_params)

  ###############################################################

  def _map_bias_over_params(self, **hyper_params):

    nparams  = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]

    local_xi = jnp.zeros(nparams)

    params_per_rank = (nparams + self.size - 1) // self.size
    start_param_idx = self.rank * params_per_rank
    end_param_idx   = min((self.rank + 1) * params_per_rank, nparams)

    local_xi = jnp.zeros(nparams)

    if start_param_idx < end_param_idx:
      local_hp = {k: jnp.atleast_1d(v)[start_param_idx:end_param_idx] for k, v in hyper_params.items()}

      res = jax.vmap(self.rank_bias)(**local_hp)

      local_xi = local_xi.at[start_param_idx:end_param_idx].set(res)

    xi, _ = mpi4jax.allreduce(local_xi, op=MPI.SUM, comm=self.comm)
    return xi

  def _map_bias_over_inj(self, **hyper_params):

    nparams  = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]

    mean_param_per_batch = nparams // self.num_param_batches
    remainder = nparams % self.num_param_batches


    # compute
    batch_dNtheta = jnp.zeros((nparams, self.tot_inj))

    res = jax.vmap(self.rank_bias.compute_pop_rate)(**hyper_params)
    batch_dNtheta = batch_dNtheta.at[:,self.local_inj_idx].set(res)

    dNdtheta, _ = mpi4jax.allreduce(batch_dNtheta, op=MPI.SUM, comm=self.comm)

    xi = jnp.sum(dNdtheta, axis=-1) / self.rank_bias.N_inj

    if self.rank_bias.check_Neff:
      s2 = jnp.sum(dNdtheta**2, axis=-1)/self.rank_bias.N_inj**2 - xi**2/self.rank_bias.N_inj
      neff = xi**2 / s2
      neff_cond = jnp.atleast_1d(neff) < self.rank_bias.neff_inj_min
      xi = jnp.where(neff_cond, 0.0, xi)

    return xi

  def _map_bias_over_both(self, **hyper_params):

    nparams  = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]

    # Distribute params to each batch
    params_per_batch = nparams // self.num_param_batches
    remainder = nparams % self.num_param_batches

    params_per_batch_list = [
        params_per_batch + 1 if i < remainder else params_per_batch
        for i in range(self.num_param_batches)
    ]

    start_param_idx = sum(params_per_batch_list[:self.batch_id])
    end_param_idx = start_param_idx + params_per_batch_list[self.batch_id]

    params_in_batch = {k: jnp.atleast_1d(v)[start_param_idx:end_param_idx] for k, v in hyper_params.items()}

    # check print
    # print(f"Global rank {self.rank} (batch {self.batch_id}) has params {params_in_batch}")

    # print rate for each params in each batch
    loc_dNtheta = jnp.zeros((nparams, self.tot_inj))

    _pres = jax.vmap(self.rank_in_batch_bias.compute_pop_rate)(**params_in_batch)
    loc_dNtheta = loc_dNtheta.at[start_param_idx:end_param_idx, self.local_inj_idx].set(_pres)

    dNdtheta, _ = mpi4jax.allreduce(loc_dNtheta, op=MPI.SUM, comm=self.comm)


    xi = jnp.sum(dNdtheta, axis=-1) / self.rank_in_batch_bias.N_inj

    if self.rank_in_batch_bias.check_Neff:
      s2 = jnp.sum(dNdtheta**2, axis=-1)/self.rank_in_batch_bias.N_inj**2 - xi**2/self.rank_in_batch_bias.N_inj
      neff = xi**2 / s2
      neff_cond = jnp.atleast_1d(neff) < self.rank_in_batch_bias.neff_inj_min
      xi = jnp.where(neff_cond, 0.0, xi)

    return xi

  #################################################################################

  def _map_log_likenum_over_params(self, **hyper_params):

    nparams  = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]

    params_per_rank = (nparams + self.size - 1) // self.size
    start_param_idx = self.rank * params_per_rank
    end_param_idx   = min((self.rank + 1) * params_per_rank, nparams)

    loc_log_num  = jnp.zeros(nparams)
    loc_n_eff_ev = jnp.zeros(nparams)

    if start_param_idx < end_param_idx:
      local_hp = {k: jnp.atleast_1d(v)[start_param_idx:end_param_idx] for k, v in hyper_params.items()}

      loc_res = jax.vmap(self.rank_like.compute_log_likenum)(**local_hp)
      loc_log_num  = loc_log_num.at[start_param_idx:end_param_idx].set(loc_res[0])
      loc_n_eff_ev = loc_n_eff_ev.at[start_param_idx:end_param_idx].set(loc_res[1])

    log_num, _  = mpi4jax.allreduce(loc_log_num, op=MPI.SUM, comm=self.comm)
    n_eff_ev, _ = mpi4jax.allreduce(loc_n_eff_ev, op=MPI.SUM, comm=self.comm)

    return log_num, n_eff_ev

  def _map_log_likenum_over_events(self,  **hyper_params):

    local_log_num, local_n_eff_ev = jax.vmap(self.rank_like.compute_log_likenum)(**hyper_params)

    log_num, _  = mpi4jax.allreduce(local_log_num, op=MPI.SUM, comm=self.comm)
    n_eff_ev, _ = mpi4jax.allreduce(local_n_eff_ev, op=MPI.SUM, comm=self.comm)

    return log_num, n_eff_ev

  def _map_log_likenum_over_both(self, **hyper_params):

    nparams  = jnp.atleast_1d(next(iter(hyper_params.values()))).shape[0]

    # Distribute params to each batch
    params_per_batch = nparams // self.num_param_batches
    remainder = nparams % self.num_param_batches

    params_per_batch_list = [
        params_per_batch + 1 if i < remainder else params_per_batch
        for i in range(self.num_param_batches)
    ]

    start_param_idx = sum(params_per_batch_list[:self.batch_id])
    end_param_idx = start_param_idx + params_per_batch_list[self.batch_id]

    params_in_batch = {k: jnp.atleast_1d(v)[start_param_idx:end_param_idx] for k, v in hyper_params.items()}

    loc_loglikenum = jnp.zeros(nparams)
    loc_neff       = jnp.zeros(nparams)

    _pres0, _pres1 = jax.vmap(self.rank_in_batch_like.compute_log_likenum)(**params_in_batch)

    loc_loglikenum = loc_loglikenum.at[start_param_idx:end_param_idx].set(_pres0)
    loc_neff = loc_neff.at[start_param_idx:end_param_idx].set(_pres1)

    log_num, _  = mpi4jax.allreduce(loc_loglikenum, op=MPI.SUM, comm=self.comm)
    n_eff_ev, _ = mpi4jax.allreduce(loc_neff, op=MPI.SUM, comm=self.comm)

    return log_num, n_eff_ev

#####################################
