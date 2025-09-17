from functools import partial
from .config import jax, jnp, xp, USE_GPU
import numba as nb
import numpy as np

if USE_GPU:
  from cupyx.scipy.spatial.distance import cdist
  from cupyx.scipy.special import logsumexp

#####################
# get_neff function #
#####################

@partial(jax.jit, static_argnames=['Ndraw'])
def get_neff(weights, Ndraw=None):
  nsamples = weights.shape[-1]
  weights /= jnp.sum(weights)
  mu = jnp.mean(weights)
  if Ndraw is None:
    Ndraw = nsamples
  s2   = jnp.sum(weights**2, axis=-1) / Ndraw**2
  sig2 = s2 - mu**2 / Ndraw
  return jax.lax.cond(jnp.abs(sig2) <= 1.e-15, lambda _ : 0.0, lambda _ : mu**2 / sig2, operand=None)

##############
# 1d binning #
##############

@partial(jax.jit, static_argnames=['num_bins'])
def binning1d(dataset, weights, num_bins=200):
  # Determine the bin edges
  min_val = jnp.min(dataset)
  max_val = jnp.max(dataset)
  bin_edges = jnp.linspace(min_val, max_val, num_bins + 1)
  # Calculate the bin centers
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  # Compute bin indices for each data point
  bin_indices = jnp.clip(jnp.floor((dataset - min_val) / (max_val - min_val) * num_bins), 0, num_bins - 1).astype(int)
  # Initialize bin counts array
  bin_counts = jnp.zeros(num_bins)
  # Use scatter-add to accumulate weights into corresponding bins
  bin_counts = bin_counts.at[bin_indices].add(weights)
  return bin_centers, bin_counts

####################################################################
# 1d KDE with Epanechnikov or Gaussian kernel and grid_cut option. #
####################################################################

@partial(jax.jit, static_argnames=['kernel', 'bw_method', 'cut_grid'])
def kde1d(dataset,
  grid,
  weights=None,
  kernel='epan',
  bw_method=None,
  cut_grid=2):
  # Normalize weights
  if weights is None:
    weights = jnp.ones_like(dataset) / dataset.size
  else:
    weights = weights / jnp.sum(weights)
  # Bw selection
  neff = 1.0 / jnp.sum(jnp.power(weights, 2))
  if bw_method == "scott" or bw_method is None:
    bandwidth = jnp.power(neff, -1. / (1 + 4))
    bandwidth *= jnp.std(dataset)
  elif bw_method == "silverman":
    bandwidth = jnp.power(neff * (1 + 2) / 4.0, -1. / (1 + 4))
    bandwidth *= jnp.std(dataset)
  elif jnp.isscalar(bw_method) and not isinstance(bw_method, str):
    bandwidth = bw_method
    bandwidth *= jnp.std(dataset)
  else:
    raise ValueError("bw_method should be 'scott', 'silverman', or a scalar")
  # Set up evaluation grid
  data_min, data_max, sigma = jnp.min(dataset), jnp.max(dataset), jnp.std(dataset)
  lb = jnp.where( data_min - cut_grid*sigma > 0.,  data_min - cut_grid*sigma, 1.e-6)
  ub = data_max + cut_grid*sigma
  _grid = jnp.linspace(lb, ub, grid.size // 2)
  # Choose kernel and compute kernel values
  kernel_fn   = _epan_kernel if kernel == 'epan' else _gaussian_kernel
  kernel_vals = kernel_fn((_grid[:,None] - dataset) / bandwidth)
  # Calculate density
  density = jnp.sum(weights * kernel_vals, axis=-1) / bandwidth
  # Interpolate density at evaluation_gird
  return jnp.interp(grid, _grid, density, left = 0., right = 0.)

@jax.jit
def _epan_kernel(u):
  return jnp.where(jnp.abs(u) <= 1, 3/4 * (1 - u**2), 0)

@jax.jit
def _gaussian_kernel(u):
  return jnp.exp(-0.5 * u**2) / jnp.sqrt(2 * jnp.pi)

########################################
# n-dimensional gaussian kde using jax #
########################################

def jax_gkde_nd(dataset, evaluation_grid, weights=None, bw_method=None, in_log=False):
  # Same as jax.scipy.stats.gaussian_kde, but simple function and not a class. Not vectorized.
  dataset = jnp.atleast_2d(dataset)
  d_dataset, n_dataset = dataset.shape
  points = jnp.atleast_2d(evaluation_grid)
  d_points, n_points = points.shape
  if d_points != d_dataset:
    if d_points == 1 and n_points == d_dataset:
      points = points.T
      n_points = points.shape[1]
    else:
      msg = "points have dimension " + str(d_points) + ", dataset has dimension " + str(d_dataset)
      raise ValueError(msg)
  if weights is not None:
    if weights.ndim != 1:
      raise ValueError("`weights` input should be one-dimensional.")
    if len(weights) != n_dataset:
      raise ValueError("`weights` input should be of length n_dataset")
    _weights = weights /jnp.sum(weights)
  else:
    _weights = jnp.full(n_dataset, 1.0 / n_dataset, dtype=dataset.dtype)
  neff = 1.0 / jnp.sum(jnp.power(_weights,2))
  if bw_method == "scott" or bw_method is None:
    factor = jnp.power(neff, -1. / (d_dataset + 4))
  elif bw_method == "silverman":
    factor = jnp.power(neff * (d_dataset + 2) / 4.0, -1. / (d_dataset + 4))
  elif jnp.isscalar(bw_method) and not isinstance(bw_method, str):
    factor = bw_method
  else:
    raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar")
  _mean            = jnp.sum(_weights * dataset, axis=1)
  _residual        = (dataset - _mean[:, None])
  _data_covariance = jnp.atleast_2d(jnp.dot(_residual * _weights, _residual.T))
  _data_covariance /= (1 - jnp.sum(_weights ** 2))
  _data_inv_cov    = jnp.linalg.inv(_data_covariance)
  covariance  = _data_covariance * factor**2
  inv_cov     = _data_inv_cov / factor**2
  whitening        = jnp.linalg.cholesky(inv_cov)
  points_whitened  = jnp.dot(points.T, whitening)
  dataset_whitened = jnp.dot(dataset.T, whitening)
  log_norm = jnp.sum(jnp.log(jnp.diag(whitening))) - 0.5 * d_dataset * jnp.log(2 * jnp.pi)
  to_ret = jax_gaussian_kernel_nd(in_log, points_whitened, dataset_whitened, _weights, log_norm)
  return to_ret

@partial(jax.jit, static_argnums=0)
def jax_gaussian_kernel_nd(in_log, points, dataset, weights, log_norm):
  def _kernel(x_test, x_train, y_train):
    arg = log_norm - 0.5 * jnp.sum(jnp.square(x_train - x_test))
    return jnp.log(y_train) + arg if in_log else y_train * jnp.exp(arg)
  def _reduced_kernel(x):
    kernel_values = jax.vmap(_kernel, in_axes=(None, 0, 0))(x, dataset, weights)
    return special.logsumexp(kernel_values) if in_log else jnp.sum(kernel_values)
  mapped_kernel = jax.vmap(_reduced_kernel)
  return mapped_kernel(points)

###############################################
# n-dimensional gaussian kde using numba/cupy #
###############################################

def numba_gkde_nd(dataset, evaluation_grid, weights=None, bw_method=None, in_log=False):
  # cast dataset and points
  dataset = xp.atleast_2d(dataset)
  d_dataset, n_dataset = dataset.shape
  points = xp.atleast_2d(evaluation_grid)
  d_points, n_points = points.shape
  if d_points != d_dataset:
    if d_points == 1 and n_points == d_dataset:
      points = points.T
      n_points = points.shape[1]
    else:
      msg = "points have dimension " + str(d_points) + ", dataset has dimension " + str(d_dataset)
      raise ValueError(msg)
  # normalize weights
  if weights is not None:
    if weights.ndim != 1:
      raise ValueError("`weights` input should be one-dimensional.")
    if len(weights) != n_dataset:
      raise ValueError("`weights` input should be of length n_dataset")
    _weights = weights / np.sum(weights)
  else:
    _weights = xp.full(n_dataset, 1.0 / n_dataset, dtype=dataset.dtype)
  # select bw
  neff = 1.0 / xp.sum(xp.power(_weights,2))
  if bw_method == "scott" or bw_method is None:
    factor = xp.power(neff, -1. / (d_dataset + 4))
  elif bw_method == "silverman":
    factor = xp.power(neff * (d_dataset + 2) / 4.0, -1. / (d_dataset + 4))
  elif np.isscalar(bw_method) and not isinstance(bw_method, str):
    factor = bw_method
  else:
    raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar")
  # data cov
  _mean     = xp.sum(_weights * dataset, axis=1)
  _residual = (dataset - _mean[:, None])
  _data_covariance = xp.atleast_2d(xp.dot(_residual * _weights, _residual.T))
  _data_covariance /= (1 - xp.sum(_weights ** 2))
  _data_inv_cov = xp.linalg.inv(_data_covariance)
  covariance  = _data_covariance * factor**2
  inv_cov     = _data_inv_cov / factor**2
  # whitening
  whitening        = xp.linalg.cholesky(inv_cov)
  points_whitened  = xp.dot(xp.ascontiguousarray(points.T), whitening)
  dataset_whitened = xp.dot(xp.ascontiguousarray(dataset.T), whitening)
  # main computation
  if USE_GPU:
    norm_factor = xp.sqrt(xp.linalg.det(2*xp.pi*covariance))
    chi2 = cdist(points_whitened, dataset_whitened, 'euclidean') ** 2
    if in_log:
      log_res = -0.5 * chi2 + xp.log(_weights) - xp.log(norm_factor)
      to_ret  = logsumexp(log_res, axis=1)
    else:
      to_ret  = xp.sum(xp.exp(-0.5 * chi2) * _weights, axis=1) / norm_factor
  else: 
    to_ret = numba_gaussian_kernel(in_log, points_whitened, dataset_whitened, _weights, whitening)
  return to_ret

@nb.njit(parallel=True)
def numba_gaussian_kernel(in_log, points_whitened, dataset_whitened, weights, whitening):
  n_dataset, d_dataset = dataset_whitened.shape
  n_points, d_points = points_whitened.shape
  log_norm = np.sum(np.log(np.diag(whitening))) - 0.5 * d_dataset * np.log(2 * np.pi)

  results = np.zeros(points_whitened.shape[0])
  for i in nb.prange(n_points):
    log_sum = -np.inf
    for j in nb.prange(n_dataset):
      log_arg = log_norm - 0.5 * np.sum((dataset_whitened[j] - points_whitened[i]) ** 2)
      if in_log:
        log_sum = np.logaddexp(log_sum, np.log(weights[j]) + log_arg)
      else:
        results[i] += weights[j] * np.exp(log_arg)
    if in_log:
      results[i] = log_sum

  return results
