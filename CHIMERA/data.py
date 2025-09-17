from .utils.config import jax, jnp, logger
from .utils import angles
from .utils.math import jax_gkde_nd
from .utils.io import save_set, load_set, load_data_h5
import numpy as np
import h5py
import healpy as hp
from typing import Optional, Union, List, Dict
import equinox as eqx

###############
# DATA STRUCT #
###############

class theta_generic(eqx.Module):
  def update(self, **kwargs):
    updated = self
    for key, value in kwargs.items():
      updated = eqx.tree_at(
        lambda x: getattr(x, key),
        updated,
        value,
        is_leaf=lambda x: x is None  # Crucial for None handling
      )
    return updated

class theta_pe_det(theta_generic):
  m1det : Optional[jnp.ndarray] = None
  m2det : Optional[jnp.ndarray] = None
  dL : Optional[jnp.ndarray] = None
  phi: Optional[jnp.ndarray] = None
  theta: Optional[jnp.ndarray] = None
  ra: Optional[jnp.ndarray] = None
  dec: Optional[jnp.ndarray] = None
  pe_prior: Optional[jnp.ndarray] = None
  # only for pixelated galaxy catalog
  pixels_pe_all_nsides: Optional[Dict[str, jnp.ndarray]] = None # Healpix index of each PE for all possible nsides. Dict with nsides keys and datssets of shape (Nev, Npe)
  opt_nsides: Optional[jnp.ndarray] = None # Optimal nsides for each event. Array of shape (Nev,)
  pixels_opt_nsides: Optional[jnp.ndarray] = None # Pixels of each event (given the optimal nsides). Padded to have shape (Nev, max_npixels)
  ra_pix: Optional[jnp.ndarray] = None # RA of the pixels of each event. Padded to have shape (Nev, max_npixels)
  dec_pix: Optional[jnp.ndarray] = None # DEC of the pixels of each event. Padded to have shape (Nev, max_npixels)
  gw_loc2d_pdf: Optional[jnp.ndarray] = None # 2D loc pdf in each pixel of each event. Padded to have shape (Nev, max_npixels)
  pixels_pe_opt_nside: Optional[jnp.ndarray] = None # Healpix index of each PE given the optmial nsides. Shape (Nev, Nper)

  def __post_init__(self):
    if self.pe_prior is None and self.dL is not None:
      self.pe_prior = jnp.ones_like(self.dL)

class theta_inj_det(theta_generic):
  m1det : Optional[jnp.ndarray] = None
  m2det : Optional[jnp.ndarray] = None
  dL : Optional[jnp.ndarray] = None
  p_draw: Optional[jnp.ndarray] = None

class theta_src(theta_generic):
  m1src : Optional[jnp.ndarray] = None
  m2src : Optional[jnp.ndarray] = None
  z : Optional[jnp.ndarray] = None
  original_distances: Optional[jnp.ndarray] = None

theta_pe_datasets = ['m1det', 'm2det', 'dL', 'pe_prior']
theta_pe_pixelated_datasets = ['m1det', 'm2det', 'dL', 'pe_prior', 'ra', 'dec', 'theta', 'phi',
  'opt_nsides', 'pixels_opt_nsides', 'ra_pix', 'dec_pix', 'gw_loc2d_pdf', 'pixels_pe_opt_nside']
theta_pe_pixelated_groups = ['pixels_pe_all_nsides']

################
# DATA LOADING #
################

def load_galaxy_catalog(file_path,
                       parameters=['ra_gal', 'dec_gal', 'z_cgal'],
                       units='rad',
                       backend='numpy'):
  """Load galaxy catalog data with optional unit conversion.

  Args:
    file_path: Path to HDF5 file containing galaxy data
    parameters: List of parameters to load (default: ['ra_gal', 'dec_gal', 'z_cgal'])
    units: Output units for angular coordinates ('rad' or 'deg')
    backend: Array backend ('numpy' or 'jax')

  Returns:
    Dictionary with 'ra' (right ascension), 'dec' (declination), 'z' (redshift).
  """
  # Validate inputs
  if units not in ['rad', 'deg']:
    raise ValueError("units must be either 'rad' or 'deg'")

  # Load data with validation
  data = load_data_h5(file_path, backend=backend, require_keys=parameters)

  # Prepare output dictionary with standardized names
  result = {
    'ra': data['ra_gal'],
    'dec': data['dec_gal'],
    'z': data['z_cgal']
  }

  # Convert units if requested
  if units == 'rad':
    xp = jnp if backend == 'jax' else np
    result['ra'] = xp.deg2rad(result['ra'])
    result['dec'] = xp.deg2rad(result['dec'])

  return result

def load_gw_pe_samples(file_ev_pe,
                      parameters=['dL', 'm1det', 'm2det', 'phi', 'theta'],
                      group='posteriors',
                      nevents=None,
                      nsamples=None,
                      return_struct=True):
  """Load GW parameter estimation samples with flexible selection.

  Args:
    file_ev_pe: Path to HDF5 file with PE samples
    parameters: List of parameters to load
    group: HDF5 group containing the data
    nevents: Number/indices of events to select
    nsamples: Number of samples per event to select
    return_struct: Return theta_pe_det struct if True, else dict

  Returns:
    theta_pe_det struct or dictionary of arrays
  """
  # Load and validate data
  data = load_data_h5(file_ev_pe, group_h5=group, require_keys=parameters)

  # Handle event selection
  max_events = data['dL'].shape[0]
  event_idx = _process_selection(nevents, max_events, 'events')

  # Handle sample selection
  max_samples = data['dL'].shape[1]
  sample_idx = _process_selection(nsamples, max_samples, 'samples')

  # Select and convert data
  result = {
    k: jnp.asarray(data[k][event_idx][:, sample_idx])
    for k in parameters
  }

  # Convert angles if available
  if {'theta', 'phi'}.issubset(parameters):
    ra, dec = angles.ra_dec_from_th_phi(result['theta'], result['phi'])
    result.update(ra=ra, dec=dec)

  return theta_pe_det(**result) if return_struct else result

def load_injection_data(file_inj,
                       snr_cut=None,
                       ninj=None,
                       group=None,
                       key_mapping=None,
                       return_struct=True):
  """Load injection data with SNR cut and optional downsampling.

  Args:
    file_inj: Path to injection file
    snr_cut: Minimum SNR threshold
    ninj: Number/indices of injections to select
    group: HDF5 group containing data
    key_mapping: Dictionary for custom key names
    return_struct: Return theta_inj_det if True, else tuple (data, prior)

  Returns:
    theta_inj_det or tuple of (data dictionary, prior array)
  """
  # Default key mapping
  defaults = {
    'm1s': 'm1src',
    'm2s': 'm2src',
    'm1d': 'm1det',
    'm2d': 'm2det',
    'dL': 'dL',
    'z': 'z',
    'snr': 'SNR_net',
    'log_pdraw': 'log_p_draw_nospin'
  }
  keys = {**defaults, **(key_mapping or {})}

  # Check if we have detector-frame or source-frame masses
  use_source_frame = 'm1s' in keys and 'm2s' in keys
  use_detector_frame = 'm1d' in keys and 'm2d' in keys
  required = [keys['m1s'], keys['m2s'], keys['z']] if use_source_frame else [keys['m1d'], keys['m2d']]
  required += [keys[k] for k in ['dL', 'snr', 'log_pdraw']]
  data = load_data_h5(file_inj, group_h5=group, require_keys=required)

  # Apply SNR cut
  keep = data[keys['snr']] > snr_cut if snr_cut else slice(None)

  # Convert to detector frame if needed
  m1d = data.get(keys['m1d'], data[keys['m1s']] * (1 + data[keys['z']]))
  m2d = data.get(keys['m2d'], data[keys['m2s']] * (1 + data[keys['z']]))

  # Validate data
  assert (m1d[keep] > 0).all() and (m2d[keep] > 0).all(), "Masses must be positive"
  assert (data[keys['dL']][keep] > 0).all(), "Distances must be positive"
  assert (m2d[keep] <= m1d[keep]).all(), "Primary mass must be >= secondary mass"

  # Prepare output data
  inj_data = {
    'm1det': m1d[keep],
    'm2det': m2d[keep],
    'dL': data[keys['dL']][keep]
  }

  # Handle injection selection
  max_inj = len(inj_data['m1det'])
  inj_idx = _process_selection(ninj, max_inj, 'injections')

  # Final selection and conversion
  result = {k: jnp.asarray(v[inj_idx]) for k, v in inj_data.items()}
  prior = jnp.exp(data[keys['log_pdraw']][keep][inj_idx])

  return theta_inj_det(**result, p_draw=prior) if return_struct else (result, prior)

# Helper functions
def _process_selection(n, max_n, name):
  if n is None:
    return slice(None)
  elif isinstance(n, (list, np.ndarray)):
    logger.info(f"Selecting specific {name}: {n}")
    return np.asarray(n)
  elif isinstance(n, int):
    if n > max_n:
      logger.warning(f"Requested more {name} than available. Using all {max_n}.")
      return slice(None)
    idx = np.random.choice(max_n, n, replace=False)
    logger.info(f"Randomly selected {n} {name}: {idx}")
    return np.sort(idx)
  else:
    raise ValueError(f"Invalid selection for {name}: must be None, list or int")

################
# PIXELIZATION #
################

def _get_threshold(norm_counts, level):
  prob_sorted     = np.sort(norm_counts)[::-1]
  prob_sorted_cum = np.cumsum(prob_sorted)
  idx      = np.searchsorted(prob_sorted_cum, level) # find index of array which bounds the confidence interval
  mincount = prob_sorted[idx]
  return mincount

def compute_sky_conf_event(healpix_pe, sky_conf, nside):
  """Return all the Healpix pixel indices where the probability of an event is above a given threshold.

  Args:
    healpix_pe (ndarray): Healpix index of PE for a particular events and nside
    event (int): number of the event
    nside (int): nside parameter for Healpix

  Returns:
    Healpix indices of the skymap where the probability of an event is above a given threshold.
  """
  unique, counts = np.unique(healpix_pe, return_counts=True)
  p = np.zeros(hp.nside2npix(nside))
  p[unique] = counts/healpix_pe.shape[0]
  return np.argwhere(p >= _get_threshold(p, sky_conf)).flatten()

def pixelize_gw_catalog(theta_gw,
                       nside_list,
                       mean_npixels_event,
                       sky_conf,
                       nest=False,
                       prefix=None,
                       ret_datastruct=True):
  """Pre-compute columns of corresponding Healpix indices for all the provided nside_list pixelization parameters.

  Args:
    theta_gw: struct with the detector frame PE of the GW catalog
    nside_list: list of nside parameters for Healpix
    mean_npixels_event: approximate number of desired pixels per event
    sky_conf: percentage of the GW are to pixelize
    nest: if the healpy map is nested or not
    prefix: if provided is the prefix of the h5 file to save with the pixelated gw catalog

  Returns:
    Instance of theta_pe_det with all pixelization fields
  """

  num_events = theta_gw.dL.shape[0]
  pixels_pe_all_nsides = {}

  for nside in nside_list:
    logger.info(f"Precomputing Healpix pixels (NSIDE={nside}, NEST={nest})")
    pixels_pe_all_nsides[f"nside_{nside}"] = angles.find_pix_RAdec(theta_gw.ra, theta_gw.dec, nside, nest)

  # Find optimal pixelization
  pixel_count_matrix = jnp.array([
    [len(compute_sky_conf_event(pixels_pe_all_nsides[f"nside_{nside}"][e], sky_conf, nside))
      for nside in nside_list]
    for e in range(num_events)
  ])

  best_nside_indices = jnp.argmin(jnp.abs(pixel_count_matrix - mean_npixels_event), axis=1)
  opt_nsides = jnp.array(nside_list)[best_nside_indices]

  unique_nsides, counts = np.unique(opt_nsides, return_counts=True)
  logger.info(f"Optimal NSIDEs: {unique_nsides}")
  logger.info(f"Event counts: {counts}")

  # Process each event
  event_pixels = [
    compute_sky_conf_event(pixels_pe_all_nsides[f"nside_{opt_nsides[e]}"][e], sky_conf, opt_nsides[e])
    for e in range(num_events)
  ]

  pixel_ra, pixel_dec = zip(*[
    angles.find_ra_dec(event_pixels[e], nside=opt_nsides[e])
    for e in range(num_events)
  ])

  # Process samples
  pixel_probabilities = []
  pe_samples_pixels = np.zeros_like(theta_gw.ra, dtype=jnp.int64)

  for e in range(num_events):
    event_ra_samples = theta_gw.ra[e]
    event_dec_samples = theta_gw.dec[e]
    event_nside = opt_nsides[e]

    # Find closest valid pixel for each sample
    sample_pixel_indices = angles.find_pix_RAdec(event_ra_samples, event_dec_samples, event_nside, nest)
    valid_pixels_mask = np.isin(sample_pixel_indices, event_pixels[e])

    angular_separations = angles.angular_separation_from_LOS(
      event_ra_samples[:, None],
      event_dec_samples[:, None],
      pixel_ra[e][None, :],
      pixel_dec[e][None, :]
    )

    closest_pixel_indices = np.argmin(angular_separations, axis=1)
    pe_samples_pixels[e] = np.where(
      valid_pixels_mask,
      sample_pixel_indices,
      event_pixels[e][closest_pixel_indices]
    )

    # Compute pixel probabilities
    sample_positions = jnp.array([event_ra_samples, event_dec_samples])
    pixel_centers = jnp.array([pixel_ra[e], pixel_dec[e]])
    pixel_probabilities.append(jax_gkde_nd(sample_positions, pixel_centers))

  # Create padded arrays
  padded_event_pixels = _pad_arr_list(event_pixels, pad_value=-100)
  padded_pixel_ra = _pad_arr_list(pixel_ra, pad_value=-100.)
  padded_pixel_dec = _pad_arr_list(pixel_dec, pad_value=-100.)
  padded_pixel_probs = _pad_arr_list(pixel_probabilities, pad_value=-100.)
  pe_samples_pixels = jnp.asarray(pe_samples_pixels)

  # Update struct
  theta_gw_pixelated = theta_gw.update(
    pixels_pe_all_nsides=pixels_pe_all_nsides,
    opt_nsides=opt_nsides,
    pixels_opt_nsides=padded_event_pixels,
    ra_pix=padded_pixel_ra,
    dec_pix=padded_pixel_dec,
    gw_loc2d_pdf=padded_pixel_probs,
    pixels_pe_opt_nside=pe_samples_pixels,
  )

  if prefix is not None:
    # save the pixelated catalog to a .h5 file
    print_list = "-".join(map(str, nside_list))
    fname = prefix + f"_pixelated_nsidelist{print_list}_meanpixels{mean_npixels_event}_skyconf{sky_conf}_nest{nest}.h5"
    save_set(theta_gw_pixelated, fname,
            datasets=theta_pe_pixelated_datasets,
            groups=theta_pe_pixelated_groups)

  if not ret_datastruct:
    return {
      "pe_data": {
        "m1det": theta_gw.m1det,
        "m2det": theta_gw.m2det,
        "dL": theta_gw.dL,
        "pe_prior": theta_gw.pe_prior,
        "ra": theta_gw.ra,
        "dec": theta_gw.dec
      },
      "pixels_pe_all_nsides":all_pix_idx,
      "opt_nsides": nside_events,
      "pixels_opt_nsides": arr_pixels_idx,
      "ra_pix": arr_ra_pix,
      "dec_pix": arr_dec_pix,
      "gw_loc2d_pdf": arr_gw_loc2d_pdf,
      "pixels_pe_opt_nside": pe_pix_idx
    }
  else:
    return theta_gw_pixelated


def load_pixelated_gw_catalog(fname):
  """Load pixelated GW catalog into a theta_pe_det struct"""
  # First load the basic PE data
  theta_gw = theta_pe_det()
  theta_gw = load_set(theta_gw, fname,
                      attrs=[],
                      datasets=theta_pe_pixelated_datasets,
                      groups=theta_pe_pixelated_groups)

  return theta_gw

def _pad_arr_list(array_list, pad_value):
  # useful functon used to save the pixelated catalog
  max_rows = max(arr.shape[0] for arr in array_list)
  max_cols = max(arr.shape[1] for arr in array_list) if array_list[0].ndim > 1 else None
  if max_cols is not None:
    # 2D arrays
    padded = np.full((len(array_list), max_rows, max_cols), pad_value, dtype=array_list[0].dtype)
    for i, arr in enumerate(array_list):
        padded[i, :arr.shape[0], :arr.shape[1]] = arr
  else:
    # 1D arrays
    padded = np.full((len(array_list), max_rows), pad_value, dtype=array_list[0].dtype)
    for i, arr in enumerate(array_list):
        padded[i, :arr.shape[0]] = arr
  return jnp.asarray(padded)

#############################
# COMPUTE LOCALIZATION AREA #
#############################

def compute_localization_areas(theta, phi, percentile=0.9, unit='deg2'):
  """Compute the localization area of each event in the dataset.

  Args:
    theta: Polar angle samples for each event
    phi: Azimuthal angle samples for each event
    percentile: Confidence level for localization area (0-100)
    unit: Output unit for area ('deg2' or 'rad2')

  Returns:
    Array of localization areas for each event
  """
  thetas = np.atleast_2d(theta)
  phis   = np.atleast_2d(phi)
  nev, nsamp = thetas.shape
  area = np.zeros(nev)
  for e in range(nev):
    theta = thetas[e]
    phi = phis[e]
    sigma2theta = np.cov(theta,theta)[0,0]
    sigma2phi   = np.cov(phi,phi)[0,0]
    cov2        = np.cov(theta, phi)[0,1]**2
    _1sigma_area = 2*np.pi*np.abs(np.sin(np.mean(theta)))*np.sqrt(sigma2theta*sigma2phi - cov2)
    area[e] = -np.log(1-percentile/100)*_1sigma_area*(180/np.pi)**2
  return area

def compute_localization_volumes(theta, phi, dL, cosmo_params_min, cosmo_param_max, percentile = 90):
  """Compute the localization volume of each event in the dataset.

  Args:
    theta: Polar angle samples for each event
    phi: Azimuthal angle samples for each event
    dL: Luminosity distance samples for each event
    cosmo_params_min: Cosmological parameters for minimum distance bound
    cosmo_param_max: Cosmological parameters for maximum distance bound
    percentile: Confidence level for localization volume (0-100)

  Returns:
    Array of localization volumes for each event in Gpc^3
  """
  dL = np.atleast_2d(dL)

  from .cosmo import z_from_dGW

  areas  = compute_localization_areas(theta, phi, percentile)/(180/np.pi)**2 # in radiant

  dL_min = np.percentile(dL, (100-percentile)/2, axis = 1) # in Gpc
  dL_max = np.percentile(dL, 100-(100-percentile)/2, axis = 1) # in Gpc

  z_min = flrw.z_from_dGW(cosmo_param_min, dL_min)
  z_max = flrw.z_from_dGW(cosmo_param_max, dL_max)

  V_min = flrw.V_at_z(cosmo_param_min, z_min) # in Gpc^3
  V_max = flrw.V_at_z(cosmo_param_max, z_max) # in Gpc^3

  # we divide the volume shell by the total solid angle factor that is present in V_at_z and  we multiply it by the localization area of the each event
  loc_vols = areas*(V_max - V_min)/(4*np.pi)  # in Gpc^3

  return loc_vols
