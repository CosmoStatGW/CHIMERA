import numpy as np
import healpy as hp
from typing import Optional, Dict
from numbers import Number
import equinox as eqx
from KDExpress import build_hist_edges
import jax
import jax.numpy as jnp
import h5py

from .utils.config import logger
from .utils import angles
from .utils.io import save_set, load_set, load_data_h5

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
  # Pixelization metadata
  max_npixels: Optional[Number] = None  # Number of maximum pixels across all events
  neff_pixels: Optional[jnp.ndarray] = None # Effective number of pixels per event
  pixels_pe_all_nsides: Optional[Dict[str, jnp.ndarray]] = None # Healpix index of each PE for all possible nsides. Dict with nsides keys and datssets of shape (Nev, Npe)
  opt_nsides: Optional[jnp.ndarray] = None # Optimal nsides for each event. Array of shape (Nev,)
  pixels_opt_nsides: Optional[jnp.ndarray] = None # Indexes of the pixels mapping the area of each event given the optimal nside. Padded with NaN to have shape (Nev, max_npixels)
  ra_pix: Optional[jnp.ndarray] = None # RA of the optimal pixels of each event. Padded with NaN to have shape (Nev, max_npixels)
  dec_pix: Optional[jnp.ndarray] = None # DEC of the optimal pixels of each event. Padded with NaN to have shape (Nev, max_npixels)
  gw_loc2d_pdf: Optional[jnp.ndarray] = None # 2D loc pdf in each pixel of each event. Padded with NaN to have shape (Nev, max_npixels)
  pixels_pe_opt_nside: Optional[jnp.ndarray] = None # Healpix index of each PE given the optmial nsides. Shape (Nev, Npe)
  # Additional metadata used for the "full" 3d KDE
  ra_grids: Optional[jnp.ndarray] = None # RA grid centers for each event. Padded with NaN to shape (Nev, max_ra_size)
  dec_grids: Optional[jnp.ndarray] = None # DEC grid centers for each event. Padded with NaN to shape (Nev, max_dec_size)
  ra_edges: Optional[jnp.ndarray] = None # RA grid edges for each event. Padded with NaN to shape (Nev, max_ra_size+1)
  dec_edges: Optional[jnp.ndarray] = None # DEC grid edges for each event. Padded with NaN to shape (Nev, max_dec_size+1)
  pixel_indices: Optional[jnp.ndarray] = None # Grid indices for each pixel. Shape (Nev, max_npixels, 2)

  def __post_init__(self):
    if self.pe_prior is None and self.dL is not None:
      self.pe_prior = jnp.ones_like(self.dL)
    if self.max_npixels is None and self.pixels_opt_nsides is not None:
      self.max_npixels = self.pixels_opt_nsides.shape[1]
    if self.neff_pixels is None and self.pixels_opt_nsides is not None:
      self.neff_pixels = jnp.sum(self.pixels_opt_nsides >= 0, axis=1)

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
theta_pe_pixelated_attrs = ['max_npixels']
theta_pe_pixelated_datasets = ['m1det', 'm2det', 'dL', 'pe_prior', 'ra', 'dec', #'theta', 'phi',
  'opt_nsides', 'pixels_opt_nsides', 'ra_pix', 'dec_pix', 'gw_loc2d_pdf', 'pixels_pe_opt_nside',
  'ra_grids', 'dec_grids', 'ra_edges', 'dec_edges', 'pixel_indices']
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
                       return_struct=True,
                       frame='auto'):  # 'auto', 'source', 'detector'
    """Load injection data with SNR cut and optional downsampling.
    
    Args:
        file_inj: Path to injection file
        snr_cut: Minimum SNR threshold
        ninj: Number/indices of injections to select
        group: HDF5 group containing data
        key_mapping: Dictionary for custom key names
        return_struct: Return theta_inj_det if True, else tuple (data, prior)
        frame: Which frame to use for masses
            - 'auto': Try detector frame first, fall back to source frame
            - 'detector': Use detector frame masses directly
            - 'source': Use source frame masses + redshift
    
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
    
    # Remove any keys mapped to None (explicitly disabled)
    keys = {k: v for k, v in keys.items() if v is not None}
    
    # Check what's available in the file (without loading everything)
    with h5py.File(file_inj, 'r') as f:
        target = f if group is None else f[group]
        available_keys = set(target.keys())
    
    # Determine which frame to use
    has_detector = keys.get('m1d') in available_keys and keys.get('m2d') in available_keys
    has_source = (keys.get('m1s') in available_keys and 
                  keys.get('m2s') in available_keys and 
                  keys.get('z') in available_keys)
    
    if frame == 'detector':
        if not has_detector:
            raise ValueError(f"Detector frame masses not found. Need: {keys['m1d']}, {keys['m2d']}")
        use_detector_frame = True
    elif frame == 'source':
        if not has_source:
            raise ValueError(f"Source frame masses or redshift not found. Need: {keys['m1s']}, {keys['m2s']}, {keys['z']}")
        use_detector_frame = False
    else:  # 'auto'
        use_detector_frame = has_detector
        if not use_detector_frame and not has_source:
            raise ValueError(f"Neither detector nor source frame masses found. "
                           f"Detector keys: {keys.get('m1d')}, {keys.get('m2d')}. "
                           f"Source keys: {keys.get('m1s')}, {keys.get('m2s')}, {keys.get('z')}")
    
    # Build required keys list for load_data_h5
    required = []
    if use_detector_frame:
        required.extend([keys['m1d'], keys['m2d']])
    else:
        required.extend([keys['m1s'], keys['m2s'], keys['z']])
    
    # Add other required keys if they exist in mapping
    for k in ['dL', 'snr', 'log_pdraw']:
        if keys.get(k):
            required.append(keys[k])
    
    # Load data using your existing load_data_h5 function
    data = load_data_h5(file_inj, group_h5=group, backend='jax', require_keys=required)
    
    # Apply SNR cut
    snr_key = keys['snr']
    keep = data[snr_key] > snr_cut if snr_cut else slice(None)
    
    # Get masses in detector frame
    if use_detector_frame:
        m1d = data[keys['m1d']]
        m2d = data[keys['m2d']]
    else:
        z = data[keys['z']]
        m1d = data[keys['m1s']] * (1 + z)
        m2d = data[keys['m2s']] * (1 + z)
    
    # Apply SNR cut to arrays
    m1d = m1d[keep]
    m2d = m2d[keep]
    dL = data[keys['dL']][keep]
    
    # Validate data
    assert (m1d > 0).all() and (m2d > 0).all(), "Masses must be positive"
    assert (dL > 0).all(), "Distances must be positive"
    
    # Ensure m1 >= m2 (swap if needed)
    swap_mask = m2d > m1d
    if swap_mask.any():
        m1d_swapped = jnp.where(swap_mask, m2d, m1d)
        m2d_swapped = jnp.where(swap_mask, m1d, m2d)
        m1d = m1d_swapped
        m2d = m2d_swapped
        print(f"Warning: Swapped {swap_mask.sum()} injections to ensure m1 >= m2")
    
    # Prepare output data
    inj_data = {
        'm1det': m1d,
        'm2det': m2d,
        'dL': dL
    }
    
    # Handle injection selection
    max_inj = len(inj_data['m1det'])
    inj_idx = _process_selection(ninj, max_inj, 'injections')
    
    # Final selection
    result = {k: jnp.asarray(v[inj_idx]) for k, v in inj_data.items()}
    
    # Handle prior (with fallback options)
    prior_key = keys.get('log_pdraw')
    if prior_key and prior_key in data:
        prior = jnp.exp(data[prior_key][keep][inj_idx])
    else:
        # Try alternative keys if the specified one isn't found
        alt_keys = ['log_p_draw', 'log_pdraw', 'log_p_draw_nospin']
        found = False
        for alt in alt_keys:
            if alt in data:
                prior = jnp.exp(data[alt][keep][inj_idx])
                found = True
                print(f"Warning: Using '{alt}' as prior key (expected '{prior_key}')")
                break
        if not found:
            # Default to uniform prior
            prior = jnp.ones(len(inj_data['m1det'])[inj_idx] if isinstance(inj_idx, slice) else len(inj_idx))
            print("Warning: No log_p_draw found, using uniform prior")
    
    # Return as struct or tuple
    if return_struct:
        # Assuming theta_inj_det is a dataclass or NamedTuple
        return theta_inj_det(**result, p_draw=prior)
    else:
        return result, prior


def _process_selection(ninj, max_inj, name):
    """Process injection selection indices.
    
    Args:
        ninj: None, int, slice, or array-like of indices
        max_inj: Maximum number of injections available
        name: Name for error messages
    
    Returns:
        slice or array of indices
    """
    if ninj is None:
        return slice(None)
    elif isinstance(ninj, int):
        if ninj > max_inj:
            raise ValueError(f"Cannot select {ninj} {name}, only {max_inj} available")
        return slice(ninj)
    elif isinstance(ninj, slice):
        # Validate slice bounds
        start = ninj.start or 0
        stop = min(ninj.stop, max_inj) if ninj.stop else max_inj
        if start >= max_inj:
            raise ValueError(f"Slice start {start} out of range for {name}")
        return slice(start, stop, ninj.step)
    elif isinstance(ninj, (list, np.ndarray, jnp.ndarray)):
        if len(ninj) > 0 and max(ninj) >= max_inj:
            raise ValueError(f"Index {max(ninj)} out of range for {name}")
        return ninj
    else:
        raise TypeError(f"ninj must be None, int, slice, or array-like, got {type(ninj)}")

################
# PIXELIZATION #
################

def _get_threshold(norm_counts, level):
  prob_sorted     = np.sort(norm_counts)[::-1]
  prob_sorted_cum = np.cumsum(prob_sorted)
  idx      = np.searchsorted(prob_sorted_cum, level) # find index of array which bounds the confidence interval
  mincount = prob_sorted[idx]
  return mincount

def _compute_sky_conf_event(healpix_pe, sky_conf, nside):
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

  # Precompute all nside pixelizations first
  for nside in nside_list:
    print(f"Precomputing Healpix pixels (NSIDE={nside}, NEST={nest})")
    pixels_pe_all_nsides[f"nside_{nside}"] = jnp.asarray(angles.find_pix_RAdec(theta_gw.ra, theta_gw.dec, nside, nest))

  # Find optimal pixelization
  pixel_count_matrix = np.zeros((num_events, len(nside_list)))
  for e in range(num_events):
    for n_idx, nside in enumerate(nside_list):
      event_pixels = pixels_pe_all_nsides[f"nside_{nside}"][e]
      pixel_count_matrix[e, n_idx] = len(_compute_sky_conf_event(event_pixels, sky_conf, nside))

  best_nside_indices = np.argmin(np.abs(pixel_count_matrix - mean_npixels_event), axis=1)
  opt_nsides = np.array(nside_list)[best_nside_indices]

  unique_nsides, counts = np.unique(opt_nsides, return_counts=True)
  print(f"Optimal NSIDEs: {unique_nsides}")
  print(f"Event counts: {counts}")

  # Initialize arrays for all events
  event_pixels = [None] * num_events
  pixel_ra = [None] * num_events
  pixel_dec = [None] * num_events
  pixel_probabilities = [None] * num_events
  pe_samples_pixels = np.zeros_like(theta_gw.ra, dtype=np.int64)

  # Initialize pixels metadata arrays
  max_npixels = 0
  max_ra_size = 0
  max_dec_size = 0

  # First pass: find maximum sizes for grid metadata
  for e in range(num_events):
    event_nside = opt_nsides[e]
    event_pixels[e] = _compute_sky_conf_event(pixels_pe_all_nsides[f"nside_{event_nside}"][e], sky_conf, event_nside)
    max_npixels = max(max_npixels, len(event_pixels[e]))

    # Get RA/DEC for pixels to determine grid sizes
    pixel_ra[e], pixel_dec[e] = angles.find_ra_dec(event_pixels[e], nside=event_nside)
    max_ra_size = max(max_ra_size, len(np.unique(pixel_ra[e])))
    max_dec_size = max(max_dec_size, len(np.unique(pixel_dec[e])))

  # Initialize grid metadata arrays with padding
  ra_grids = np.full((num_events, max_ra_size), np.nan)
  dec_grids = np.full((num_events, max_dec_size), np.nan)
  ra_edges = np.full((num_events, max_ra_size + 1), np.nan)
  dec_edges = np.full((num_events, max_dec_size + 1), np.nan)
  pixel_indices = np.full((num_events, max_npixels, 2), -1, dtype=np.float_)

  # Second pass: process all events and populate grid metadata
  for e in range(num_events):
    event_nside = opt_nsides[e]

    # Process samples for this event
    event_ra_samples = theta_gw.ra[e]
    event_dec_samples = theta_gw.dec[e]

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
    sample_positions = np.array([event_ra_samples, event_dec_samples])
    pixel_centers = np.array([pixel_ra[e], pixel_dec[e]])
    pixel_probabilities[e] = jax.scipy.stats.gaussian_kde(sample_positions)(pixel_centers)

    # Setup grid metadata for this event
    npix = len(event_pixels[e])
    if npix > 0:
      # Create unique sorted grids
      ra_unique = np.sort(np.unique(pixel_ra[e]))
      dec_unique = np.sort(np.unique(pixel_dec[e]))

      ra_size = len(ra_unique)
      dec_size = len(dec_unique)

      # Store grids
      ra_grids[e, :ra_size] = ra_unique
      dec_grids[e, :dec_size] = dec_unique

      # Create edges
      ra_edges_ev = build_hist_edges(ra_unique)
      dec_edges_ev = build_hist_edges(dec_unique)
      ra_edges[e, :ra_size+1] = ra_edges_ev
      dec_edges[e, :dec_size+1] = dec_edges_ev

      # Create pixel to grid index mapping
      indices = np.full((max_npixels, 2), -1, dtype=np.float_)
      for i in range(npix):
        ra_val, dec_val = pixel_ra[e][i], pixel_dec[e][i]
        ra_idx = np.argmin(np.abs(ra_unique - ra_val))
        dec_idx = np.argmin(np.abs(dec_unique - dec_val))
        indices[i] = np.array([ra_idx, dec_idx])
      pixel_indices[e] = indices

  # Create padded arrays for the main pixel data
  padded_event_pixels = _pad_arr_list(event_pixels, pad_value=np.nan, dtype=np.float_)
  padded_pixel_ra = _pad_arr_list(pixel_ra, pad_value=np.nan)
  padded_pixel_dec = _pad_arr_list(pixel_dec, pad_value=np.nan)
  padded_pixel_probs = _pad_arr_list(pixel_probabilities, pad_value=np.nan)

  # Calculate neff_pixels
  neff_pixels = np.sum(padded_event_pixels >= 0, axis=1)

  # Update struct with all pixelization and grid metadata
  theta_gw_pixelated = theta_gw.update(
    max_npixels=max_npixels,
    neff_pixels=jnp.asarray(neff_pixels),
    pixels_pe_all_nsides=pixels_pe_all_nsides,
    opt_nsides=jnp.asarray(opt_nsides),
    pixels_opt_nsides=padded_event_pixels,
    ra_pix=padded_pixel_ra,
    dec_pix=padded_pixel_dec,
    gw_loc2d_pdf=padded_pixel_probs,
    pixels_pe_opt_nside=jnp.asarray(pe_samples_pixels).astype(jnp.float_),
    ra_grids=jnp.asarray(ra_grids),
    dec_grids=jnp.asarray(dec_grids),
    ra_edges=jnp.asarray(ra_edges),
    dec_edges=jnp.asarray(dec_edges),
    pixel_indices=jnp.asarray(pixel_indices)
  )

  if prefix is not None:
    # save the pixelated catalog to a .h5 file
    print_list = "-".join(map(str, nside_list))
    fname = prefix + f"_pixelated_nsidelist{print_list}_meanpixels{mean_npixels_event}_skyconf{sky_conf}_nest{nest}.h5"
    save_set(theta_gw_pixelated, fname,
            attrs=theta_pe_pixelated_attrs,
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
      "max_npixels": theta_gw_pixelated.max_npixels,
      "neff_pixels": theta_gw_pixelated.neff_pixels,
      "pixels_pe_all_nsides": theta_gw_pixelated.pixels_pe_all_nsides,
      "opt_nsides": theta_gw_pixelated.opt_nsides,
      "pixels_opt_nsides": theta_gw_pixelated.padded_event_pixels,
      "ra_pix": theta_gw_pixelated.padded_pixel_ra,
      "dec_pix": theta_gw_pixelated.padded_pixel_dec,
      "gw_loc2d_pdf": theta_gw_pixelated.padded_pixel_probs,
      "pixels_pe_opt_nside": theta_gw_pixelated.pe_samples_pixels,
      "grid_metadata": {
        "ra_grids": theta_gw_pixelated.ra_grids,
        "dec_grids": theta_gw_pixelated.dec_grids,
        "ra_edges": theta_gw_pixelated.ra_edges,
        "dec_edges": theta_gw_pixelated.dec_edges,
        "pixel_indices": theta_gw_pixelated.pixel_indices
      }
    }
  else:
    return theta_gw_pixelated

def load_pixelated_gw_catalog(fname):
  """Load pixelated GW catalog into a theta_pe_det struct"""
  # First load the basic PE data
  return load_set(theta_pe_det(), fname,
                  attrs=theta_pe_pixelated_attrs,
                  datasets=theta_pe_pixelated_datasets,
                  groups=theta_pe_pixelated_groups)

def _pad_arr_list(array_list, pad_value, dtype=None):
  dtype = array_list[0].dtype if dtype is None else dtype
  # useful functon used to save the pixelated catalog
  max_rows = max(arr.shape[0] for arr in array_list)
  max_cols = max(arr.shape[1] for arr in array_list) if array_list[0].ndim > 1 else None
  if max_cols is not None:
    # 2D arrays
    padded = np.full((len(array_list), max_rows, max_cols), pad_value, dtype=dtype)
    for i, arr in enumerate(array_list):
        padded[i, :arr.shape[0], :arr.shape[1]] = arr
  else:
    # 1D arrays
    padded = np.full((len(array_list), max_rows), pad_value, dtype=dtype)
    for i, arr in enumerate(array_list):
        padded[i, :arr.shape[0]] = arr
  return jnp.asarray(padded)
#############################
# COMPUTE LOCALIZATION AREA #
#############################

def compute_localization_areas(theta, phi, percentile=90, unit='deg2'):
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

