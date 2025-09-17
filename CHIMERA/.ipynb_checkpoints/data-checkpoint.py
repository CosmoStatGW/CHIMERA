from .utils.config import jax, jnp, logger
from .utils import angles
from .utils.kde import jax_gkde_nd
import numpy as np
import h5py
import healpy as hp

################
# DATA LOADING #
################

def load_data_h5(fname, group_h5=None, backend='jax', require_keys=False):
  """Generic function to load data from h5 files"""
  if backend == 'jax':
    xp = jnp
  elif backend == 'numpy':
    xp = np
  
  def _check_keys(f, fname, require_keys):
    missing_keys = [k for k in require_keys if k not in list(f.keys())]
    if len(missing_keys)>0:
      raise ValueError(f"{fname} does not contain the required keys {missing_keys}")

  events={}
  if group_h5 is None:
    with h5py.File(fname, 'r') as f:
      if require_keys:
        _check_keys(f, fname, require_keys)
      for key in f.keys():
        events[key] = xp.array(f[key][:])
    return events
  else:
    with h5py.File(fname, 'r') as f:
      if require_keys:
        _check_keys(f, fname, require_keys)
      for key in f[group_h5].keys():
        events[key] = xp.array(f[group_h5][key][:])
    return events

def load_data_gal(file_data_gal, keys=['ra', 'dec', 'z'], add_keys=None, units='rad', backend='numpy'):

  """Function to load MICE data"""
  if backend == 'jax':
    xp = jnp
  elif backend == 'numpy':
    xp = np

  keys += [add_keys] if isinstance(add_keys, str) else add_keys or []

  all_data = load_data_h5(file_data_gal, backend=backend, require_keys=keys)
  data_gal = {k:all_data[k][:] for k in keys}

  # TO BE REMOVED, else POTENTIAL ISSUES WHEN CALLED BY OTHER MODULES
  if units=='deg':
    data_gal['ra']  = xp.deg2rad(data_gal['ra'])
    data_gal['dec'] = xp.deg2rad(data_gal['dec'])

  return data_gal

def load_data_gw(file_ev_pe, keys_to_load=['dL', 'm1det', 'm2det', 'phi', 'theta'], group_h5= 'posteriors', nevents = None, nsamples = None):
  """Function to load GW PE"""
  all_data      = load_data_h5(file_ev_pe, group_h5=group_h5)
  missing_keys  = [key for key in keys_to_load if key not in all_data]
  if missing_keys:
    raise ValueError(f"{file_ev_pe} does not contain the required keys {missing_keys}")

  # asusming the h5 file contains data in the format (nevents, nsamples)
  max_nev     = all_data['dL'].shape[0]
  max_samples = all_data['dL'].shape[1]

  if nevents:
    if type(nevents)==list:
      mask_ev = jnp.array(nevents)
      logger.info('The analysys will be performed on the events: ', mask_ev)
    elif isinstance(nevents, float)  or isinstance(nevents,  int):
      if nevents > max_nev:
        logger.warning('More  events requested than available in the catalog. Use all the catalog.')
        mask_ev = jnp.arange(max_nev)
      else:
        mask_ev  = jnp.array(sorted(np.random.choice(max_nev, nevents, replace=False)))
        logger.info('The analysys will be performed on the randomly chosen events: ', mask_ev)
    else:
      raise ValueError('`nevents` can only be `None`, an int or a list of int.')
  else:
    mask_ev = jnp.arange(max_nev)

  if nsamples:
    mask_sample = jnp.array(sorted(np.random.choice(max_samples, nsamples, replace=False)))
  else:
    mask_sample = jnp.arange(max_samples)

  ev_pe     = {k:all_data[k][mask_ev,:] for k in keys_to_load}
  events_pe = {k:jnp.array(ev_pe[k][:,mask_sample]) for k in ev_pe.keys()}

  if ('theta' in keys_to_load) and ('phi' in keys_to_load):
    ra, dec = angles.ra_dec_from_th_phi(events_pe['theta'], events_pe['phi'])
    events_pe['ra']  = ra
    events_pe['dec'] = dec

  return events_pe

def load_data_injection(file_inj, snr_cut = None, ninj = None):
  """Function to load injection data"""
  keys_needed1 = ['m1src', 'm2src', 'z', 'dL', 'SNR_net', 'log_p_draw_nospin']
  keys_needed2 = ['m1det', 'm2det', 'dL', 'SNR_net', 'log_p_draw_nospin']
  all_data  = load_data_h5(file_inj, group_h5=None)
  missing_keys1 = [key for key in keys_needed1 if key not in all_data]
  missing_keys2 = [key for key in keys_needed2 if key not in all_data]

  if missing_keys1 and missing_keys2:
    raise ValueError(f"{file_inj} does not contain the required keys {missing_keys1} or {missing_keys2}")

  m1det        = all_data['m1det'] if 'm1det' in all_data.keys() else all_data['m1src']*(1.+all_data['z'])
  m2det        = all_data['m2det'] if 'm2det' in all_data.keys() else all_data['m2src']*(1.+all_data['z'])
  dL           = all_data['dL']
  snr          = all_data['SNR_net']
  inj_logprior = all_data['log_p_draw_nospin']

  keep = jnp.full(len(inj_logprior), True)

  if snr_cut is not None:
    keep = keep & (snr > snr_cut)
  else:
    logger.info(f"Minimum SNR found in the file is {np.min(snr)}")

  assert (m1det > 0).all()
  assert (m2det > 0).all()
  assert (dL > 0).all()
  assert (m2det<=m1det).all()

  inj_data = {'m1det':m1det[keep], 'm2det': m2det[keep], 'dL': dL[keep]}
  max_ninj = len(inj_data['m1det'])

  inj_logprior = inj_logprior[keep]

  if ninj:
    if type(ninj)==list:
      mask_inj = jnp.array(ninj)
      logger.info('Loading injections: ', mask_inj)
    elif isinstance(ninj, float)  or isinstance(ninj,  int):
      if ninj > max_ninj:
        logger.warning('More injections requested than available. Use all of them.')
        mask_inj = jnp.arange(max_ninj)
      else:
        mask_inj  = jnp.array(sorted(np.random.choice(max_ninj, ninj, replace=False)))
        logger.info('The analysys will be performed on the randomly chosen injections: ', mask_inj)
    else:
        raise ValueError('`ninj` can only be `None`, an int or a list of int.')
  else:
    mask_inj = jnp.arange(max_ninj)

  inj_data = {k : jnp.array(inj_data[k][mask_inj]) for k in inj_data}
  inj_logprior = inj_logprior[mask_inj]

  return inj_data, jnp.exp(inj_logprior)

################
# PIXELIZATION #
################
def _get_threshold(norm_counts, level):
  '''
  Finds value mincount of normalized number counts norm_counts that bouds the x% credible region , with x=level
  Then to select pixels in that region: all_pixels[norm_count>mincount]
  '''
  prob_sorted     = np.sort(norm_counts)[::-1]
  prob_sorted_cum = np.cumsum(prob_sorted)
  idx      = np.searchsorted(prob_sorted_cum, level) # find index of array which bounds the confidence interval
  mincount = prob_sorted[idx]

  return mincount

def _compute_sky_conf_event(pe_data, event, sky_conf, nside):
  """Return all the Healpix pixel indices where the probability of an event is above a given threshold.

  Args:
    pe_data (dict(ndarray)): loaded PE of GW catalog
    event (int): number of the event
    nside (int): nside parameter for Healpix

  Returns:
    ndarray: Healpix indices of the skymap where the probability of an event is above a given threshold.
  """

  pixel_key      = f"pix{nside}"
  unique, counts = np.unique(pe_data[pixel_key][event], return_counts=True)
  p = np.zeros(hp.nside2npix(nside))
  p[unique] = counts/pe_data[pixel_key][event].shape[0]

  return np.argwhere(p >= _get_threshold(p, sky_conf)).flatten()

def pixelize_gw_catalog(pe_data,
  nside_list,
  mean_npixels_event,
  sky_conf,
  nest=False,
  prefix=None):
  """Pre-compute columns of corresponding Healpix indices for all the provided `nside_list` pixelization parameters.

  Args:
    pe_data (dict(ndarray)): loaded PE of GW catalog
    nside_list (list(int)): list of nside parameters for Healpix
    mean_npixels_event (int): approximate number of desired pixels per event
    sky_conf (0<=float<=1): percentage of the GW are to pixelize
    nest (bool): if the healpy map is nested or not
    prefix (str, optional): if provided is the prefix of the h5 file to save with the pixelated gw catalog

  Returns:
    dict with;
      array of int: optimized nside parameter for each event
      list of int: pixel indexes in the sky_conf area for each event
      list of 1d array: ra of the pixels in the sky_conf area for each event
      list of 1d array: dec of the pixels in the sky_conf area for each event
      list of 1d array: value of the GW localization probability in the pixels for each event
      ndarray(int): ndarray of shape (Nevents, Nsamples) containing the pixel idx to which each sample belongs
  """
  
  # Ensure all keys in pe_data are at least 2D (Nevents x Nsamples)
  pe_data = {k:jnp.atleast_2d(v) for k,v in pe_data.items()}
  Nevents = pe_data['dL'].shape[0]

  for n in nside_list:
    logger.info(f"Precomputing Healpix pixels for the GW events (NSIDE={n}, NEST={nest})")
    pe_data[f"pix{n}"] = angles.find_pix_RAdec(pe_data["ra"], pe_data["dec"], n, nest)

  logger.info(f"Finding optimal pixelization for each event (~{mean_npixels_event} pix/event)")

  mat = jnp.array([[len(_compute_sky_conf_event(pe_data, e, sky_conf, n)) for n in nside_list] for e in range(Nevents)])
 
  ind = jnp.argmin(jnp.abs(mat - mean_npixels_event), axis=1)
  nside_events = jnp.array(nside_list)[ind]

  u, c  = np.unique(nside_events, return_counts=True)

  logger.info(" > NSIDE: " + " ".join(f"{x:4d}" for x in u))
  logger.info(" > event:" + " ".join(f"{x:4d}" for x in c))

  pixels_idx =  [_compute_sky_conf_event(pe_data, e, sky_conf, nside_events[e]) for e in range(Nevents)]
  ra_pix, dec_pix = zip(*[angles.find_ra_dec(pixels_idx[e], nside=nside_events[e]) for e in range(Nevents)])

  logger.info("Identifying the pixel of each PE samples for each event")

  gw_loc2d_pdf = []
  pe_pix_idx = np.zeros_like(pe_data['ra'], dtype=jnp.int64) # array of shape (Nevents, Nsamples)
  for e in range(Nevents):
    ra_e  = pe_data['ra'][e]
    dec_e = pe_data['dec'][e]
    ns_e  = nside_events[e]
    ra_pix_e  = ra_pix[e]
    dec_pix_e = dec_pix[e]
    pix_idx   = pixels_idx[e]

    # compute the pixel idx of each sample
    sample_pix_idx = angles.find_pix_RAdec(ra_e, dec_e, ns_e, nest)
    # if the pixel idx is present in pix_idx list keep it otherwise assing that sample to the closest pixel
    valid_idx = np.isin(sample_pix_idx, pix_idx)
    ang_sep_matrix = angles.angular_separation_from_LOS(ra_e[:,None], dec_e[:,None], ra_pix_e[None,:], dec_pix_e[None,:])
    argmin_ang_sep = np.argmin(ang_sep_matrix, axis = 1)
    closest_pix_idx = np.where(
      valid_idx,
      sample_pix_idx,
      pix_idx[argmin_ang_sep]
    )
    pe_pix_idx[e] = closest_pix_idx

    # compute localization area propbability in each pixel
    samples_loc2d = jnp.array([ra_e, dec_e])
    grid_loc2d    = jnp.array([ra_pix_e, dec_pix_e])
    gw_loc2d_pdf.append(jax_gkde_nd(samples_loc2d, grid_loc2d))

  arr_pixels_idx = _pad_arr_list(pixels_idx, pad_value=-100)
  arr_ra_pix = _pad_arr_list(ra_pix, pad_value=-100.)
  arr_dec_pix = _pad_arr_list(dec_pix, pad_value=-100.)
  arr_gw_loc2d_pdf = _pad_arr_list(gw_loc2d_pdf, pad_value=-100.)
  pe_pix_idx = jnp.asarray(pe_pix_idx)

  if prefix is not None:
    # save the pixelated catalog to a .h5 file
    print_list = "-".join(map(str, nside_list))
    fname = prefix + f"_pixelated_nsidelist{print_list}_meanpixels{mean_npixels_event}_skyconf{sky_conf}_nest{nest}.h5"
    with h5py.File(fname, 'w') as f:
      f.create_dataset("nsides", data=nside_events)
      f.create_dataset("pixels", data=arr_pixels_idx)
      f.create_dataset("ra_pix", data=arr_ra_pix)
      f.create_dataset("dec_pix", data=arr_dec_pix)
      f.create_dataset("gw_loc2d_pdf", data=arr_gw_loc2d_pdf)
      f.create_dataset("pe_pixel", data=pe_pix_idx)
      post_group = f.create_group("posteriors")
      for k in pe_data.keys():
        post_group.create_dataset(k, data=pe_data[k])
    f.close()

  dict_toret=dict()
  dict_toret["pe_data"] = pe_data
  dict_toret["nsides"] = nside_events
  dict_toret["pixels"] = arr_pixels_idx
  dict_toret["ra_pix"] = arr_ra_pix
  dict_toret["dec_pix"] = arr_dec_pix
  dict_toret["gw_loc2d_pdf"] = arr_gw_loc2d_pdf
  dict_toret["pe_pixel"] = pe_pix_idx

  return dict_toret

def load_pixelated_gw_catalog(fname):
  """Load pixelated GW catalog"""
  dict_toret=dict()
  with h5py.File(fname, 'r') as f:
    pe_data = {}
    for k in f['posteriors'].keys():
      pe_data[k] = jnp.asarray(f['posteriors'][k][:])
    dict_toret["pe_data"] = pe_data

    dict_toret["nsides"] = jnp.asarray(f["nsides"][:])
    dict_toret["pixels"] = jnp.asarray(f["pixels"][:])
    dict_toret["ra_pix"] = jnp.asarray(f["ra_pix"][:])
    dict_toret["dec_pix"] = jnp.asarray(f["dec_pix"][:])
    dict_toret["gw_loc2d_pdf"] = jnp.asarray(f["gw_loc2d_pdf"][:])
    dict_toret["pe_pixel"] = jnp.asarray(f["pe_pixel"][:])

  f.close()
  return dict_toret

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
  """Compute the 90% localization area of each event in the dataset"""
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
  """Compute the 90% localization volume of each event in the dataset in Gpc^3"""
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
