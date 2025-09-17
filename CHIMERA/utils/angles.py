from .config import jax, jnp
import healpy as hp

###########################
#  Angles-related functions
###########################

def th_phi_from_ra_dec(ra, dec):
  """
  Computes (theta, phi) from (RA, dec)
  Args:
    ra (jnp.ndarray): right ascension [rad]
    dec (jnp.ndarray): declination [rad]
  Returns:
    (jnp.ndarray, jnp.ndarray): tuple of theta and phi arrays
  """
  return 0.5 * jnp.pi - dec, ra


def ra_dec_from_th_phi(theta, phi):
  """
  Computes (RA, dec) from (theta, phi)
  Args:
    theta (jnp.ndarray): angle from the north pole [rad]
    phi (jnp.ndarray): angle from the x-axis [rad]
  Returns:
    (jnp.ndarray, jnp.ndarray): tuple of RA and dec arrays
  """
  return phi, 0.5 * jnp.pi - theta


def find_pix_RAdec(ra, dec, nside, nest=False):
  """
  Computes the HEALPix pixel index of (RA, dec) given nside
  Args:
    ra (jnp.ndarray): right ascension [rad]
    dec (jnp.ndarray): declination [rad]
    nside (int): HEALPix nside parameter
    nest (bool, optional): HEALPix nest parameter. Defaults to False.
  Returns:
    jnp.ndarray: list of the corresponding HEALPix pixel indices
  """
  theta, phi = th_phi_from_ra_dec(ra, dec)

  return hp.ang2pix(nside, theta, phi, nest=nest)

def find_pix(theta, phi, nside, nest=False):
  """
  Computes the HEALPix pixel index of (theta, phi) given nside
  Args:
    theta (jnp.ndarray): angle from the north pole [rad]
    phi (jnp.ndarray): angle from the x-axis [rad]
    nside (int): HEALPix nside parameter
    nest (bool, optional): HEALPix nest parameter. Defaults to False.
  Returns:
    jnp.ndarray: list of the corresponding HEALPix pixel indices
  """
  pix = hp.ang2pix(nside, theta, phi, nest=nest)
  return pix

def find_theta_phi(pix, nside, nest=False):
  """
  Computes (theta, phi) given the HEALPix pixel and nside
  Args:
    pix (int): HEALPix pixel index
    nside (int): HEALPix nside parameter
    nest (bool, optional): HEALPix nest parameter. Defaults to False.
  Returns:
    (jnp.ndarray, jnp.ndarray): tuple of theta and phi arrays
  """
  return hp.pix2ang(nside, pix, nest=nest)

def find_ra_dec( pix, nside,  nest=False):
  """
  Computes (RA, dec) given the HEALPix pixel and nside
  Args:
    pix (int): HEALPix pixel index
    nside (int): HEALPix nside parameter
    nest (bool, optional): HEALPix nest parameter. Defaults to False.
  Returns:
    (jnp.ndarray, jnp.ndarray): tuple of RA and dec arrays
  """
  theta, phi = find_theta_phi(pix, nside,  nest=nest)
  ra, dec = ra_dec_from_th_phi(theta, phi)
  return ra, dec

def hav(theta):
  return (jnp.sin(theta/2))**2

def haversine(phi, theta, phi0, theta0):
  return jnp.arccos(1 - 2*(hav(theta-theta0)+hav(phi-phi0)*np.sin(theta)*np.sin(theta0)))

def gal_to_eq(l, b):
  """
  Computed equatorial (RA, dec) coordinates from galactic coordinates.
  See: https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galacti
  Args:
    l (jnp.ndarray): galactic longitude [rad]
    b (jnp.ndarray): glacitc latitude [rad]
  Returns:
    (jnp.ndarray, jnp.ndarray): (RA, dec) coordinates [rad]
  """
  l_NCP     = jnp.radians(122.93192)
  del_NGP   = jnp.radians(27.128336)
  alpha_NGP = jnp.radians(192.859508)

  RA = jnp.arctan((jnp.cos(b)*np.sin(l_NCP-l))/(jnp.cos(del_NGP)*jnp.sin(b)-jnp.sin(del_NGP)*jnp.cos(b)*jnp.cos(l_NCP-l)))+alpha_NGP
  dec = jnp.arcsin(jnp.sin(del_NGP)*jnp.sin(b)+jnp.cos(del_NGP)*jnp.cos(b)*jnp.cos(l_NCP-l))

  return RA, dec

def healpixelize(nside, ra, dec, nest=False):
  """
  HEALPix index from RA and dec (expressed in radians!)
  Args:
    nside (int): HEALPix nside parameter
    ra (jnp.ndarray): right ascension [rad]
    dec (jnp.ndarray): declination [rad]
    nest (bool, optional): HEALPix nest parameter. Defaults to False.
  Returns:
    Dict[int, jnp.ndarray]: indexes of objects falling within the same HEALPix pixel.

  """
  # Taken from gwcosmo
  # Convert (RA, DEC) to (theta, phi)
  theta, phi = th_phi_from_ra_dec(ra, dec)

  # Hierarchical Equal Area isoLatitude Pixelation and corresponding sorted indices
  healpix          = hp.ang2pix(nside, theta, phi, nest=nest)
  healpix_idx_sort = jnp.argsort(healpix)
  healpix_sorted   = healpix[healpix_idx_sort]

  # healpix_hasobj: healpix containing an object (ordered)
  # idx_split: where to cut, i.e. indices of 'healpix_sorted' where a change occours
  healpix_hasobj, idx_start = jnp.unique(healpix_sorted, return_index=True)

  # Split healpix
  healpix_splitted = jnp.split(healpix_idx_sort, idx_start[1:])

  dicts = {}
  for i, key in enumerate(healpix_hasobj):
    dicts[key] = healpix_splitted[i]

  return dicts

def angular_separation_from_LOS(ra, dec, ra_los, dec_los):
  """
  Finds the angular separation between the point defined by (RA, dec) and the LOS defined by (RA_los, dec_los)
  Args:
    ra (jnp.ndarray): point right ascension [rad]
    dec (jnp.ndarray): point declination [rad]
    ra_los (jnp.ndarray): LOS right ascension [rad]
    dec_los (jnp.ndarray): LOS declination [rad]
  Rerturns:
    jnp.ndarray: angular separation
  """

  cos_angle = jnp.sin(dec)*jnp.sin(dec_los) + jnp.cos(dec)*jnp.cos(dec_los)*jnp.cos(ra-ra_los)
  angle = jnp.arccos(cos_angle)
  return angle


def convert_pixelization(pixels, nside_in, nside_out, nest_in=False, nest_out=False):
    """
    Converts HEALPix pixels from one resolution/ordering scheme to another.

    Args:
        pixels (numpy.ndarray): Input pixel indices (2D array of shape [n_configs, n_samples])
        nside_in (int or numpy.ndarray): NSIDE parameter of input pixels (scalar or array of length n_configs)
        nside_out (int): NSIDE parameter for output pixels
        nest_in (bool, optional): Input uses NESTED ordering. Defaults to False.
        nest_out (bool, optional): Output uses NESTED ordering. Defaults to False.

    Returns:
        jnp.ndarray: Converted pixel indices with same shape as input
    """
    # import jax.numpy as jnp
    # import jax_healpy as jhp
    import numpy as np

    pixels = np.atleast_2d(pixels)
    nside_in = np.atleast_1d(nside_in)

    assert pixels.shape[0] == nside_in.shape[0], f"nside_in shape {nside_in.shape} does not match first dimension of pixels {pixels.shape}"

    results = []
    for i in range(pixels.shape[0]):
        theta, phi = hp.pix2ang(int(nside_in[i]), pixels[i], nest=nest_in)
        results.append(hp.ang2pix(nside_out, theta, phi, nest=nest_out))

    return xp.stack(results)
