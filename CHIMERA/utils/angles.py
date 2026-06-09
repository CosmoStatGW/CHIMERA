import jax.numpy as jnp
import healpy as hp
import numpy as np

_L_NCP = jnp.radians(122.93192)
_DEL_NGP = jnp.radians(27.128336)
_ALPHA_NGP = jnp.radians(192.859508)
_COS_NGP = jnp.cos(_DEL_NGP)
_SIN_NGP = jnp.sin(_DEL_NGP)

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
  return hp.ang2pix(nside, theta, phi, nest=nest)

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
  return ra_dec_from_th_phi(theta, phi)

def hav(theta):
  """Haversine function: sin²(θ/2).
  
  Args:
    theta (jnp.ndarray): Angle [rad]
  
  Returns:
    jnp.ndarray: sin²(θ/2)
  """
  return jnp.square(jnp.sin(theta * 0.5))

def haversine(phi, theta, phi0, theta0):
  """Great circle distance using haversine formula.
  
  Args:
    phi (jnp.ndarray): Azimuthal angle [rad]
    theta (jnp.ndarray): Polar angle [rad]
    phi0 (jnp.ndarray): Reference azimuthal angle [rad]
    theta0 (jnp.ndarray): Reference polar angle [rad]
  
  Returns:
    jnp.ndarray: Angular distance [rad]
  """
  return jnp.arccos(1.0 - 2.0 * (hav(theta - theta0) + hav(phi - phi0) * jnp.sin(theta) * jnp.sin(theta0)))

def gal_to_eq(l, b):
  """
  Converts galactic (l, b) to equatorial (RA, dec) coordinates.
  
  Uses IAU transformation from galactic to equatorial coordinates.
  See: https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic
  
  Args:
    l (jnp.ndarray): Galactic longitude [rad]
    b (jnp.ndarray): Galactic latitude [rad]
  
  Returns:
    tuple[jnp.ndarray, jnp.ndarray]: (RA, dec) coordinates [rad]
  """
  cos_b = jnp.cos(b)
  sin_b = jnp.sin(b)
  cos_l_diff = jnp.cos(_L_NCP - l)
  
  numerator = cos_b * jnp.sin(_L_NCP - l)
  denominator = _COS_NGP * sin_b - _SIN_NGP * cos_b * cos_l_diff
  RA = jnp.arctan2(numerator, denominator) + _ALPHA_NGP
  
  dec = jnp.arcsin(_SIN_NGP * sin_b + _COS_NGP * cos_b * cos_l_diff)
  
  return RA, dec

def healpixelize(nside, ra, dec, nest=False):
  """Groups object indices by their HEALPix pixel location.
  
  Converts RA/dec coordinates to HEALPix pixels and groups object indices
  that fall within the same pixel. Useful for spatial clustering and
  efficient lookups of objects within sky regions.
  
  Args:
    nside (int): HEALPix resolution parameter (npix = 12 * nside²)
    ra (jnp.ndarray): Right ascension [rad]
    dec (jnp.ndarray): Declination [rad]
    nest (bool, optional): Use NESTED ordering scheme. Defaults to False (RING).
  
  Returns:
    dict[int, jnp.ndarray]: Mapping from HEALPix pixel index to array of
                            object indices falling within that pixel.
  
  Example:
    >>> ra = jnp.array([0.1, 0.2, 0.1])
    >>> dec = jnp.array([0.5, 0.5, 0.51])
    >>> pix_dict = healpixelize(32, ra, dec)
    >>> # Objects 0 and 2 likely in same pixel, object 1 in another
  """
  theta, phi = th_phi_from_ra_dec(ra, dec)
  
  healpix = hp.ang2pix(nside, theta, phi, nest=nest)
  healpix_idx_sort = jnp.argsort(healpix)
  healpix_sorted = healpix[healpix_idx_sort]
  
  healpix_unique, idx_start = jnp.unique(healpix_sorted, return_index=True)
  healpix_groups = jnp.split(healpix_idx_sort, idx_start[1:])
  
  return {int(pix): indices for pix, indices in zip(healpix_unique, healpix_groups)}

def angular_separation_from_LOS(ra, dec, ra_los, dec_los):
  """Computes angular separation between sky position and line-of-sight.
  
  Uses spherical trigonometry to compute the angular distance between
  a point (RA, dec) and a reference direction (RA_los, dec_los).
  
  Args:
    ra (jnp.ndarray): Point right ascension [rad]
    dec (jnp.ndarray): Point declination [rad]
    ra_los (jnp.ndarray): Line-of-sight right ascension [rad]
    dec_los (jnp.ndarray): Line-of-sight declination [rad]
  
  Returns:
    jnp.ndarray: Angular separation [rad]
  """
  cos_angle = jnp.sin(dec) * jnp.sin(dec_los) + jnp.cos(dec) * jnp.cos(dec_los) * jnp.cos(ra - ra_los)
  return jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))


def convert_pixelization(pixels, nside_in, nside_out, nest_in=False, nest_out=False):
    """Converts HEALPix pixels between different resolutions/orderings.
    
    Transforms pixel indices from one HEALPix resolution (nside_in) to another
    (nside_out), optionally converting between RING and NESTED ordering schemes.
    Handles both uniform and varying input resolutions.
    
    Args:
        pixels (array-like): Input pixel indices. Shape: (n_configs, n_samples) or (n_samples,)
        nside_in (int or array-like): Input NSIDE parameter(s). If array, length must match
                                      first dimension of pixels.
        nside_out (int): Output NSIDE parameter for all pixels
        nest_in (bool, optional): Input uses NESTED ordering. Defaults to False (RING).
        nest_out (bool, optional): Output uses NESTED ordering. Defaults to False (RING).
    
    Returns:
        numpy.ndarray: Converted pixel indices with same shape as input
    
    Raises:
        AssertionError: If nside_in shape doesn't match first dimension of pixels
    
    Example:
        >>> # Convert from nside=64 to nside=128
        >>> pixels_in = np.array([[100, 101, 102]])
        >>> pixels_out = convert_pixelization(pixels_in, 64, 128)
    """
    import numpy as np
    
    pixels = np.atleast_2d(pixels)
    nside_in = np.atleast_1d(nside_in)
    
    assert pixels.shape[0] == nside_in.shape[0], \
        f"nside_in length {nside_in.shape[0]} != pixels first dimension {pixels.shape[0]}"
    
    results = np.empty_like(pixels, dtype=np.int64)
    for i in range(pixels.shape[0]):
        theta, phi = hp.pix2ang(int(nside_in[i]), pixels[i], nest=nest_in)
        results.append(hp.ang2pix(nside_out, theta, phi, nest=nest_out))

    return np.stack(results)
