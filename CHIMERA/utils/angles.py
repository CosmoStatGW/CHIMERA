import numpy as np
import healpy as hp

###########################
#  Angles-related functions
###########################

def th_phi_from_ra_dec(ra, dec):
    """From (RA, dec) to (theta, phi)

    Args:
        ra (np.ndarray): right ascension [rad]
        dec (np.ndarray): declination [rad]

    Returns:
        [np.ndarray, np.ndarray]: list of theta and phi
    """
    return 0.5 * np.pi - dec, ra


def ra_dec_from_th_phi(theta, phi):
    """From (theta, phi) to (RA, dec)

    Args:
        theta (np.ndarray): angle from the north pole [rad]
        phi (np.ndarray): angle from the x-axis [rad]

    Returns:
        [np.ndarray, np.ndarray]: list of RA and dec
    """
    return phi, 0.5 * np.pi - theta


def find_pix_RAdec(ra, dec, nside, nest=False):
    """From (RA, dec) to HEALPix pixel index

    Args:
        ra (np.ndarray): right ascension [rad]
        dec (np.ndarray): declination [rad]
        nside (int): HEALPix nside parameter
        nest (bool, optional): HEALPix nest parameter. Defaults to False.

    Returns:
        np.ndarray: list of the corresponding HEALPix pixel indices
    """
    theta, phi = th_phi_from_ra_dec(ra, dec)

    return hp.ang2pix(nside, theta, phi, nest=nest)


def find_pix(theta, phi, nside, nest=False):
    """From (theta, phi) to HEALPix pixel index

    Args:
        theta (np.ndarray): angle from the north pole [rad]
        phi (np.ndarray): angle from the x-axis [rad]
        nside (int): HEALPix nside parameter
        nest (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix


def find_theta_phi(pix, nside, nest=False):
    '''
    input:  pixel
    output: (theta, phi)of pixel center in rad, with nside given by that of the skymap 
    '''
    return hp.pix2ang(nside, pix, nest=nest)


def find_ra_dec( pix, nside,  nest=False):
    '''
    input:  pixel ra dec in degrees
    output: (ra, dec) of pixel center in degrees, with nside given by that of the skymap 
    '''
    theta, phi = find_theta_phi(pix, nside,  nest=nest)
    ra, dec = ra_dec_from_th_phi(theta, phi)
    return ra, dec


def hav(theta):
    return (np.sin(theta/2))**2

def haversine(phi, theta, phi0, theta0):
    return np.arccos(1 - 2*(hav(theta-theta0)+hav(phi-phi0)*np.sin(theta)*np.sin(theta0)))


def gal_to_eq(l, b):
    """ From galactic coordinates to equatorial (RA, dec) coordinates.
    See: https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic

    Args:
        l (np.ndarray): galactic longitude [rad]
        b (np.ndarray): glacitc latitude [rad]

    Returns:
        [np.ndarray, np.ndarray]: (RA, dec) coordinates [rad]
    """
    l_NCP     = np.radians(122.93192)
    del_NGP   = np.radians(27.128336)
    alpha_NGP = np.radians(192.859508)
    
    RA = np.arctan((np.cos(b)*np.sin(l_NCP-l))/(np.cos(del_NGP)*np.sin(b)-np.sin(del_NGP)*np.cos(b)*np.cos(l_NCP-l)))+alpha_NGP
    dec = np.arcsin(np.sin(del_NGP)*np.sin(b)+np.cos(del_NGP)*np.cos(b)*np.cos(l_NCP-l))
    
    return RA, dec


def healpixelize(nside, ra, dec, nest=False):
    """HEALPix index from RA and dec (expressed in radians!)"""
    # Taken from gwcosmo

    # Convert (RA, DEC) to (theta, phi)
    theta = np.pi/2. - dec
    phi   = ra

    # Hierarchical Equal Area isoLatitude Pixelation and corresponding sorted indices
    healpix             = hp.ang2pix(nside, theta, phi, nest=nest)
    healpix_idx_sort    = np.argsort(healpix)
    healpix_sorted      = healpix[healpix_idx_sort]

    # healpix_hasobj: healpix containing an object (ordered)
    # idx_split: where to cut, i.e. indices of 'healpix_sorted' where a change occours
    healpix_hasobj, idx_start = np.unique(healpix_sorted, return_index=True)

    # Split healpix 
    healpix_splitted = np.split(healpix_idx_sort, idx_start[1:])

    dicts = {}
    for i, key in enumerate(healpix_hasobj):
        dicts[key] = healpix_splitted[i]

    return dicts
