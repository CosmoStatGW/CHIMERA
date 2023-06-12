# 
#   Base functions and parameters
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import os, h5py
import healpy as hp
import numpy as np


#########################
# Some base parameters
#########################

# MASS: lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass 
lambda_mass_PLP_mock_v1   = [0.039, 3.4, 1.1, 4.8, 5.1, 87., 34., 3.6]

# RATE: R0, alphaRedshift , betaRedshift, zp 
lambda_rate_Madau_mock_v1 = {"gamma":1.9,"kappa":3.4,"zp":2.4,"R0":17,}  

# COSMOLOGY: Planck18 H0, Om0
lambda_cosmo_mock_v1      = {"H0":67.66,"Om0":0.30966}
lambda_cosmo_GLADE        = {"H0":70., "Om0":0.27}


#############################
# Lists of observed events
#############################

# All confident events with SNR>8 and FAR>1
list_O1O2_events   = ['GW150914', 'GW170729', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW151012', 'GW170823', 'GW170608']
list_O3a_events    = ['GW190701_203306', 'GW190413_134308', 'GW190720_000836', 'GW190527_092055', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190413_052954', 
                      'GW190514_065416', 'GW190731_140936', 'GW190828_065509', 'GW190706_222641', 'GW190930_133541', 'GW190408_181802', 'GW190803_022701', 'GW190915_235702', 
                      'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190421_213856', 'GW190521', 'GW190521_074359', 'GW190910_112807', 
                      'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190517_055101', 'GW190513_205428', 'GW190929_012149', 'GW190620_030421']
list_O3b_events    = ['GW191222_033537', 'GW200112_155838', 'GW200202_154313', 'GW191216_213338', 'GW191204_171526', 'GW200208_130117', 'GW191230_180458', 'GW200302_015811', 
                      'GW200219_094415', 'GW191215_223052', 'GW191127_050227', 'GW200128_022011', 'GW200225_060421', 'GW200311_115853', 'GW191105_143521', 'GW191103_012549', 
                      'GW200316_215756', 'GW200224_222234', 'GW200129_065458', 'GW191129_134029', 'GW191109_010717', 'GW200216_220804', 'GW200209_085452']

# All confident events with SNR>11 and FAR>1
list_O1O2_events11 = ['GW150914', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW170823', 'GW170608']
list_O3a_events11  = ['GW190701_203306', 'GW190720_000836', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190828_065509', 'GW190706_222641', 'GW190408_181802', 
                      'GW190915_235702', 'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190521', 'GW190521_074359', 
                      'GW190910_112807', 'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190513_205428']
list_O3b_events11  = ['GW191222_033537', 'GW200112_155838', 'GW191216_213338', 'GW191204_171526', 'GW191215_223052', 'GW200225_060421', 'GW200311_115853', 'GW200224_222234', 
                      'GW200129_065458', 'GW191129_134029', 'GW191109_010717']
list_all_SNR11     = ['GW150914', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW170823', 'GW170608',
                      'GW190701_203306', 'GW190720_000836', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190828_065509', 'GW190706_222641', 'GW190408_181802', 
                      'GW190915_235702', 'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190521', 'GW190521_074359', 'GW190910_112807', 
                      'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190513_205428',
                      'GW191222_033537', 'GW200112_155838', 'GW191216_213338', 'GW191204_171526', 'GW191215_223052', 'GW200225_060421', 'GW200311_115853', 'GW200224_222234', 
                      'GW200129_065458', 'GW191129_134029', 'GW191109_010717']
# NSBH events
list_NSBH          = ['GW190814', 'GW200210'] 


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


##############################
#  Multiprocessing functions
##############################

import multiprocessing

nCores = max(1,int(multiprocessing.cpu_count()/2)-1)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
def parmap(f, X):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nCores)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nCores)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]



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



###################
# Other functions
###################

def get_Neff_log(log_weights, log_norm):
    """Compute Neff as in Farr+2019 with log_weigths

    Args:
        log_weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_s2      = np.logaddexp.reduce(2.*log_weights) - 2.*np.log(len(log_weights)) 
    log_sig2    = logdiffexp(log_s2, 2.*log_norm-np.log(len(log_weights)))
   
    return np.exp(2.*log_norm - log_sig2)

def get_Neff(weights, norm):
    s2 = np.sum(weights**2) / len(weights)
    sig2 = s2 - norm**2

    return norm**2 / sig2

def logdiffexp(x, y):
    return x + np.log1p(-np.exp(y-x))


def log1m_exp(x):

    arr_x = 1.0 * np.array(x)
    oob = arr_x < np.log(np.finfo(arr_x.dtype).smallest_normal)
    mask = arr_x > -0.6931472  # appox -log(2)
    more_val = np.log(-np.expm1(arr_x))
    less_val = np.log1p(-np.exp(arr_x))

    return np.where(oob,0.,np.where(mask,more_val,less_val))

def log_diff_exp(a, b):
    mask = a > b
    masktwo = (a == b) & (a < np.inf)
    return np.where(mask, 1.0 * a + log1m_exp(1.0 * b - 1.0 * a), np.where(masktwo,-np.inf,np.nan))


def nanaverage(A,weights,axis):
    return np.nansum(A*weights,axis=axis) /((~np.isnan(A))*weights).sum(axis=axis)
    
def load_data_h5(fname):
    """Generic function to load data from h5 files

    Args:
        fname (str): path to the h5 file

    Returns:
        h5py.File: h5py file
    """
    events={}
    with h5py.File(fname, 'r') as f:
        for key in f.keys(): 
            events[key] = np.array(f[key])
    return events


def remapMinMax(value, a=0, b=1):
    return (value - value.min()) / (value.max() - value.min()) * (b - a) + a

import time

class Stopwatch:
    """
    Simple stopwatch class
    """
    def __init__(self):
        import time

        self.start_time = time.time()

    def __call__(self, msg=None):
        elapsed_time = time.time() - self.start_time

        if msg is None:
            print("Elapsed time: {:.6f} s".format(elapsed_time))
        else:
            print("{:s}: {:.6f} s".format(msg, elapsed_time))
        self.start_time = time.time()


# Temporary 
def load_data_LVK(events, run, nSamplesUse=None, verbose=False, BBH_only=True, SNR_th = 12, FAR_th = 1):
    import MGCosmoPop
    from MGCosmoPop.dataStructures.O1O2data import O1O2Data
    from MGCosmoPop.dataStructures.O3adata import O3aData
    from MGCosmoPop.dataStructures.O3bdata import O3bData
    dir_data = os.path.join(MGCosmoPop.Globals.dataPath, run)
    events   = {'use':events, 'not_use':None}

    

    if run == "O1O2":
        data = O1O2Data(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose, BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)
    elif run == "O3a":
        data = O3aData(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose, BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)
    elif run == "O3b":
        data = O3bData(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose,BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)

    # dt          = []
    new_data    = {"m1z" : data.m1z,
                   "m2z" : data.m2z,
                   "dL"  : data.dL, # Gpc
                   "ra"  : data.ra,
                   "dec" : data.dec}

    return new_data



###################
# Magnitudes
###################

def Mag2lum(M, band='K'):
    """Converts magnitudes to solar luminosities

    Args:
        M (float): magnitude
        band (str, optional): obs. magnitude. K corr not implemented. Defaults to 'K'.

    Returns:
        float: luminositu in units of solar luminosity
    """
    
    if band == 'bol':
        M_sun = 4.83
    elif band == 'B':
        M_sun = 4.72
    elif band == 'W1':
        M_sun = 3.24
    elif band == 'K':
        M_sun = 3.28
    else:
        ValueError("Not supported")

    return np.power(10, 0.4*(M_sun-M))


def lum2Mag(L, band='K'):
    """Converts magnitudes to solar luminosities

    Args:
        M (float): magnitude
        band (str, optional): obs. magnitude. K corr not implemented. Defaults to 'K'.

    Returns:
        float: luminositu in units of solar luminosity
    """
    
    if band == 'bol':
        M_sun = 4.83
    elif band == 'W1':
        M_sun = 3.24
    else:
        ValueError("Not supported")

    return -2,5*np.log10(L) + M_sun





######################
# Schechter functions
######################


import numpy as np
from scipy.integrate import quad

def Lstar_default(band):
    if band=="B":
        Lstar = 2.45e10
    elif band=="K":
        Lstar = 1.1e11 
    return Lstar


def lambda_default(band):
    if band=="B":
        # GWCOSMO
        return {"alpha":-1.21, "M_star":-19.70, "phi_star":5.5e-3} # "Mmin":-22.96, "Mmax":-12.96
    elif band=="K":
        # GWCOSMO
        # https://iopscience.iop.org/article/10.1086/322488/pdf
        # Mmax from  Fig 3 of the above reference 
        return {"alpha":-1.09, "M_star":-23.39, "phi_star":5.5e-3}  # , "Mmin":-27.0, "Mmax":-19.0
    elif band=="B_Gehrels_Arcavi_17":
        # used in Fischbach 2019
        return {"alpha":-1.07, "M_star":-20.47, "phi_star":5.5e-3} 
    elif band=="K_GLADE_2.4":
        # used in Fischbach 2019
        return {"alpha":-1.02, "M_star":-23.55, "phi_star":5.5e-3} 

    else:
        raise ValueError("{band} band not implemented")




# phistar does not affect the analysis, (cancels in numerator and denominator of the main analysis, as for H0)


def log_phiM(M, args_Sch, args_cosmo):
    """Compute Schechter function (in log)

    Args:
        M (np.array, float): detector frame absolute magnitudes
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        _type_: _description_
    """    
    alpha, M_star, phi_star = args_Sch.values()
    h                       = args_cosmo["H0"]/100.

    M_star   = M_star + 5.*np.log10(h)
    phi_star = phi_star * h**3
    factor   = np.power(10., 0.4*(M-M_star))

    return 0.4 * np.log(10.0) * phi_star * factor**(1.+alpha) * np.exp(-factor)
    


def log_phiM_normalized(M, Mmin, Mmax, args_Sch, args_cosmo):
    """Compute Schechter function normalized in [Mmin, Mmax]

    Args:
        M (np.array, float): detector frame absolute magnitudes
        Mmin (float): lower limit of integration
        Mmax (float): upper limit of integration
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        np.array: Normalized Schechter function 
    """
    h     = args_cosmo["H0"]/100.
    Mmin  = Mmin + 5.*np.log10(h)
    Mmax  = Mmax + 5.*np.log10(h)

    return phiM(M,args_Sch,args_cosmo)/quad(phiM, Mmin, Mmax, args=(args_Sch, args_cosmo))[0]



