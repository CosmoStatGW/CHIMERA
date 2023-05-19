import os
import time

import numpy as np
import healpy as hp
import h5py

from scipy.stats import gaussian_kde, norm


from cosmologies import fLCDM

# All confident events with SNR>8 and FAR>1
list_O1O2_events = ['GW150914', 'GW170729', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW151012', 'GW170823', 'GW170608']
list_O3a_events  = ['GW190701_203306', 'GW190413_134308', 'GW190720_000836', 'GW190527_092055', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190413_052954', 'GW190514_065416', 'GW190731_140936', 'GW190828_065509', 'GW190706_222641', 'GW190930_133541', 'GW190408_181802', 'GW190803_022701', 'GW190915_235702', 'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190421_213856', 'GW190521', 'GW190521_074359', 'GW190910_112807', 'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190517_055101', 'GW190513_205428', 'GW190929_012149', 'GW190620_030421']
list_O3b_events  = ['GW191222_033537', 'GW200112_155838', 'GW200202_154313', 'GW191216_213338', 'GW191204_171526', 'GW200208_130117', 'GW191230_180458', 'GW200302_015811', 'GW200219_094415', 'GW191215_223052', 'GW191127_050227', 'GW200128_022011', 'GW200225_060421', 'GW200311_115853', 'GW191105_143521', 'GW191103_012549', 'GW200316_215756', 'GW200224_222234', 'GW200129_065458', 'GW191129_134029', 'GW191109_010717', 'GW200216_220804', 'GW200209_085452']

# All confident events with SNR>11 and FAR>1
list_O1O2_events11 = ['GW150914', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW170823', 'GW170608']
list_O3a_events11  = ['GW190701_203306', 'GW190720_000836', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190828_065509', 'GW190706_222641', 'GW190408_181802', 'GW190915_235702', 'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190521', 'GW190521_074359', 'GW190910_112807', 'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190513_205428']
list_O3b_events11  = ['GW191222_033537', 'GW200112_155838', 'GW191216_213338', 'GW191204_171526', 'GW191215_223052', 'GW200225_060421', 'GW200311_115853', 'GW200224_222234', 'GW200129_065458', 'GW191129_134029', 'GW191109_010717']



list_all_SNR11 = ['GW150914', 'GW170814', 'GW170809', 'GW151226', 'GW170104', 'GW170818', 'GW170823', 'GW170608',
                  'GW190701_203306', 'GW190720_000836', 'GW190708_232457', 'GW190503_185404', 'GW190924_021846', 'GW190828_065509', 'GW190706_222641', 'GW190408_181802', 'GW190915_235702', 'GW190728_064510', 'GW190727_060333', 'GW190707_093326', 'GW190828_063405', 'GW190602_175927', 'GW190521', 'GW190521_074359', 'GW190910_112807', 'GW190519_153544', 'GW190412', 'GW190512_180714', 'GW190630_185205', 'GW190513_205428',
                  'GW191222_033537', 'GW200112_155838', 'GW191216_213338', 'GW191204_171526', 'GW191215_223052', 'GW200225_060421', 'GW200311_115853', 'GW200224_222234', 'GW200129_065458', 'GW191129_134029', 'GW191109_010717']

list_NSBH = ['GW190814', 'GW200210'] 

###########################
 # Angles-related functions
##########################

# def ra_dec_from_th_phi(theta, phi):
#            ra = np.rad2deg(phi)
#            dec = np.rad2deg(0.5 * np.pi - theta)
#            return ra, dec

     
# def th_phi_from_ra_dec(ra, dec):
#        theta = 0.5 * np.pi - np.deg2rad(dec)
#        phi = np.deg2rad(ra)
#        return theta, phi


def th_phi_from_ra_dec(ra, dec):
    return 0.5 * np.pi - dec, ra

def ra_dec_from_th_phi(theta, phi):
    return phi, 0.5 * np.pi - theta



# def ra_dec_from_ipix(nside, ipix, nest=False, verbose=False):
#     """RA and dec from HEALPix index"""
#     (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
#     if verbose:
#         print("pixel has RA,DEC = ({:.2f},{:.2f})r = ({:.2f},{:.2f})d".format(phi, np.pi/2.-theta, np.rad2deg(phi), np.rad2deg(np.pi/2.-theta)))
#     return (phi, np.pi/2.-theta)


# def ipix_from_ra_dec(nside, ra, dec, nest=False):
#     """HEALPix index from RA and dec"""
#     (theta, phi) = (np.pi/2.-dec, ra)
#     return hp.ang2pix(nside, theta, phi, nest=nest)


def find_pix_RAdec(ra, dec, nside, nest=False):
    '''
    input: ra dec in radians
    output: corresponding pixel with nside given by that of the skymap
    '''
    theta, phi = th_phi_from_ra_dec(ra, dec)
    
    # Note: when using ang2pix, theta and phi must be in rad 
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix

def find_pix(theta, phi, nside, nest=False):
    '''
    input: theta phi in rad
    output: corresponding pixel with nside given by that of the skymap
    '''

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix

def find_theta_phi( pix, nside, nest=False):
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
    

def hpx_downgrade_idx(hpx_array, nside_out=1024):
    #Computes the list of explored indices in a hpx array for the chosen nside_out
    arr_down = hp.ud_grade(hpx_array, nside_out)
    return np.where(arr_down>0.)[0] 


def hav(theta):
    return (np.sin(theta/2))**2

def haversine(phi, theta, phi0, theta0):
    return np.arccos(1 - 2*(hav(theta-theta0)+hav(phi-phi0)*np.sin(theta)*np.sin(theta0)))


def gal_to_eq(l, b):
    '''
    input: galactic coordinates (l, b) in radians
    returns equatorial coordinates (RA, dec) in radians
    
    https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic
    '''
    
    l_NCP = np.radians(122.93192)
    
    del_NGP = np.radians(27.128336)
    alpha_NGP = np.radians(192.859508)
    
    
    RA = np.arctan((np.cos(b)*np.sin(l_NCP-l))/(np.cos(del_NGP)*np.sin(b)-np.sin(del_NGP)*np.cos(b)*np.cos(l_NCP-l)))+alpha_NGP
    dec = np.arcsin(np.sin(del_NGP)*np.sin(b)+np.cos(del_NGP)*np.cos(b)*np.cos(l_NCP-l))
    
    return RA, dec


###########################
###########################

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


def add_pixelized_group_GLADEp(dir_catalog, nsides = [32, 64, 128], nest=False):
    """Add new groups to the hdf5 file of GLADe+ (from zenodo GWTC-3) that
    contain the Healpix pixel indices. In this way we can load smaller portions
    of the catalog on runtime without memory probles.

    Args:
        dir_catalog (str): path to the .hdf5 file
        nsides (list, optional): pixelizations columns to add. Defaults to [32, 64, 128].
        nest (bool, optional): Healpix parameter nest. Defaults to False.
    """

    # Open the HDF5 file and get the dataset
    with h5py.File(dir_catalog, 'a') as f:
        table = f['table']
        
        # Iterate over the resolutions
        for nside in nsides:
            print("Computing indices for NSIDE={:d}".format(nside))
            # Calculate the HEALPix pixel indices for each row in the dataset
            indices = find_pix_RAdec(np.array(table['ra']), np.array(table['dec']), nside, nest)
            
            # Create a new subgroup in the HDF5 file to store the index data
            index_group = f.create_group('indices_' + str(nside))
            
            # Create a dataset in the index group to store the pixel indices
            index_ds = index_group.create_dataset('pixel_indices', data=indices)


def sum_of_gaussians(z_grid, mu, sigma, weights=None):
    # Vectorized sum of multiple Gaussians on z_grid each one with its own weight

    z_grid = np.array(z_grid)[:, np.newaxis]
    mu     = np.array(mu)
    sigma  = np.array(sigma)

    if weights is None:
        weights = np.ones(len(mu))

    dVdz     = fLCDM.dV_dz(z_grid, {"H0":70,"Om0":0.3})
    gauss    = norm.pdf(z_grid, mu, sigma)
    # integral = np.trapz(dVdz*gauss, z_grid, axis=0)

    # return np.sum(weights * dVdz * gauss/integral, axis=1)
    return np.sum(weights * dVdz * gauss, axis=1)



class Stopwatch:
    def __init__(self):
        self.start_time = time.time()

    def __call__(self, msg=None):
        elapsed_time = time.time() - self.start_time

        if msg is None:
            print("Elapsed time: {:.6f} s".format(elapsed_time))
        else:
            print("{:s}: {:.6f} s".format(msg, elapsed_time))
        self.start_time = time.time()

# Temporary 
def load_data(events, run, nSamplesUse=None, verbose=False, BBH_only=True, SNR_th = 12, FAR_th = 1):
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
                   "dL"  : 1000 * data.dL, #Mpc
                   "ra"  : data.ra,
                   "dec" : data.dec}

    return new_data






def remapMinMax(value, a=0, b=1):
    return (value - value.min()) / (value.max() - value.min()) * (b - a) + a


#####
####################
#####

def m2M(m, dL, lambda_cosmo={"H0":70.0, "Om0":0.3}):
    """Converts observed to absolute magnitude, given luminosity distance in Mpc.

    Args:
        m (float): _description_
        dL (float): _description_
    """
    return m - 5*np.log10(dL) - 25 #- 0.5*np.log10(1+Om*(D/H0)**2) 


def M2m(M, dL, lambda_cosmo={"H0":70.0, "Om0":0.3}):
    """Converts absolute to observed magnitude, given luminosity distance in Mpc.

    Args:
        m (float): _description_
        dL (float): _description_
    """
    return M + 5*np.log10(dL) + 25 #+ 0.5*np.log10(1+Om*(D/H0)**2) 