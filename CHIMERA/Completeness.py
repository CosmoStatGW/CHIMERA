#
#   This module contains classes for the completeness correction.
#   Partially adapted from DarkSirensStat <https://github.com/CosmoStatGW/DarkSirensStat>
#

####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

from CHIMERA.utils import angles, presets
from sklearn.cluster import AgglomerativeClustering
import healpy as hp
from scipy import interpolate, signal
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

import logging
log = logging.getLogger(__name__)

import pickle
import numpy as np
import healpy as hp

from CHIMERA.cosmo import fLCDM
lambda_cosmo = {"H0":70.0, "Om0":0.3}

class Completeness(ABC):


    def __init__(self, **kwargs):
        self._computed = False

    @abstractmethod
    def compute(self): 
        """Perform computation and set self._computed to True"""
        pass

    @abstractmethod
    def P_compl(self):
        """Return the completeness as a function of z"""
        pass

       

def step(z, z_range):
    """Return 1 if z is in z_range, 0 otherwise."""
    return np.where(np.logical_and(z>z_range[0], z<z_range[1]), 1., 0.)

def step_smooth(x, xsig, xthr):
    """Return a smoothed step function."""
    t_thr= (xthr-x)/xsig
    return 0.5*(1+erf(t_thr/np.sqrt(2)))
    

class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        super().__init__()       
        
    def compute_implementation(self):
        self._P_compl = lambda z: np.ones_like(z)

    def P_compl(self):
        """Return the completeness function."""
        if not self._computed:
            raise RuntimeError("Must run compute method before accessing P_compl.")
        return self._P_compl


class CompletenessMICEv2(Completeness):
    """Class representing the completeness model for the MICEv2 mock catalog."""

    def __init__(self, z_range = [0.073, 1.3], kind = "step", z_sig = None):
        """Initialize the class with a threshold, a type of function, and an optional smoothing parameter

        Args:
            z_range (list, optional): completeness range. Defaults to [0.073, 1.3].
            kind (str, optional): kind of threshold. Defaults to "step".
            z_sig (_type_, optional): smoothing parameter. Defaults to None.
        """
        super().__init__()

        self.z_range = z_range
        self.kind = kind
        self.z_sig = z_sig
        self.compute()

    def compute(self):
        if self.kind == "step":
            self._P_compl = lambda z, dummy1, dummy2: step(z, self.z_range)
        elif self.kind == "step_smooth":
            self._P_compl = lambda z, dummy1, dummy2: step_smooth(z, self.z_sig, self.z_range)
        else:
            raise ValueError("kind must be step or step_smooth")

        self._computed = True

    def P_compl(self):
        """Return the completeness function."""
        if not self._computed:
            raise RuntimeError("Must run compute method before accessing P_compl.")
        return self._P_compl
    




class MaskCompleteness():

    def __init__(self, N_z_bins, N_masks=2, compl_goal=None, nside=32, compl_key=None, sigma_filter=30):
      
        assert(N_masks >= 1)
        super().__init__()
        
        self.N_masks        = N_masks
        self.N_z_bins       = N_z_bins
        self.compl_goal     = compl_goal
        self.compl_key      = compl_key
        self.sigma_filter   = sigma_filter

        self._nside         = nside
        self._npix          = hp.nside2npix(self._nside)
        self._pixarea       = hp.nside2pixarea(self._nside)
        self._nsidecol      = f"pix{self._nside}"

        self._computed      = False
        self._mask          = None
        self.z_edges        = []
        self._areas         = []
        self._interp        = []
        self._zstar         = []
    
        self.attrs_save = ["_interp", "_nside", "pix2mask", "_computed",
                           "N_masks", "N_z_bins", "compl_goal", "compl_key",
                           "sigma_filter", "_zstar"]


    def _compute_avg_mask(self):
        # Compute the sum of the completeness weights in each pixel
        unique_pixels, inverse_indices = np.unique(self.gal_pix, return_inverse=True)
        avg_val = np.bincount(inverse_indices, weights=self.compl_wei)
        avg_mask = np.zeros(self._npix)
        avg_mask[unique_pixels] = avg_val

        return avg_mask


    def _compute_coarse(self):

        counts_coarse = np.zeros((self.N_masks, self.N_z_bins), dtype=float)
        rho_coarse    = np.zeros((self.N_masks, self.N_z_bins), dtype=float)

        for i in range(self.N_masks):
            in_gal = (self.gal_mask == i)
            area   = self.N_pix_in[i] * self._pixarea

            if sum(in_gal) > 0:
                volMpc = 1e9 * area * np.array((fLCDM.dC(self.z_edges[1:], presets.lambda_cosmo_737)**3 -\
                                       fLCDM.dC(self.z_edges[:-1], presets.lambda_cosmo_737)**3)/3)
                                
                counts_coarse[i], _ = np.histogram(self.gal_z[in_gal], bins=self.z_edges, weights=self.compl_wei[in_gal])
                rho_coarse[i]       = counts_coarse[i] / volMpc

        return counts_coarse, rho_coarse



    def _compute_nbar(self):
        rho_max = 0
        
        for i in range(self.N_masks):
            if self.N_pix_in[i] > 0:
                rho_near = np.max(self.rho_coarse[i])
                rho_max = max(rho_max, rho_near)
                
        return rho_max
    

    def _compute_interpolants(self):

        for i in range(self.N_masks):
            
            area = self.N_pix_in[i] * self._pixarea
            self._areas.append(area)

            if self.N_pix_in[i] == 0:
                self._interp.append(None)
                self._zstar.append(None)
                self._interp.append(lambda x: np.squeeze(np.zeros(np.atleast_1d(x).shape)))

            else:
                # 1. Make higher resolution grid, filter the fluctuations, and renormalize to the target density
                zz           = np.linspace(0, self.z_centers[-1], 10000)
                rho_filtered = gaussian_filter1d(np.interp(zz, self.z_centers, self.rho_coarse[i], right=None), self.sigma_filter)
                # rho_filtered = savgol_filter(np.interp(zz, self.z_centers, self.rho_coarse[i], right=None), 251, 2)
                compl        = rho_filtered/self.compl_goal

                # 2. Find `zstar`, i.e. the redshift point at which `rho_filtered` crosses 1 for the last time
                zz_r    = np.array(zz)[::-1]
                compl_r = np.array(compl)[::-1]
                idx     = np.argmax(compl_r >= 1)

                # 3. Save `zstar`. If np.argmax returns 0 => catalog undercomplete / overcomplete at all redshifts
                if idx == 0:
                    if compl_r[1] < 1:
                        zstar = 0.
                        log.info(f"Catalog undercomplete in region {i}")
                    else:
                        zstar = zz_r[idx]
                        log.info(f"Catalog overcomplete in region {i} even at largest redshift z={self.z_edges[-1]:.4f}")
                else:
                    zstar = zz_r[idx]
                    log.info(f"Catalog overcomplete in region {i} up to redshift {zz_r[idx]:.4f}")

                self._zstar.append(zstar)

                # 4. Set completeness to 1 at values below zstar
                compl[zz < zstar] = 1.

                # 5. Save the interpolant
                self._interp.append(interpolate.interp1d(zz, compl, kind='linear', bounds_error=False, fill_value=(1, 0)))



    def compute(self, data_gal):

        if self._computed:
            raise RuntimeError("Completeness already computed. Re-initialize the completeness class.")

        # Initialize data
        self.data_gal       = data_gal

        if not self._nsidecol in self.data_gal:
            self.data_gal[self._nsidecol] = angles.find_pix_RAdec(data_gal['ra'], data_gal['dec'], self._nside, nest=True)

        self.gal_z      = self.data_gal['z']
        self.gal_pix    = self.data_gal[self._nsidecol]

        if self.compl_key is None:
            self.compl_wei  = np.ones_like(self.gal_z)
        else:
            self.compl_wei  = self.data_gal[self.compl_key]

        # Actual computation 
        log.info("Computing average mask")
        self.avg_mask = self._compute_avg_mask()
        self.avg_mask_log = np.log2(self.avg_mask+10) # to improve clustering

        log.info("Computing clustering")
        clusterer     = AgglomerativeClustering(self.N_masks, linkage='ward')
        self.pix2mask = clusterer.fit(self.avg_mask_log.reshape(-1,1)).labels_.astype(int)
        self.gal_mask = self.pix2mask[self.gal_pix]
        self.N_pix_in = np.array([np.sum(self.pix2mask == i) for i in range(self.N_masks)])
        
        log.info("Computing z grids")
        self.z_edges   = np.linspace(0, 1.5*np.quantile(self.gal_z, 0.9), self.N_z_bins+1)
        self.z_centers = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

        log.info("Computing coarse density")
        self.counts_coarse, self.rho_coarse = self._compute_coarse()

        if self.compl_goal is None: self.compl_goal = self._compute_compl_goal()
        log.info(f"Comoving density of galaxies goal set to: {self.compl_goal:.4f} Mpc^-3")

        log.info("Computing interpolants")
        self._compute_interpolants()
        self._computed = True

    def get_pix2mask(self, pixel, nside_cat):
        # Convert galaxy catalog => completeness pixelization
        pixel_compl = hp.ang2pix(self._nside, *hp.pix2ang(nside_cat, pixel))
        # Find the mask number corresponding to the pixel
        return self.pix2mask[pixel_compl]


    def get_fR(self, lambda_cosmo, z_res = 1000, z_det_range=[0,20]):

        zz = np.linspace(*z_det_range, z_res)
        fR_array = np.zeros(self.N_masks, dtype=float)

        for i_mask in range(self.N_masks):
            P_compl    = self._interp[i_mask](zz)
            fR_array[i_mask] = np.trapz(P_compl*np.array(fLCDM.dV_dz(zz, lambda_cosmo)), zz)

        return fR_array


    def P_compl(self, z, pixel, nside_cat):

        P_compl_array = np.zeros((len(pixel), len(z)), dtype=float)
        
        idxs_mask     = self.get_pix2mask(pixel, nside_cat)

        for i_mask in range(self.N_masks):
            P_compl_array[(idxs_mask == i_mask)] = self._interp[i_mask](z)

        return P_compl_array.T
    

    def save(self, filename):
        if not self._computed:
            raise RuntimeError("Must run compute method before saving.")
        
        state = {attr: getattr(self, attr, None) for attr in self.attrs_save}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

        log.info("Completeness saved to completeness.pkl")


    def load(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for key in state:
            setattr(self, key, state[key])