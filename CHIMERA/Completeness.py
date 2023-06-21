#
#   This Temporary module contains classes for the completeness correction.
#   Adapted from DarkSirensStat <https://github.com/CosmoStatGW/DarkSirensStat>
#


####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

from CHIMERA.utils import angles, presets
from sklearn.cluster import AgglomerativeClustering
import healpy as hp
from scipy import interpolate, signal

import logging
log = logging.getLogger(__name__)


import numpy as np
import healpy as hp

from CHIMERA.cosmo import fLCDM
lambda_cosmo = {"H0":70.0, "Om0":0.3}

        
class Completeness(ABC):
    
    def __init__(self):
        self._computed = False
       
    def compute(self, galdata, useDirac = False): 
        log.info('Computing completeness')
        self.compute_implementation(galdata, useDirac)
        self._computed = True
    
    @abstractmethod
    def compute_implementation(self, galdata):
        pass
        
    @abstractmethod
    def zstar(self, theta, phi):
        pass
     
    def get(self, theta, phi, z, oneZPerAngle=False):
        assert(self._computed)
        if np.isscalar(z):
            return np.where(z > self.zstar(theta, phi), self.get_at_z_implementation(theta, phi, z), 1)
        else:
            if oneZPerAngle:
                assert(not np.isscalar(theta))
                assert(len(z) == len(theta))
                close = z < self.zstar(theta, phi)
                ret = np.zeros(len(z))
                ret[~close] = self.get_many_implementation(theta[~close], phi[~close], z[~close])
                ret[close] = 1
                return ret
            else:
                if not np.isscalar(theta):
                    if len(z) == len(theta):
                        log.info('Completeness::get: number of redshift bins and number of angles agree but oneZPerAngle is not True. This may be a coincidence, or indicate that the flag should be True')
                
                    ret = self.get_implementation(theta, phi, z)
                    close = z[np.newaxis, :] < self.zstar(theta, phi)[:, np.newaxis]
                    return np.where(close, 1, ret)
                else:
                    ret = self.get_implementation(theta, phi, z)
                    close = z < self.zstar(theta, phi)
                    return np.where(close, 1, ret)
                    
    
    # z is scalar (theta, phi may or may not be)
    # useful to make maps at fixed z and for single points
    @abstractmethod
    def get_at_z_implementation(self, theta, phi, z):
        pass
        
    # z, theta, phi are vectors of same length
    @abstractmethod
    def get_many_implementation(self, theta, phi, z):
        pass
    
    # z is grid, we want matrix: for each theta,phi compute *each* z
    @abstractmethod
    def get_implementation(self, theta, phi, z):
        pass
        
  
    
class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        Completeness.__init__(self, comovingDensityGoal=1, **kwargs)
        
        
    def zstar(self, theta, phi):
        if np.isscalar(theta):
            return 0
        return np.zeros(theta.size)
        
    def compute_implementation(self, galdata, useDirac):
        log.info("SkipCompleteness: nothing to compute")
        
    def get_at_z_implementation(self, theta, phi, z):
        if np.isscalar(theta):
            return 1
        return np.ones(theta.size)
        
    def get_implementation(self, theta, phi, z):
        return np.ones((theta.size, z.size))
    
    def get_many_implementation(self, theta, phi, z):
        return np.ones(theta.size)
        



class MaskCompleteness():

    def __init__(self, N_z_bins, N_masks=2, N_bar=None):
      
        assert(N_masks >= 1)
        self.N_masks        = N_masks
        self.N_z_bins       = N_z_bins
        self.N_bar          = N_bar

        self._nside         = 32
        self._npix          = hp.nside2npix(self._nside)
        self._pixarea       = hp.nside2pixarea(self._nside)

        # integer mask
        self._mask          = None

        self.z_edges        = []
        self.z_centers      = []
        self._areas         = []
        self._interp        = []
        self._zstar         = []
          


    def _compute_avg_mask(self):
        # Compute the mean luminosity for each pixel
        unique_pixels, inverse_indices = np.unique(self.gal_pix, return_inverse=True)
        avg_val = np.bincount(inverse_indices, weights=self.compl_vals) / np.bincount(inverse_indices)
        # Create an array of mean luminosities for all pixels
        avg_mask = np.zeros(self._npix)
        avg_mask[unique_pixels] = avg_val

        return avg_mask


    def _compute_z_grids(self):
        edges   = []
        centers = []

        for i in range(self.N_masks):
            # Select galaxies in the current mask group
            gal_in_mask = self.gal_mask == i

            if np.sum(gal_in_mask):
                # Compute the redshift bins for this mask
                zmax      = 1.5 * np.quantile(self.gal_z[gal_in_mask], 0.9)
                z_edges   = np.linspace(0., zmax, self.N_z_bins)
                z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])

                edges.append(z_edges)
                centers.append(z_centers)
            else:
                edges.append(None)
                centers.append(None)

        return edges, centers



    def _compute_coarse_density(self):

        def hist(mask_id):
            in_gal = (self.gal_mask == mask_id)
            z_edges = self.z_edges[mask_id]

            if z_edges is None:
                return np.zeros(self.N_z_bins)
            
            else:
                res, _ = np.histogram(self.gal_z[in_gal], bins=z_edges, weights=self.compl_vals[in_gal])
                return res.astype(float)

        rho_coarse = [np.sum(hist(i)) for i in range(self.N_masks)]     

        return rho_coarse

    # def _compute_coarse_density(self):
    #     # Initialize a 2D array to store histograms for all mask groups
    #     histograms = np.zeros((self.N_masks, self.N_z_bins))

    #     # Get bin indices for each galaxy's z value in corresponding z_edges
    #     bin_indices = np.array([np.digitize(self.gal_z[i], self.z_edges[self.gal_mask[i]]) for i in range(len(self.gal_z))])

    #     # Use np.add.at to compute the histograms
    #     np.add.at(histograms, (self.gal_mask, bin_indices), self.compl_vals)

    #     return histograms


    def _compute_Nbar(self):
        # TB checked
        rho_max = 0
        for i in range(self.N_masks):
            
            if self.z_edges[i] is not None:
                z1, z2 = self.z_edges[i][:-1], self.z_edges[i][1:]
                area = np.sum(self.mask == i) * self._pixarea
                vol = area * (fLCDM.dC(z2, presets.lambda_cosmo_737)**3 - fLCDM.dC(z1, presets.lambda_cosmo_737)**3)/3

                vol[vol < 10000] = 1e30  # spurious values, ignored by reducing the density
                rho_near = np.max(self.rho_coarse[i]/vol)

                print(rho_max, rho_near, z1, z2, vol)

                if rho_near > rho_max:
                    rho_max = rho_near
        
        print("Comoving density of galaxies: ", rho_max, "Mpc^-3")

        return rho_max
    

    def _compute_interpolants(self):

        for i in range(self.N_masks):
            area = np.sum(self.mask == i) * self._pixarea
            self._areas.append(area)

            if self.z_edges[i] is None:
                self._interp.append(None)
                self._zstar.append(None)

            else:
                z1, z2 = self.z_edges[i][:-1], self.z_edges[i][1:]
                vol = area * (fLCDM.dC(z2, presets.lambda_cosmo_737)**3 - fLCDM.dC(z1, presets.lambda_cosmo_737)**3)/3

                self.rho_coarse[i] /= vol


                # 1. Make higher resolution grid
                z_hires   = np.linspace(0, self.z_centers[i][-1], 1000)
                rho_hires = np.interp(z_hires, self.z_centers[i], self.rho_coarse[i], right=None)

                # 2. Filter the fluctuations
                rho_filtered = signal.savgol_filter(rho_hires, window_length=251, polyorder=2)
                rho_filtered_sampled = np.interp(self.z_edges[i], z_hires, rho_filtered)
                rho_filtered_sampled/= self.N_bar

                # 3. Save interpolator
                if self.z_edges[i].size > 3:
                    self._interp.append(interpolate.interp1d(self.z_edges[i], rho_filtered_sampled, kind='linear', 
                                                                    bounds_error=False, fill_value=(1, 0)))
                else:
                    self._interp.append(lambda x: np.squeeze(np.zeros(np.atleast_1d(x).shape)))

                # 4. Find the point where the interpolated result crosses 1 for the last time
                zmax    = self.z_edges[i][-1]
                z_hires = np.linspace(0, zmax, 10000)[::-1]
                evals   = self._interp[i](z_hires)
                idx     = np.argmax(evals >= 1)

                # 5. Determine the value of zstar
                if idx.size == 0 or (idx.size == 1 and evals[1] < 1):
                    self._zstar.append(0)
                    print(f"Completeness.py: catalog nowhere overcomplete in region {i}")
                else:
                    self._zstar.append(z_hires[idx])
                    if idx.size == 1:
                        print(f"Completeness.py: overcomplete catalog {i} region even at largest redshift {zmax:.3f}")
                    else:
                        print(f"Completeness.py: catalog overcomplete in region {i} up to redshift {z_hires[idx]}")

        self._zstar = np.array(self._zstar)


    def compute_implementation(self, data_gal, compl_key):

        self._nsidecol      = f"pix{self._nside}"

        self.data_gal       = data_gal

        if not self._nsidecol in self.data_gal:
            self.data_gal[self._nsidecol] = angles.find_pix_RAdec(data_gal['ra'], data_gal['dec'], self._nside)

        self.gal_z          = data_gal['z']
        self.gal_pix        = data_gal[self._nsidecol]
        self.compl_vals     = data_gal[compl_key]


        log.info("Computing average mask")
        avg_mask = self._compute_avg_mask()

        # to improve clustering
        avg_mask = np.log(avg_mask+10)

        log.info("Computing clustering")
        clusterer     = AgglomerativeClustering(self.N_masks, linkage='ward')
        self.mask     = clusterer.fit(avg_mask.reshape(-1,1)).labels_.astype(int)
        self.gal_mask = self.mask[self.gal_pix]
        

        log.info("Computing z grids")
        self.z_edges, self.z_centers = self._compute_z_grids()

        log.info("Computing coarse density")
        self.rho_coarse  = self._compute_coarse_density()


        if self.N_bar is None:
            self.N_bar = self._compute_Nbar()

        log.info("Computing interpolants")
        self._compute_interpolants()



class PixelizedCompleteness():
    
    def __init__(self, nside, N_z_bins, N_bar, interpolateOmega=False):

        self._nside    = nside
        self._npix     = hp.nside2npix(self._nside)
        self._pixarea  = hp.nside2pixarea(self._nside)

        self.N_z_bins  = N_z_bins
        self.N_bar     = N_bar
        self._interpolateOmega = interpolateOmega

        self.z_edges   = None
        self.z_centers = None
        self._zstar    = None
        self._interp   = np.zeros((self._npix, self.N_z_bins), dtype=float)

        super().__init__()


    def load(self, dir_file):

        self.dir_file = dir_file
        
        f = open(dir_file)
        header = f.readline()
        f.close()
        
        self._nside = int(header.split(',')[3].split('=')[1])
        self.N_z_bins = int(header.split(',')[4].split('=')[1])
                
        zcentermin = float(header.split(',')[1].split('=')[1])
        zcentermax = float(header.split(',')[2].split('=')[1])
        
        self.z_centers = np.linspace(zcentermin, zcentermax, self.N_z_bins)
        deltaz = self.z_centers[1]-self.z_centers[0]
        self.zMin = self.z_centers[0] - deltaz*0.5
        self.zMax = self.z_centers[-1] + deltaz*0.5
        
        self.z_edges = np.linspace(self.zMin, self.zMax, self.N_z_bins + 1)

        self.compute_implementation = self.precomputed_implementation


    def precomputed_implementation(self):
        self._map = np.loadtxt(self.dir_file).T

        zFine = np.linspace(0, self.zMax, 3000)[::-1]
        evals = self.get_implementation(*hp.pix2ang(self._nside, np.arange(self._npix)), zFine)

        idx = np.argmax(evals >= 1, axis=1)

        self._zstar = np.where(idx == 0, 0, zFine[idx])

        log.info('Completeness done.')

    def zstar(self, theta, phi):
        pix_index = hp.ang2pix(self._nside, theta, phi)
        return self._zstar[pix_index] if not self._interpolateOmega else hp.get_interp_val(self._zstar, theta, phi)

    def compute_implementation(self, data_gal, weight=None):

        self.gal_z   = data_gal['z']
        self.gal_pix = angles.find_pix_RAdec(data_gal['ra'], data_gal['dec'], self._nside)
        self.gal_w   = data_gal[weight] if weight is not None else np.ones_like(self.gal_z)

        coarseden = self._interp.copy()
        zmax = 1.5*np.quantile(self.gal_z, 0.9)
        self.z_edges  = np.linspace(0, zmax, self.N_z_bins)
        z1, z2 = self.z_edges[:-1], self.z_edges[1:]
        self.z_centers = 0.5*(z1 + z2)

        def hist(i):
            in_gal = (self.gal_pix == i)

            if in_gal.sum() == 0:
                return np.zeros(self.N_z_bins-1)
            else:
                res, _ = np.histogram(self.gal_z[in_gal], bins=self.z_edges, weights=self.gal_w[in_gal])
                return res.astype(float)
            
        coarseden = np.vstack([hist(idx) for idx in range(self._npix)])
        
        vol = self._pixarea * (fLCDM.dC(z2, presets.lambda_cosmo_737)**3 - fLCDM.dC(z1, presets.lambda_cosmo_737)**3)/3
        self._interp = coarseden / vol / self.N_bar
        
        zFine = np.linspace(0, zmax, 3000)[::-1]
        evals = self.get_implementation(*hp.pix2ang(self._nside, np.arange(self._npix)), zFine)
        idx = np.argmax(evals >= 1, axis=1)
        self._zstar = np.where(idx == 0, 0, zFine[idx])

    def get_at_z_implementation(self, theta, phi, z):
        pix_index = hp.ang2pix(self._nside, theta, phi)
        f = interpolate.interp1d(self.z_centers, self._interp[pix_index, :], kind='linear',
                                 bounds_error=False, fill_value=(1,0))
        return f(z)

    def get_many_implementation(self, theta, phi, z):
        tensorProductThresh = 4000
        if len(z) < tensorProductThresh:
            return np.diag(self.get_implementation(theta, phi, z))
        else:
            return np.array([self.get_at_z_implementation(theta[i], phi[i], z[i]) for i in range(len(z))])

    def get_implementation(self, theta, phi, z):
        pix_indices = hp.ang2pix(self._nside, theta, phi)
        f = interpolate.interp1d(self.z_centers, self._interp, kind='linear', bounds_error=False, fill_value=(1,0))
        buf = f(z)
        return buf[pix_indices, :]