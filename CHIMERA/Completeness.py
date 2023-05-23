#
#   This Temporary module contains classes for the completeness correction.
#   Taken from DarkSirensStat <https://github.com/CosmoStatGW/DarkSirensStat> with some modifications.
#


####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

# from software.CHIMERA.CHIMERA._keelin import bounded_keelin_3_discrete_probabilities_between
# from globals import *

import CHIMERA.chimeraUtils as chimeraUtils

import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

from CHIMERA.cosmologies import fLCDM
lambda_cosmo = {"H0":70.0, "Om0":0.3}

from sklearn import cluster

def get_SchNorm(phistar, Lstar, alpha, Lcut):
        '''
        
        Input:  - Schechter function parameters L_*, phi_*, alpha
                - Lilit of integration L_cut in units of 10^10 solar lum.
        
        Output: integrated Schechter function up to L_cut in units of 10^10 solar lum.
        '''
        from scipy.special import gammaincc
        from scipy.special import gamma
                
        norm= phistar*Lstar*gamma(alpha+2)*gammaincc(alpha+2, Lcut)
        return norm

        
class Completeness(ABC):
    
    def __init__(self, comovingDensityGoal, **kwargs):
        self._computed = False
        self._comovingDensityGoal = comovingDensityGoal
       
    def compute(self, galdata, useDirac = False): 
        log.info('Computing completeness')
        self.compute_implementation(galdata, useDirac)
        self._computed = True
    
    @abstractmethod
    def compute_implementation(self, galdata, useDirac):
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
        


 




class MaskCompleteness_v2(Completeness):

    def __init__(self, comovingDensityGoal, zRes, nMasks=2, **kwargs):
      
        assert(nMasks >= 1)
        self._nMasks        = nMasks
        self._zRes          = zRes
        
        # for the resolution of the masks
        self._nside         = 32
        self._npix          = hp.nside2npix(self._nside)
        self._pixarea       = hp.nside2pixarea(self._nside)
        # integer mask
        self._mask          = None

        self.zedges         = []
        self.zcenters       = []
        self.areas          = []
        self._compl         = []
        self._interpolators = []
        self._zstar         = []
          
          
        Completeness.__init__(self, comovingDensityGoal, **kwargs)
          
    def zstar(self, theta, phi):
        return self._zstar[ self._mask[hp.ang2pix(self._nside, theta, phi)] ]
          
      
    def compute_implementation(self, galdata, useDirac):
    
        ### MAKE MASKS ###
        # the only feature we use is number of galaxies in a pixel

        X           = np.zeros((self._npix, 1))
        avgweight   = np.mean(galdata.completenessGoal.to_numpy())

        galdata.set_index(keys=['pix'], drop=False, inplace=True)
        foo         = galdata.groupby(level=0).completenessGoal.sum()/avgweight
        X[foo.index.to_numpy(), 0] = foo.to_numpy()
        X           = np.log(X+10)  # this improves the clustering (at least for GLADE)

        log.info(" > making {:d} masks...".format(self._nMasks))
        clusterer   = cluster.AgglomerativeClustering(self._nMasks, linkage='ward')
        self._mask  = clusterer.fit(X).labels_.astype(np.int)
        
        galdata.loc[:,'component'] = self._mask[galdata['pix'].to_numpy()]
        galdata.set_index(keys=['component'], drop=False, inplace=True)
        gr          = galdata.groupby(level=0)
        
        for i in np.arange(self._nMasks):

            try:
                galcomp = gr.get_group(i)

                #zmax = 1.0001*np.max(galcomp.z.to_numpy())
                zmax = 1.5*np.quantile(galcomp.z.to_numpy(), 0.9)
                self.zedges.append(np.linspace(0, zmax, self._zRes+1))
                self.zcenters.append(0.5*(self.zedges[i][:-1] + self.zedges[i][1:]))
                self.areas.append(np.sum(self._mask == i)*self._pixarea)

            except KeyError as e: 
                # label i was never put into the map, or it is the component without any galaxies.
                # fill in some irrelevant but non-breaking stuff
                #catparts.append(pd.DataFrame())
                self.zedges.append(np.array([0,1,2]))
                self.zcenters.append(np.array([0.5,1.5]))
                self.areas.append([1,1])
        
        log.info(" > computing in parallel")
       

        def g(galgroups, maskId, batchId, nBatches):
       
            zedges   = self.zedges[maskId]
            
            try:
                gals = galgroups.get_group(maskId)
            
            except KeyError as e: #if len(galpixel) == 0:
                return np.zeros(len(zedges)-1)
            
            N      = len(gals)
            n      = int(N/nBatches)
            start  = n*batchId
            stop   = n*(1+batchId) 
            if batchId == nBatches-1:
                stop = N

            batch = gals.iloc[start:stop]

            if useDirac:
                res, _ = np.histogram(a=batch.z.to_numpy(), bins=zedges, weights=batch.completenessGoal.to_numpy())
                return res.astype(float)
            else:
                # TBD
                weights = bounded_keelin_3_discrete_probabilities_between(zedges, 0.16, batch.z_lower, batch.z, batch.z_upper, batch.z_lowerbound, batch.z_upperbound, N=100)
                            
                if weights.ndim == 1:  # if there is 1 galaxy only, weights won't be a matrix - fix
                    weights = weights[np.newaxis, :]
                
                return np.sum(weights * batch.completenessGoal[:, np.newaxis], axis=0)
                
        coarseden   = []  
        nBatches    = int(60.*len(galdata)/1000000.)
        log.info('    > batch number: ' + str(nBatches))
            
        for i in np.arange(self._nMasks):   
            coarseden.append(sum(chimeraUtils.parmap(lambda b : g(gr, maskId=i, batchId=b, nBatches=nBatches), range(nBatches))))
       
        log.info("Final computations for completeness")
       
        if self._comovingDensityGoal == 'auto':
            maxden = 0 
            for i in np.arange(self._nMasks):

                z1 = self.zedges[i][:-1]
                z2 = self.zedges[i][1:]
                vol = self.areas[i] * (fLCDM.dC(z2, lambda_cosmo)**3 - fLCDM.dC(z1, lambda_cosmo)**3)/3
                # too small volumes can be spurious. Ignore these by reducing the density so they won't be picked 
                vol[vol < 10000] = 1e30
                nearden = np.max(coarseden[i]/vol)
                
                log.info("auto density goal: {:.2f}, {:.2f}, {:.3f}, {:.3f}".format(maxden, nearden, z1, z2, vol))

                if nearden > maxden:
                    import copy
                    maxden = copy.copy(nearden)

            self._comovingDensityGoal = maxden

            log.info("Comoving density goal is set to " + str(self._comovingDensityGoal) )


        for i in np.arange(self._nMasks):
            z1 = self.zedges[i][:-1]
            z2 = self.zedges[i][1:]
            vol = self.areas[i] * (fLCDM.dC(z2, lambda_cosmo)**3 - fLCDM.dC(z1, lambda_cosmo)**3)/3
        
            coarseden[i] /= vol
            
            from scipy import interpolate
            # first, make a 1000 pt linear interpolation
           
            zmax = self.zedges[i][-1]*1.001
            zFine = np.linspace(0, zmax, 1000)
            
            coarseden_interp = np.interp(zFine, self.zcenters[i], coarseden[i], right=None)
            
            # now filter our fluctuations.
            from scipy.signal import savgol_filter
            # with 251 points (must be odd) there are effectively 4 independent points left
            # to describe the decay in the intervall adjusted to the part of the mask
            coarseden_filtered      = np.zeros(coarseden_interp.shape)
            n                       = 0
            coarseden_filtered[n:]  = savgol_filter(coarseden_interp[n:], 251, 2)
            
            # build a quadratic interpolator. A subseet of points is enough (will be faster to evaluate).
            coarseden_filtered_sampled = np.interp(self.zcenters[i], zFine, coarseden_filtered)
            
            # save this just in case
            self._compl.append(coarseden_filtered_sampled.copy()/self._comovingDensityGoal)
        
            # interpolator
            if self.zcenters[i].size > 3:
                self._interpolators.append(interpolate.interp1d(self.zcenters[i], self._compl[i], kind='linear', bounds_error=False, fill_value=(1,0)))
            else:
                self._interpolators.append(lambda x: np.squeeze(np.zeros(np.atleast_1d(x).shape)))
        
        
            # find the point where the interpolated result crosses 1 for the last time
            zFine = np.linspace(0, self.zcenters[i][-1]*0.999, 10000)
            zFine = zFine[::-1]
            evals = self._interpolators[i](zFine)
        
            # argmax returns "first" occurence of maximum, which is True in a boolean array. we search starting at large z due to the flip
            idx = np.argmax(evals >= 1)
            # however, if all enries are False, argmax returns 0, which would be the largest redshift, while we want 0 in that case
            # if all entries are True, we want indeed the largest 
            if idx == 0:
                if evals[1] < 1:
                    self._zstar.append(0)
                    log.info(" - catalog nowhere overcomplete in region {}".format(i))
                else:
                    self._zstar.append(zFine[idx])
                    log.warning(" - WARNING: overcomplete catalog {} region even at largest redshift {}".format(i, zmax))
            else:
                self._zstar.append(zFine[idx])
                log.info(" - catalog overcomplete in region {} up to z={:.3f}".format(i, zFine[idx]))

            
        self._zstar = np.array(self._zstar)

        log.info("Completeness done!")
        
        
        
    # z is a number
    def get_at_z_implementation(self, theta, phi, z):

        from scipy import interpolate

        # if only one point, things are pretty clear
        if np.isscalar(theta):
            
            component = self._mask[hp.ang2pix(self._nside, theta, phi)]
            return self._interpolators[component](z)
            
        else:
            components = self._mask[hp.ang2pix(self._nside, theta, phi)]
            
            ret = np.zeros(len(theta))
            
            for i in np.arange(self._nMasks):
                # select arguments in this component
                compMask = (components == i)
                
                if np.sum(compMask) > 0:
                    
                    # do a single calculation
                    
                    res = self._interpolators[i](z)
                    
                    # put it into all relevant outputs
                    ret[compMask] = res
                    
            return ret
                
    def get_many_implementation(self, theta, phi, z):
        
        tensorProductThresh = 4000 # copied from SuperpixelCompletenss, ideally recheck
        
        if (len(z) < tensorProductThresh):
        
            res = self.get_implementation(theta, phi, z)
            return np.diag(res)
        
        
        ret = np.zeros(len(z))
                
        components = self._mask[hp.ang2pix(self._nside, theta, phi)]

        for i in np.arange(self._nMasks):
            # select arguments in this component
            compMask = (components == i)
            
            if np.sum(compMask) > 0:
                
                # res is a vector here
                res = self._interpolators[i](z[compMask])
                
                # put it into all relevant outputs
                ret[compMask] = res
                
        return ret
        
    def get_implementation(self, theta, phi, z):
        
        from scipy import interpolate
        
        components = self._mask[hp.ang2pix(self._nside, theta, phi)]
        
        if not np.isscalar(theta):
            ret = np.zeros((len(theta), len(z)))
        else:
            ret = np.zeros(len(z))
        
        for i in np.arange(self._nMasks):
            # select arguments in this component
            compMask = (components == i)
            
            if np.sum(compMask) > 0:
                
                # do a single calculation
                
                res = self._interpolators[i](z)
                
                # put it into all relevant outputs
                if not np.isscalar(theta):
                    ret[compMask, :] = res
                else:
                    ret = res
                
        return ret



