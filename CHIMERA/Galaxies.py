####
# This module contains a abstract classes to handle a galaxy catalogue
####

import os
import pandas as pd
import healpy as hp
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy

import logging
log = logging.getLogger(__name__)

import DSglobals as glob  
import DSutils as utils  


# from cosmologies import fLCDM
# lambda_cosmo_GLADE = {"H0":glob.H0GLADE, "Om0":glob.Om0GLADE}




class GalCat(ABC):
    
    def __init__(self, 
                 foldername,

                 # Pixellization
                 nside = None,
                 nest  = False,
                 
                 # Completeness
                 completeness = None,
                 useDirac = None,

                 **kwargs,
                 ):
        
        self._path = os.path.join( glob.dirName, 'data', foldername)

        self.nside = np.atleast_1d(nside)
        self.nest  = nest

        self.data  = pd.DataFrame()
        self.load(**kwargs)
        self.selectedData = self.data

        if completeness is not None:
            self._completeness          = deepcopy(completeness)
            self._useDirac              = useDirac
            self._completeness.compute(self.data, self._useDirac)

        if nside is not None:
            self.prepixelize()
        

    @abstractmethod
    def load(self):
        pass



    def prepixelize(self):
        """
        Pre-compute columns of corresponding Healpix indices for all the pixelization parameters provided.
        """
        for nsid in np.unique(self.nside):
            log.info("Precomputing Healpixels for the galaxy catalog (NSIDE={:d}, NEST={})".format(nsid,self.nest))

            self.data.loc[:,"pix"+str(nsid)] = utils.find_pix_RAdec(self.data.ra.to_numpy(),self.data.dec.to_numpy(),nsid,self.nest)
        
        # coln  = "pix"+str(nside)
        # N_gal_pix = np.array([sum(self.selectedData["pix"+str(nside)] == pix) for pix in pixels])


    def precompute(self, nside, pix_todo, z_grid, names=None, weights="N"):

        Nevents  = len(nside)
        assert Nevents == len(pix_todo) == len(z_grid)
        log.info("Precomputing p_GAL for {:d} events...".format(Nevents))


        # To be improved
        res = []
        wei = []

        for e in range(Nevents):

            if names is None:
                log.info("### Event {:d}/{:d} ###".format(e+1, Nevents))
            else:
                log.info("### {:s} ###".format(names[e]))


            r, data = self.compute_event(nside[e], pix_todo[e], z_grid[e])
            
            if weights == "N":
                w = np.array([sum(data["pix"+str(nside[e])] == p) for p in pix_todo[e]])
            else:
                w = np.ones_like(r)

            res.append(r)
            wei.append(w)
            
        return res, wei


    def compute_event(self, nside, pix_todo, z_grid):

        data   = self.select_event_region(z_grid[0], z_grid[-1], pix_todo, nside)
        pixels = data["pix"+str(nside)]
        p_gal  = np.vstack([utils.sum_of_gaussians(z_grid, data.z[pixels == p], data.z_err[pixels == p]) for p in pix_todo]).T
        # p_gal /= np.trapz(p_gal, z_grid, axis=0) 
        p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls into an empty pixel



        # data   = self.select_event_region(z_grid[0], z_grid[-1], pix_todo, nside)
        # zextmax = z_grid[-1]+np.max(data.z_err)
        # # print(z_grid[-1], np.max(data.z_err), zextmax)
        # z_grid_ext = np.linspace(z_grid[0], z_grid[-1]+zextmax, 
        #                          len(z_grid) + int((zextmax - z_grid[-1]) / (z_grid[-1] - z_grid[0])))

        # data   = self.select_event_region(z_grid_ext[0], z_grid_ext[-1], pix_todo, nside)
        
        # pixels = data["pix"+str(nside)]
        
        # p_gal  = np.vstack([utils.sum_of_gaussians(z_grid_ext, data.z[pixels == p], data.z_err[pixels == p]) for p in pix_todo]).T
        # p_gal /= np.trapz(p_gal, z_grid_ext, axis=0) 
        # p_gal  = p_gal[:len(z_grid),:] #next on z_grid_ext but throwing away the others
        # p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls into an empty pixel

        return p_gal, data


    def select_event_region(self, z_min, z_max, pixels, nside):
        """Select a sub-catalog containing the galaxies inside the given redshift interval and Healpix pixels.

        Args:
            z_min (float): minimum redshift
            z_max (float): maximum redshift
            pixels (np.ndarray, int): Helpix pixels to select.
        """
        
        log.info("Setting sky area to {:d} confident pixels (nside={:d})".format(pixels.shape[0], nside))
        mask              = self.data.isin({"pix"+str(nside): pixels}).any(1)
        selected          = self.data[mask]
        log.info(" > kept {:d} galaxies".format(selected.shape[0]))
        log.info("Setting z range: {:.3f} < z < {:.3f}".format(z_min, z_max,3))
        selected          = selected[(selected.z >= z_min) & (selected.z < z_max)]
        log.info(" > kept {:d} galaxies".format(selected.shape[0]))
        N_gal_pix         = np.array([sum(selected["pix"+str(nside)] == pix) for pix in pixels])
        log.info(" > mean {:.1f} galaxies per pixel".format(np.mean(N_gal_pix)))
        
        return selected

    # def get_data(self):
    #     return self.selectedData

    # def count_selection(self):
    #     return self.selectedData.shape[0]

    def completeness(self, theta, phi, z, oneZPerAngle=False):
        return self._completeness.get(theta, phi, z, oneZPerAngle) + 1e-9










    # def select_event_region(self, z_min, z_max, pixels):
    #     """Select a sub-catalog containing the galaxies inside the given redshift interval and Healpix pixels.

    #     Args:
    #         z_min (float): minimum redshift
    #         z_max (float): maximum redshift
    #         pixels (np.ndarray, int): Helpix pixels to select.
    #     """
        
    #     log.info("Setting sky area to {:d} pixels (nside={:d})".format(pixels.shape[0], self.nside))
    #     mask              = self.data.isin({self.colname_pix: pixels}).any(1)
    #     self.selectedData = self.data[mask]
    #     log.info(" > kept {:d} galaxies".format(self.selectedData.shape[0]))
    #     log.info("Setting z range: {:.3f} < z < {:.3f}".format(z_min, z_max,3))
    #     self.selectedData = self.selectedData[(self.selectedData.z >= z_min) & (self.selectedData.z < z_max)]
    #     log.info(" > kept {:d} galaxies".format(self.selectedData.shape[0]))
        
    #     # self.N_gal_pix    = np.array([sum(gal.selectedData["pix"+str(NSIDE)] == pix) for pix in pix_conf])



    def group_correction(self, df, df_groups, which_z='z_cosmo'):
        """Corrects cosmological redshift in heliocentric frame for peculiar velocities in galaxies 
           inside the group galaxy catalogue in arXiv:1705.08068, table 2.

           TO BE APPLIED BEFORE CHANGING TO CMB FRAME.

           Corrects `which_z` column for peculiar velocities and add a new column named which_z+'_or'
           with the original redshift.

        Args:
            df (pd.dataframe): dataframe to correct
            df_groups (pd.dataframe): dataframe of group velocities
            which_z (str, optional): name of column to correct. Defaults to 'z_cosmo'.
        """
       
        df.loc[:, which_z+'_or'] = df[which_z].values
        zs                       = df.loc[df['PGC'].isin(df_groups['PGC'])][['PGC', which_z]]
        z_corr_arr               = []

        for PGC in zs.PGC.values:
            PGC1    = df_groups[df_groups['PGC']==PGC]['PGC1'].values[0]
            z_group = df_groups[df_groups['PGC1']== PGC1].HRV.mean()/glob.clight

            z_corr_arr.append(z_group)

        z_corr_arr = np.array(z_corr_arr)

        df.loc[df['PGC'].isin(df_groups['PGC']), which_z] = z_corr_arr
        correction_flag_array = np.where(df[which_z+'_or'] != df[which_z], 1, 0)
        df.loc[:, 'group_correction'] = correction_flag_array

    def CMB_correction(self, df, which_z='z_cosmo'):
        """Gives cosmological redshift in CMB frame starting from heliocentric.

        Corrects df,  with a new column given by which_z +'_CMB'.

        Args:
            df (pd.dataframe): dataframe to correct
            which_z (str, optional): name of column to correct. Defaults to 'z_cosmo'.
        """       

        v_gal            = glob.clight*df[which_z].values
        phi_CMB, dec_CMB = utils.gal_to_eq(np.radians(glob.l_CMB), np.radians(glob.b_CMB))
        theta_CMB        = 0.5 * np.pi - dec_CMB
        delV             = glob.v_CMB * (np.sin(df.theta)*np.sin(theta_CMB)*np.cos(df.phi-phi_CMB) +\
                                         np.cos(df.theta)*np.cos(theta_CMB))
        v_corr           = v_gal + delV  # at first order in v/c ...
        z_corr           = v_corr/glob.clight

        df.loc[:,which_z+'_CMB'] = z_corr
  
    def include_vol_prior(self, df):
        batchSize = 10000
        nBatches  = max(int(len(df)/batchSize), 1)
         
        log.info("Computing galaxy posteriors...")
          
        from software.CHIMERA.CHIMERA._keelin import convolve_bounded_keelin_3
        from astropy.cosmology import FlatLambdaCDM
        fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        zGrid = np.linspace(0, 1.4*np.max(df.z_upperbound), 500)
        jac = fiducialcosmo.comoving_distance(zGrid).value**2 / fiducialcosmo.H(zGrid).value
        
        from scipy import interpolate
        func = interpolate.interp1d(zGrid, jac, kind='cubic')
        
        def convolve_batch(df, func, batchId, nBatches): 
            N = len(df)
            # actual batch size, different from batchSize only due to integer rounding 
            n = int(N/nBatches) 
            start = n*batchId
            stop = n*(batchId+1)
            if batchId == nBatches-1:
                stop = N 

            batch = df.iloc[start:stop]

            if batchId % 100 == 0:
                log.info("Batch " + str(batchId) + " of " + str(nBatches) )
            
            ll = batch.z_lowerbound.to_numpy()
            l  = batch.z_lower.to_numpy()
            m  = batch.z.to_numpy()
            u  = batch.z_upper.to_numpy()
            uu = batch.z_upperbound.to_numpy()
               
            return convolve_bounded_keelin_3(func, 0.16, l, m, u, ll, uu, N=1000)

        res = np.vstack(utils.parmap(lambda b: convolve_batch(df, func, b, nBatches), range(nBatches)))
        
        mask = (res[:,0] >= res[:,1]) | (res[:,1] >= res[:,2]) | (res[:,2] >= res[:,3]) | (res[:,3] >= res[:,4]) | (res[:,0] < 0)
      
        log.info('Removing ' + str( np.sum(mask) ) + ' galaxies with unfeasible redshift pdf after r-squared prior correction.' )

        df.z_lowerbound = res[:, 0]
        df.z_lower = res[:, 1]
        df.z = res[:, 2]
        df.z_upper = res[:, 3]
        df.z_upperbound = res[:, 4]

        df = df[~mask]

        return df 




