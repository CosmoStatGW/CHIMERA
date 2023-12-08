#
#   This module handles the computation of the overall likelihood.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#
import pickle

from abc import ABC, abstractmethod

import numpy as onp
import numpy as np
import healpy as hp

from CHIMERA.GW import GW
from CHIMERA.EM import UVC_distribution
from CHIMERA.DataEM import (MockGalaxiesMICEv2, GLADEPlus)
from CHIMERA.utils import (misc, plotting)
from CHIMERA.Completeness import EmptyCatalog, MaskCompleteness, CompletenessMICEv2

class Likelihood():
    # Generate docstring for the class
    """ Abstract class to handle likelihood (numerator) operations.    
    """


    attrs_baseline = ["model_cosmo", "model_mass", "model_rate", 
                      "data_GW_smooth", "data_GAL_dir", "data_GAL_weights", "data_GAL_int_dir",
                     "nside_list", "npix_event", "sky_conf", 
                     "z_int_H0_prior", "z_int_sigma", "z_int_res", "z_det_range", 'neff_data_min']

    attrs_basesave = ["model_cosmo_name", "model_mass_name", "model_rate_name", 
                     "data_GW_names", "data_GW_smooth", "data_GAL_dir", "data_GAL_int_dir",
                     "nside_list", "npix_event", "sky_conf", 
                     "z_int_H0_prior", "z_int_sigma", "z_int_res", "z_det_range", 'neff_data_min']

    attrs_compute  = ['nside', 'pix_conf', 'npix_event', 'z_grids',
                      'p_gw_all', 'p_gal_all', 'p_cat_all', 'compl_all', 'p_rate_all', 'p_z_all', 
                      'like_pix_all']

    attrs_mock     = ["data_GAL_zerr"]

    attrs_galcat   = ["Lcut", "band"]
    

    def __init__(self):
        for attr_name in self.attrs_baseline:
            setattr(self, attr_name, None)
        for attr_name in self.attrs_compute:
            setattr(self, attr_name, [])

    def save(self, filename):
        """Save the likelihood state to a file."""
        state = {attr: getattr(self, attr, None) for attr in self.attrs_store}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filename):
        """Load the likelihood state from a file."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for key in state:
            setattr(self, key, state[key])

    def _get_p_z_norm(self, lambda_cosmo, lambda_rate):
        """Compute overall rate normalization given lambda_rate (returns 1. if self.z_det_range is None)
        """
        if self.z_det_range is None:
            return 1.
        zz       = np.linspace(*self.z_det_range, 1000)
        p_z_norm = self.gw.model_rate(zz, lambda_rate)/(1.+zz)*self.gw.model_cosmo.dV_dz(zz, lambda_cosmo)
        return np.trapz(p_z_norm, zz, axis=0)
    
        
    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        """Method to compute the likelihood for the given hyperparameters.
        
        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters
            
        Returns:
            np.ndarray: likelihood for each event
        """
        pass




class MockLike(Likelihood):

    """Class to handle likelihood (numerator) operations for mock catalogs.
    
    Args:
        model_cosmo (CHIMERA.cosmo): :class:`CHIMERA.cosmo` object
        model_mass (CHIMERA.mass): :class:`CHIMERA.mass.mass` object
        model_rate (CHIMERA.rate): :class:`CHIMERA.rate.rate` object
        data_GW (dict): dictionary with GW data
        data_GW_names (list): list of names of the events
        data_GW_smooth (bool): whether to smooth the GW data or not
        data_GAL_dir (str): path to the galaxy catalog
        data_GAL_int_dir (str): path to the galaxy catalog interpolant
        data_GAL_zerr (float): redshift error for the galaxy catalog
        nside_list (list): list of nside for the pixelization
        npix_event (list): list of number of pixels for each event
        sky_conf (list): list of sky configurations for each event
        z_int_H0_prior (float): H0 prior for the redshift integration
        z_int_sigma (float): sigma for the redshift integration
        z_int_res (int): resolution for the redshift integration
        z_det_range (list): redshift range for the detection
        data_GAL_weights (np.ndarray): weights for the galaxy catalog
        neff_data_min (int): minimum number of galaxies for the galaxy catalog    
    """

    def __init__(self, 
                 
                 # Models
                 model_cosmo, 
                 model_mass, 
                 model_rate,

                 # Data
                 data_GW,
                 data_GW_names,
                 data_GW_smooth,
                 data_GAL_dir,
                 data_GAL_int_dir,
                 data_GAL_zerr,

                 # Parameters for pixelization
                 nside_list,
                 npix_event,
                 sky_conf,

                 # Parameters for integration in z
                 z_int_H0_prior,
                 z_int_sigma,
                 z_int_res,
                 z_det_range,

                 # Other parameters
                 data_GAL_weights = None,
                 neff_data_min    = 5,
                 ):

        super().__init__()

        # Initialize all the parameters
        self.model_cosmo      = model_cosmo
        self.model_mass       = model_mass
        self.model_rate       = model_rate
        self.model_cosmo_name = model_cosmo.__name__
        self.model_mass_name  = model_mass.__name__
        self.model_rate_name  = model_rate.__name__

        self.nevents          = data_GW["dL"].shape[0]
        self.data_GW_names    = [f"Mock_{i:02d}" for i in range(self.nevents)] if data_GW_names is None else data_GW_names
        self.data_GW_smooth   = data_GW_smooth
        self.data_GAL_dir     = data_GAL_dir
        self.data_GAL_int_dir = data_GAL_int_dir

        self.data_GAL_weights = data_GAL_weights
        self.data_GAL_zerr    = data_GAL_zerr

        self.nside_list       = nside_list
        self.npix_event       = npix_event
        self.sky_conf         = sky_conf

        self.z_int_H0_prior   = z_int_H0_prior
        self.z_int_sigma      = z_int_sigma
        self.z_int_res        = z_int_res
        self.z_det_range      = z_det_range
        self.neff_data_min    = neff_data_min

        self.attrs_store      = self.attrs_basesave + self.attrs_mock + self.attrs_compute

        # Initialize GW class, then precompute pixels and redshift grids
        self.gw         = GW(data=data_GW, data_names=self.data_GW_names, data_smooth=self.data_GW_smooth,
                             model_mass=self.model_mass, model_rate=self.model_rate, model_spin="", model_cosmo=self.model_cosmo, 
                             npix_event=self.npix_event, nside_list=self.nside_list, sky_conf=self.sky_conf, 
                             data_Neff=self.neff_data_min)

        self.nside      = self.gw.nside
        self.pix_conf   = self.gw.pix_conf
        self.npix_event = self.gw.npix_event
        self.z_grids    = self.gw.compute_z_grids(self.z_int_H0_prior, self.z_int_sigma, self.z_int_res)

        # Load galaxy catalog, precompute p_gal and p_bkg
        if self.data_GAL_dir is not None:
            self.gal = MockGalaxiesMICEv2(self.data_GAL_dir, z_err = self.data_GAL_zerr, nside = self.nside)
            self.p_cat_all, self.ngal_pix = self.gal.precompute(self.nside, self.pix_conf, self.z_grids, self.data_GW_names, self.data_GAL_weights)
            self.gal.compute_completeness()
            self.P_compl   = self.gal.P_compl
            self.p_gal_bkg = self.gal.get_interpolant(data_GAL_int_dir)
            
        else:
            self.p_cat_all = [np.ones((self.z_int_res,self.npix_event[e])) for e in range(self.nevents)]        
            self.P_compl   = lambda z: np.zeros_like(z)

            # def _generic_interpolant(z):
            #     p_cat_int = np.where((z>z_range[0])&(z<z_range[1]), fLCDM.dV_dz(z, {"H0": 70, "Om0": 0.3}), 0)
            #     p_cat_int /= _fR({"H0": 70, "Om0": 0.3})
            #     return p_cat_int

            # self.p_gal_bkg = _generic_interpolant


    def get_fR(self, lambda_cosmo, normalized=False):
        norm = float(self.model_cosmo.V(20, lambda_cosmo)) if normalized else 1.
        return  float(self.model_cosmo.V(1.3, lambda_cosmo))/norm
    

    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        """Compute the likelihood for the given hyperparameters.

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            np.ndarray: likelihood for each event
        """
        like_events = np.empty(self.nevents)

        # Compute overall rate normalization given lambda_rate (returns 1. if self.z_det_range is None)
        p_z_norm = self._get_p_z_norm(lambda_cosmo, lambda_rate)

        # Compute completeness fraction (kind of)
        fR       = self.get_fR(lambda_cosmo)

        # Compute like for each event
        for e in range(self.nevents):
            z_grid    = self.z_grids[e]
            p_cat     = self.p_cat_all[e]

            # OLD (before 2021-05-18)
            # compl     = self.compl(z_grid)[:,np.newaxis]   
            # p_cat_vol = np.array(self.model_cosmo.dV_dz(z_grid, {"H0":70, "Om0":0.3}))[:,np.newaxis]
            # norm1     = np.trapz(p_cat, z_grid, axis=0)
            # norm2     = np.trapz(p_cat_vol, z_grid, axis=0)
            # p_gal     = (compl * (norm2/norm1)*p_cat/p_cat_vol + (1.-compl))*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo))[:,np.newaxis]


            
            

            if self.data_GAL_dir is None:
                p_gal     = np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo))[:,np.newaxis]
            else:
                compl     = self.P_compl(z_grid, 0, 0)
                p_gal     = fR * p_cat + ( (1.-compl)*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo)) )[:,np.newaxis]
        
            # p_gal    *= (self.npix_event[e]*hp.pixelfunc.nside2pixarea(self.nside[e],degrees=False)) # for the integral in dOmega

            p_rate    = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)
            p_z       = (p_rate/p_z_norm)[:,np.newaxis] * p_gal
            p_gw      = self.gw.compute_event(e, z_grid, lambda_cosmo, lambda_mass)
            like_pix = np.trapz(p_gw*p_z, z_grid, axis=0)

            # p_z  /= np.trapz(p_z, self.z_grids[e], axis=0)
            # p_gal /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega

            like_events[e] = np.nansum(like_pix, axis=0) / hp.pixelfunc.nside2npix(self.nside[e])

            if inspect:
                self.p_gw_all.append(onp.array(p_gw))
                self.p_gal_all.append(onp.array(p_gal))
                self.compl_all.append(onp.array(compl))
                self.p_rate_all.append(onp.array(p_rate))
                self.p_z_all.append(onp.array(p_z))
                self.like_pix_all.append(onp.array(like_pix))

        return onp.array(like_events)





class LikeLVK(Likelihood):

    def __init__(self, 
                 
                 # Models
                 model_cosmo, 
                 model_mass, 
                 model_rate,

                 # Data
                 data_GW,
                 data_GW_names,
                 data_GW_smooth,
                 data_GAL_dir,
                 data_GAL_int_dir,
                 completeness,

                 # Parameters for pixelization
                 nside_list,
                 npix_event,
                 sky_conf,

                 # Parameters for integration in z
                 z_int_H0_prior,
                 z_int_sigma,
                 z_int_res,
                 z_det_range,

                 # Parameters for the galaxy catalog
                 Lcut,
                 band,
                 data_GAL_weights=None,

                 # Other parameters
                 neff_data_min = 5,
                 ):


        super().__init__()

        # Initialize all the parameters
        self.model_cosmo      = model_cosmo
        self.model_mass       = model_mass
        self.model_rate       = model_rate
        self.model_cosmo_name = model_cosmo.__name__
        self.model_mass_name  = model_mass.__name__
        self.model_rate_name  = model_rate.__name__

        self.nevents          = data_GW["dL"].shape[0]
        self.data_GW_names    = data_GW_names
        self.data_GW_smooth   = data_GW_smooth
        self.data_GAL_dir     = data_GAL_dir
        self.data_GAL_int_dir = data_GAL_int_dir
        self.data_GAL_weights = data_GAL_weights
        self.band             = band
        self.Lcut             = Lcut 
        self.completeness     = completeness

        self.nside_list       = nside_list
        self.npix_event       = npix_event
        self.sky_conf         = sky_conf

        self.z_int_H0_prior   = z_int_H0_prior
        self.z_int_sigma      = z_int_sigma
        self.z_int_res        = z_int_res
        self.z_det_range      = z_det_range
        self.neff_data_min    = neff_data_min

        self.attrs_store      = self.attrs_basesave + self.attrs_mock + self.attrs_compute

        # Initialize GW class, then precompute pixels and redshift grids
        self.gw         = GW(data=data_GW, data_names=self.data_GW_names, data_smooth=self.data_GW_smooth,
                             model_mass=self.model_mass, model_rate=self.model_rate, model_spin="", model_cosmo=self.model_cosmo, 
                             npix_event=self.npix_event, nside_list=self.nside_list, sky_conf=self.sky_conf, 
                             data_Neff=self.neff_data_min)

        self.nside      = self.gw.nside
        self.pix_conf   = self.gw.pix_conf
        self.npix_event = self.gw.npix_event
        self.z_grids    = self.gw.compute_z_grids(self.z_int_H0_prior, self.z_int_sigma, self.z_int_res)

        # Load galaxy catalog, precompute p_gal, P_compl, and p_bkg
        if self.data_GAL_dir is not None:
            self.gal = GLADEPlus(data_GAL_dir, nside = self.gw.nside, Lcut=Lcut, band=band)
            self.p_cat_all, self.ngal_pix = self.gal.precompute(self.nside, self.pix_conf, self.z_grids, self.data_GW_names, self.data_GAL_weights)
            
            # if completeness str load
            if isinstance(self.completeness, str):
                dir_compl = self.completeness
                self.completeness = MaskCompleteness(N_z_bins=30, N_masks=4, compl_goal=0.01) # CompletenessMICEv2(z_range = [0., 0.12])
                self.completeness.load(dir_compl)


                dir_avg_compl = "/home/debian/software/CHIMERA/data/compl_GLADEp_avg.pkl"
                self.avg_completeness =  MaskCompleteness(N_z_bins=30, N_masks=4, compl_goal=0.01) #dull
                self.avg_completeness.load(dir_avg_compl)

                print("Using completeness from "+dir_compl)
                print("Using average completeness from "+dir_avg_compl)

                P_compl_avg = lambda z : self.avg_completeness.P_compl(z, np.array([0]), 32)
                # fR_avg = lambda ll : self.model.cosmo.V(0.12, ll) 
                fR_avg = lambda ll : self.avg_completeness.get_fR(ll, z_det_range=[0,.12])
            
                def _p_bkg_fcn(z, lambda_cosmo):
                    return fR_avg(lambda_cosmo)*p_cat_int(z) + (1-P_compl_avg(z).T)*np.array(self.model_cosmo.dV_dz(z, lambda_cosmo))  

                self.p_gal_bkg = _p_bkg_fcn

            else:
                # print("APPROX COMPL")
                with open(data_GAL_int_dir, "rb") as f:
                    p_cat_int = pickle.load(f)

                def P_compl(z, z_range=[0,0.12]):
                    """Return 1 if z is in z_range, 0 otherwise."""
                    return np.where(np.logical_and(z>z_range[0], z<z_range[1]), 1., 0.)

                def _fR(lambda_cosmo):
                    return float(self.model_cosmo.V(0.12, lambda_cosmo))

                def _p_bkg_fcn(z, lambda_cosmo):
                    pn = p_cat_int(z) / np.trapz(p_cat_int(z), z)
                    return self._fR(lambda_cosmo)*pn + (1-P_compl(z))*np.array(self.model_cosmo.dV_dz(z, lambda_cosmo))  


                self.P_compl   = P_compl
                self.p_gal_bkg = _p_bkg_fcn

            # elif self.completeness is None:
            #     print("No completeness given, assuming completeness = 1")
            #     self.completeness = None
            # else:
            #     self.completeness.compute(self.gal.data)
            #     self.avg_completeness =  EmptyCatalog() #dull

                # self.
            # if interpolant


            
            # P_compl_avg = lambda z : self.avg_completeness.P_compl(z, np.array([0]), 32)
            # fR_avg = lambda ll : self.model.cosmo.V(0.12, ll) 
            
            # self.avg_completeness.get_fR(ll, z_det_range=[0,1.3])
            
            # def _p_bkg_fcn(z, lambda_cosmo):
                # return fR_avg(lambda_cosmo)*p_cat_int(z) + (1-P_compl_avg(z).T)*np.array(self.model_cosmo.dV_dz(z, lambda_cosmo))  

            # self.p_gal_bkg = _p_bkg_fcn

        else:
            self.p_cat_all = [np.ones((self.z_int_res,self.npix_event[e])) for e in range(self.nevents)]        

    def get_fR(self, lambda_cosmo, norm=False):
        res = float(self.model_cosmo.V(0.12, lambda_cosmo))
        return res 

    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        """Compute the likelihood for the given hyperparameters.

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            np.ndarray: likelihood for each event
        """
        like_events = np.empty(self.nevents)

        # Compute overall rate normalization given lambda_rate (returns 1. if self.z_det_range is None)
        p_z_norm = self.get_p_z_norm(lambda_cosmo, lambda_rate)
        fR       = self.completeness.get_fR(lambda_cosmo, z_det_range=[0,0.12])

        # fR_ev = self.get_fR(lambda_cosmo)

        # Compute like for each event
        for e in range(self.nevents):
            z_grid    = self.z_grids[e]
            p_cat     = self.p_cat_all[e]

            idxs_mask = self.completeness.get_pix2mask(self.pix_conf[e], self.nside[e])
            fR_ev     = fR[idxs_mask]
            compl     = self.completeness.P_compl(z_grid, self.pix_conf[e], self.nside[e])

            # compl     = self.P_compl(z_grid)

            p_gal     = fR_ev * p_cat + ( (1.-compl)*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo)[:, np.newaxis] ))

            # p_gal     = fR_ev * p_cat + ( (1.-compl)*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo)))[:, np.newaxis]

            if 0:
            # print(fR_ev * p_cat )
                for pix in range(self.npix_event[e]):
                    zz         = np.linspace(0, 10, 1000)
                    compl      = self.P_compl(z_grid, self.pix_conf[e][pix], self.nside[e])
                    # print(compl)
                    fR         = np.trapz(self.P_compl(zz, self.pix_conf[e][pix], self.nside[e])*np.array(self.model_cosmo.dV_dz(zz, lambda_cosmo)), zz)
                    p_gal[:,pix] = fR * p_cat[:,pix] + (1.-compl)*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo)) 


            p_rate    = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)
            p_z       = (p_rate/p_z_norm)[:,np.newaxis] * p_gal
            p_gw      = self.gw.compute_event(e, z_grid, lambda_cosmo, lambda_mass)
            like_pix = np.trapz(p_gw*p_z, z_grid, axis=0)


            like_events[e] = np.nansum(like_pix, axis=0) / hp.pixelfunc.nside2npix(self.nside[e])

            if inspect:
                self.p_gw_all.append(onp.array(p_gw))
                self.p_gal_all.append(onp.array(p_gal))
                self.compl_all.append(onp.array(compl))
                self.p_rate_all.append(onp.array(p_rate))
                self.p_z_all.append(onp.array(p_z))
                self.like_pix_all.append(onp.array(like_pix))

        return onp.array(like_events)



    def compute_complete(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        """Compute the likelihood for the given hyperparameters.

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            np.ndarray: likelihood for each event
        """
        like_events = np.empty(self.nevents)

        # Compute overall rate normalization given lambda_rate (returns 1. if self.z_det_range is None)
        p_z_norm = self._get_p_z_norm(lambda_cosmo, lambda_rate)

        # Compute completeness fraction (kind of)


        # Compute like for each event
        for e in range(self.nevents):
            z_grid    = self.z_grids[e]
            p_cat     = self.p_cat_all[e]
            compl     = None
            # OLD (before 2021-05-18)
            # compl     = self.compl(z_grid)[:,np.newaxis]   
            # p_cat_vol = np.array(self.model_cosmo.dV_dz(z_grid, {"H0":70, "Om0":0.3}))[:,np.newaxis]
            # norm1     = np.trapz(p_cat, z_grid, axis=0)
            # norm2     = np.trapz(p_cat_vol, z_grid, axis=0)
            # p_gal     = (compl * (norm2/norm1)*p_cat/p_cat_vol + (1.-compl))*np.array(self.model_cosmo.dV_dz(z_grid, lambda_cosmo))[:,np.newaxis]


            p_gal     = p_cat*np.array(self.model_cosmo.V(20, lambda_cosmo))
            # p_gal    *= (self.npix_event[e]*hp.pixelfunc.nside2pixarea(self.nside[e],degrees=False)) # for the integral in dOmega

            p_rate    = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)
            p_z       = (p_rate/p_z_norm)[:,np.newaxis] * p_gal
            p_gw      = self.gw.compute_event(e, z_grid, lambda_cosmo, lambda_mass)
            like_pix = np.trapz(p_gw*p_z, z_grid, axis=0)

            # p_z  /= np.trapz(p_z, self.z_grids[e], axis=0)
            # p_gal /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega

            like_events[e] = np.nansum(like_pix, axis=0) / hp.pixelfunc.nside2npix(self.nside[e])

            if inspect:
                self.p_gw_all.append(onp.array(p_gw))
                self.p_gal_all.append(onp.array(p_gal))
                self.compl_all.append(onp.array(compl))
                self.p_rate_all.append(onp.array(p_rate))
                self.p_z_all.append(onp.array(p_z))
                self.like_pix_all.append(onp.array(like_pix))

        return onp.array(like_events)
