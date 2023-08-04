#
#   This module handles the computation of the overall likelihood.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#
import pickle

from abc import ABC, abstractmethod

# import jax.numpy as np
import numpy as onp
import numpy as np
import healpy as hp

from CHIMERA.GW import GW
from CHIMERA.EM import UVC_distribution
from CHIMERA.DataEM import (MockGalaxiesMICEv2, GLADEPlus)
from CHIMERA.utils import (misc, plotting)
from CHIMERA.Completeness import SkipCompleteness

class Likelihood():
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

    def load(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for key in state:
            setattr(self, key, state[key])

    def save(self, filename):
        state = {attr: getattr(self, attr, None) for attr in self.attrs_store}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
    
    def get_p_z_norm(self, lambda_cosmo, lambda_rate):
        """Compute overall rate normalization given lambda_rate (returns 1. if self.z_det_range is None)

        Args:
            lambda_rate (_type_): _description_
            lambda_cosmo (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.z_det_range is None:
            return 1.
        zz       = np.linspace(*self.z_det_range, 1000)
        p_z_norm = self.gw.model_rate(zz, lambda_rate)/(1.+zz)*self.gw.model_cosmo.dV_dz(zz, lambda_cosmo)
        return np.trapz(p_z_norm, zz, axis=0)

    def get_fR(self, lambda_cosmo, norm=False):
        """Compute completeness fraction (kind of)

        Args:
            lambda_cosmo (_type_): _description_

        Returns:
            _type_: _description_
        """
        def _fR_integrand(zz, lambda_cosmo):
            return self.compl(zz)*np.array(self.model_cosmo.dV_dz(zz, lambda_cosmo))

        # res = quad(_fR_integrand, 0, 10, args=({"H0": 70, "Om0": 0.3}))[0]  # general
        res = float(self.model_cosmo.V(1.3, lambda_cosmo)) # MICEv2

        if norm: res /= self.gw.model_cosmo.V(20, lambda_cosmo)

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


            compl     = self.P_compl(z_grid)
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





class MockLike(Likelihood):

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
        self.data_GW_names    = [f"Mock_{i:02d}" for i in range(self.Nevents)] if data_GW_names is None else data_GW_names
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
            self.P_compl   = lambda z: np.ones_like(z)

            def _generic_interpolant(z):
                p_cat_int = np.where((z>z_range[0])&(z<z_range[1]), fLCDM.dV_dz(z, {"H0": 70, "Om0": 0.3}), 0)
                p_cat_int /= _fR({"H0": 70, "Om0": 0.3})
                return p_cat_int

            self.p_gal_bkg = _generic_interpolant



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
            self.gal = GLADEPlus(data_GAL_dir, nside = self.gw.nside, Lcut=Lcut, band=band)
            self.p_cat_all, self.ngal_pix = self.gal.precompute(self.nside, self.pix_conf, self.z_grids, self.data_GW_names, self.data_GAL_weights)
            self.gal.compute_completeness()
            self.P_compl   = self.gal.P_compl
            self.p_gal_bkg = self.gal.get_interpolant(data_GAL_int_dir)
            
        else:
            self.p_cat_all = [np.ones((self.z_int_res,self.npix_event[e])) for e in range(self.nevents)]        








    # def compute_log(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
    #     """Compute the log likelihood for the given hyperparameters.

    #     Args:
    #         lambda_cosmo (dict): cosmological hyperparameters
    #         lambda_mass (dict): mass hyperparameters
    #         lambda_rate (dict): rate hyperparameters
    #         inspect (bool, optional): append intermediate pdfs. Defaults to False.

    #     Returns:
    #         np.ndarray: log likelihood for each event
    #     """
    #     log_like_event = np.empty(self.gw.Nevents)

    #     # Overall rate normalization given lambda_rate
    #     if self.z_det_range is None:
    #         log_p_rate_norm = 0.
    #     else:
    #         z_det           = np.linspace(*self.z_det_range, 1000)
    #         log_p_rate_norm = self.gw.model_rate(z_det, lambda_rate) - np.log1p(z_det) + self.gw.model_cosmo.log_dV_dz(z_det, lambda_cosmo)
    #         log_p_rate_norm = np.log(np.trapz(np.exp(log_p_rate_norm), z_det, axis=0))

    #     # Compute like for each event
    #     for e in range(self.gw.Nevents):
    #         log_p_gw   = self.gw.compute_event_log(e, self.z_grids[e], lambda_cosmo, lambda_mass)
    #         log_p_rate = self.gw.model_rate(self.z_grids[e], lambda_rate) - np.log1p(self.z_grids[e])
    #         log_p_gal  = np.log(self.p_gal[e])

    #         if self.pixelize is False:
    #             log_p_gw = (np.logaddexp.reduce(log_p_gw, axis=1) - np.log(log_p_gw.shape[1]))[:, np.newaxis]

    #         log_p_z            = (log_p_rate - log_p_rate_norm)[:,np.newaxis] + log_p_gal

    #         log_like_pix       = np.log(np.trapz(np.exp(log_p_gw), self.z_grids[e], axis=0))
    #         log_like_event[e]  = np.logaddexp.reduce(log_like_pix, axis=0) - np.log(log_like_pix.shape[0])
            
    #         if inspect:
    #             self.p_gw_all.append([log_p_gw])
    #             self.p_rate_all.append([log_p_rate])
    #             self.p_z_all.append([log_p_z])
    #             self.like_pix_all.append([log_like_pix])

    #     return log_like_event
    



    # def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
    #     """Compute the likelihood for the given hyperparameters.

    #     Args:
    #         lambda_cosmo (dict): cosmological hyperparameters
    #         lambda_mass (dict): mass hyperparameters
    #         lambda_rate (dict): rate hyperparameters
    #         inspect (bool, optional): append intermediate pdfs. Defaults to False.

    #     Returns:
    #         np.ndarray: likelihood for each event
    #     """
    #     like_event = np.empty(self.gw.Nevents)

    #     # Compute overall rate normalization given lambda_rate
    #     if self.z_det_range is None:
    #         p_rate_norm = 1.
    #     else:
    #         z_det       = np.linspace(*self.z_det_range, 1000)
    #         p_rate_norm = self.gw.model_rate(z_det, lambda_rate)/(1.+z_det)*self.gw.model_cosmo.dV_dz(z_det, lambda_cosmo)
    #         p_rate_norm = np.trapz(p_rate_norm, z_det, axis=0)

    #     # Compute like for each event
    #     for e in range(self.gw.Nevents):
    #         z_grid   = np.array(self.z_grids[e])
    #         p_gal    = np.array(self.p_gal_all[e])
    #         p_gw     = self.gw.compute_event(e, z_grid, lambda_cosmo, lambda_mass)

    #         # p_gal[z_grid>1.3] = 1.

    #         if self.data_GAL_dir is None:
    #             p_rate   = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)*self.gw.model_cosmo.dV_dz(z_grid, lambda_cosmo)
    #         else:
    #             p_rate   = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)#*self.gw.model_cosmo.dV_dz(z_grid, lambda_cosmo)


    #         if self.pixelize is False: 
    #             p_gw = np.mean(p_gw, axis=1, keepdims=True)
            
    #         p_z  = (p_rate/p_rate_norm)[:,np.newaxis] * p_gal

    #         # p_z  /= np.trapz(p_z, self.z_grids[e], axis=0)
    #         # p_gal /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega
    #         # dOmega = hp.pixelfunc.nside2pixarea(self.gw.nside[e],degrees=False)

    #         like_pix      = np.trapz(p_gw*p_z, z_grid, axis=0)
    #         like_event[e] = np.nanmean(like_pix, axis=0) #/dOmega

    #         if inspect:
    #             self.p_gw_all.append([p_gw])
    #             self.p_rate_all.append([p_rate])
    #             self.p_z_all.append([p_z])
    #             self.like_pix_all.append([like_pix])

    #     return like_event
    




# Temp

            # self.z_grids[e] = np.linspace(0.1,1.4,300)


            # p_rate  /= np.trapz(p_rate, self.z_grids[e], axis=0)


            # p_gal   /= np.trapz(p_gal, self.z_grids[e], axis=0)


            # p_z      = (p_rate)[:,np.newaxis] * p_gal


            # like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)


            # \int dOmega, then dz
            # l1 = nanaverage(p_gw*p_z, weights=self.p_gal_w[e], axis=1)
            # like_event[e] = np.trapz(l1, self.z_grids[e], axis=0)/beta