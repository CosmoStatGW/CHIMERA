#
#   This module handles the computation of the overall likelihood.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import numpy as np
# from jax import jit

from CHIMERA.GW import GW
from CHIMERA.DataEM import (MockGalaxiesMICEv2, GLADEPlus)
from CHIMERA.utils import (misc, plotting)

    
class MockLike():

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
                 pixelize = True,
                 data_GAL_weights = None,
                 check_Neff = True
                 ):

        self.pixelize = pixelize
        self.data_GAL_weights = data_GAL_weights
        self.check_Neff = check_Neff
        self.z_det_range = z_det_range

        # Initialize GW and Galaxies classes, load data and precompute redshift grids
        self.gw      = GW(data=data_GW, data_names=data_GW_names,  data_smooth=data_GW_smooth,
                          model_mass=model_mass, model_rate=model_rate, model_spin="", model_cosmo=model_cosmo, 
                          npix_event=npix_event, nside_list=nside_list, sky_conf=sky_conf, check_Neff=check_Neff)
        
        self.z_grids = self.gw.compute_z_grids(z_int_H0_prior, z_int_sigma, z_int_res)

        self.gal     = MockGalaxiesMICEv2(data_GAL_dir, z_err = data_GAL_zerr, nside = self.gw.nside)
        
        # Precompute p_gal and associated weights
        self.p_gal, self.N_gal = self.gal.precompute(self.gw.nside, self.gw.pix_conf, self.z_grids, self.gw.data_names, self.data_GAL_weights)

        self.p_gw        = []
        self.p_rate      = []
        self.p_z         = []
        self.like_allpix = []

        if self.pixelize is False:
            for e in range(self.gw.Nevents):
                self.p_gal = np.mean(self.p_gal[e], axis=1, keepdims=True)


    def compute_log(self, lambda_cosmo, lambda_mass, lambda_rate, return_full=False):

        log_like_event = np.empty(self.gw.Nevents)

        # Overall rate normalization
        z_det           = np.linspace(*self.z_det_range, 1000)
        p_rate_norm     = self.gw.model_rate(z_det, lambda_rate)/(1.+z_det)*self.gw.model_cosmo.dV_dz(z_det, lambda_cosmo)
        log_p_rate_norm = np.log(np.trapz(p_rate_norm, z_det, axis=0))


        if return_full:
            like_allpix_ev, p_gw_ev, p_rate_ev = [], [], []

        # c = chimeraUtils.Stopwatch()

        for e in range(self.gw.Nevents):

            log_p_gw   = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass, lambda_rate=None)
            # c("p_gw")
            log_p_rate = np.log(self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e]))#*self.gw.model_cosmo.dV_dz(self.z_grids[e], lambda_cosmo))
            # c("p_rate")
            log_p_gal  = np.log(self.p_gal[e])
            # c("p_gal")

            log_p_z    = (log_p_rate - log_p_rate_norm)[:,np.newaxis] + log_p_gal


            # \int dz, then \int dOmega
            log_like_event_pix = np.log(np.trapz(np.exp(log_p_gw+log_p_z), self.z_grids[e], axis=0))
            mask               = np.isnan(log_like_event_pix)
            log_like_event[e]  = np.logaddexp.reduce(log_like_event_pix[~mask], axis=0) - np.log(log_like_event_pix[~mask].shape[0])
            
            # np.nanmean(like_pix, axis=0)
        
        # log_like = np.logaddexp.reduce(log_like_event) - np.log(len(log_like_event))
        
        return log_like_event
    



    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis) /((~np.isnan(A))*weights).sum(axis=axis)

        like_event = np.empty(self.gw.Nevents)

        # Compute overall rate normalization given lambda_rate
        if self.z_det_range is None:
            p_rate_norm = 1.
        else:
            z_det       = np.linspace(*self.z_det_range, 1000)
            p_rate_norm = self.gw.model_rate(z_det, lambda_rate)/(1.+z_det)*self.gw.model_cosmo.dV_dz(z_det, lambda_cosmo)
            p_rate_norm = np.trapz(p_rate_norm, z_det, axis=0)

        for e in range(self.gw.Nevents):
            p_gw     = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass, lambda_rate=None)
            p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e])
            p_gal    = self.p_gal[e]

            if self.pixelize is False:
                p_gw  = np.mean(p_gw, axis=1, keepdims=True)
            
            p_z      = (p_rate/p_rate_norm)[:,np.newaxis] * p_gal

            like_pix      = np.trapz(p_gw*p_z, self.z_grids[e], axis=0)
            # like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)
            like_event[e] = np.nanmean(like_pix, axis=0)

            if inspect:
                self.p_gw.append([p_gw])
                self.p_rate.append([p_rate])
                self.p_z.append([p_z])
                self.like_allpix.append([like_pix])

        return like_event
    




# Temp

            # self.z_grids[e] = np.linspace(0.1,1.4,300)


            # p_rate  /= np.trapz(p_rate, self.z_grids[e], axis=0)


            # p_gal   /= np.trapz(p_gal, self.z_grids[e], axis=0)


            # p_z      = (p_rate)[:,np.newaxis] * p_gal


            # like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)


            # \int dOmega, then dz
            # l1 = nanaverage(p_gw*p_z, weights=self.p_gal_w[e], axis=1)
            # like_event[e] = np.trapz(l1, self.z_grids[e], axis=0)/beta






class LikeLVK():

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

                 # Parameters for pixelization
                 nside_list,
                 npix_event,
                 sky_conf,

                 # Parameters for integration in redshift
                 z_int_H0_prior,
                 z_int_sigma,
                 z_int_res,
                 z_det_range,

                 # Parameters for the galaxy catalog
                 Lcut,
                 band,
                 pixelize=True,
                 data_GAL_weights=None,

                 # Other
                 check_Neff = True
                 ):

        self.pixelize = pixelize
        self.data_GAL_weights = data_GAL_weights
        self.check_Neff = check_Neff
        self.z_det_range = z_det_range

        self.gw      = GW(data=data_GW, data_names=data_GW_names,  data_smooth=data_GW_smooth,
                          model_mass=model_mass, model_rate=model_rate, model_spin="", model_cosmo=model_cosmo, 
                          npix_event=npix_event, nside_list=nside_list, sky_conf=sky_conf, check_Neff=check_Neff)
        
        self.z_grids = self.gw.compute_z_grids(z_int_H0_prior, z_int_sigma, z_int_res)
        
        self.gal     = GLADEPlus(data_GAL_dir, nside = self.gw.nside, Lcut=Lcut, band=band)
        
        self.p_gal, self.N_gal = self.gal.precompute(self.gw.nside, self.gw.pix_conf, self.z_grids, self.gw.data_names, self.data_GAL_weights)

        self.p_gw        = []
        self.p_rate      = []
        self.p_z         = []
        self.like_allpix = []

        if self.pixelize is False:
            for e in range(self.gw.Nevents):
                self.p_gal = np.mean(self.p_gal[e], axis=1, keepdims=True)



    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, inspect=False):
        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis) /((~np.isnan(A))*weights).sum(axis=axis)

        like_event = np.empty(self.gw.Nevents)

        # Compute overall rate normalization given lambda_rate
        if self.z_det_range is None:
            p_rate_norm = 1.
        else:
            z_det       = np.linspace(*self.z_det_range, 1000)
            p_rate_norm = self.gw.model_rate(z_det, lambda_rate)/(1.+z_det)*self.gw.model_cosmo.dV_dz(z_det, lambda_cosmo)
            p_rate_norm = np.trapz(p_rate_norm, z_det, axis=0)


        for e in range(self.gw.Nevents):
            p_gw     = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass, lambda_rate=None)
            p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e])#*self.gw.model_cosmo.dV_dz(self.z_grids[e], lambda_cosmo)
            p_gal    = self.p_gal[e]


            if self.pixelize is False:
                p_gw  = np.mean(p_gw, axis=1, keepdims=True)
            

            # p_z      = (p_rate/p_rate_norm)[:,np.newaxis] * p_gal

            p_z      = (p_rate)[:,np.newaxis] * p_gal

            # p_z      = p_rate[:,np.newaxis] * p_gal
            # p_z     /= np.trapz(p_z, self.z_grids[e], axis=0)

            like_pix      = np.trapz(p_gw*p_z, self.z_grids[e], axis=0)
            # like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)
            like_event[e] = np.nanmean(like_pix, axis=0)

            if inspect:
                self.p_gw.append([p_gw])
                self.p_rate.append([p_rate])
                self.p_z.append([p_z])
                self.like_allpix.append([like_pix])

        
        return like_event

