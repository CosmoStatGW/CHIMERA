#
#   This module handles the computation of the overall likelihood.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import h5py, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from CHIMERA.GW import GW
from CHIMERA.DataEM import (MockGalaxiesMICEv2, GLADEPlus)
from CHIMERA.utils import (misc, plotting)


from jax import jit

    
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





class IndividualLikelihoods_old():

    def __init__(self, event_list, obs_run, conf, kind, savefig=False):

        self.event_list = event_list
        self.Nevents    = len(event_list)
        self.conf       = conf
        self.savefig    = savefig


        if kind == "BBH":
            self.dataGW = misc.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=True, SNR_th=11)
        
        elif kind == "other":
            self.dataGW = misc.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=False)
        else:
            print("ERROR")
    

    def load_gw(self):
        self.gw = GW(data=self.dataGW, 
                     data_names=self.event_list, 
                     data_smooth=self.conf["GW_dirBWs"],
                     model_mass=self.conf["mass_dist"],
                     model_rate=self.conf["rate_evol"],
                     model_spin="",
                     model_cosmo=self.conf["cosmo"],

                     nside_list = self.conf["NSIDE"],
                     sky_conf  = self.conf["GW_perc_conf_Omega"]
                     )

        # For plotting purposes
        # self.pGW70, _ = self.gw.like(self.conf["lambda_cosmo"], self.conf["lambda_mass"])

        self.z_limits = self.gw.compute_z_ranges(H0_prior_range=self.conf["H0_prior"], z_conf_range=self.conf["GW_perc_conf_z"])


    def load_cat(self, e=0):
        self.cat = GLADEPlus_v2(foldername  = self.conf["CAT_dir"], 
                            compl       = SkipCompleteness(), 
                            useDirac    = self.conf["CAT_dirac"],
                            catalogname = self.conf["CAT_name"],
                            groupname   = "Galaxy_Group_Catalogue.csv",
                            band        = self.conf["CAT_band"],
                            Lcut        = self.conf["CAT_Lcut"],
                            pixels      = self.gw.pix_conf_finite[e],
                            nside       = self.conf["NSIDE"],
                            z_range     = self.z_limits[e]
                            )



    def analysis(self):
        
        self.res = []

        # Event by event
        for e in range(self.Nevents):
            print("\n##################")
            print("################## {:s}".format(self.event_list[e]))
            print("##################")

            # Select Healpix pixels in the confidence region
            conf_pixels = self.gw.pix_conf[e]

            # Load galaxy catalog and completeness in this region
            gal_completed = GalCompleted(completionType=self.conf["COMPL_complType"])
            compl         = MaskCompleteness_v2(comovingDensityGoal=self.conf["COMPL_nbar"], zRes=30, nMasks=9)

            cat = GLADEPlus_v2(foldername  = self.conf["CAT_dir"], 
                               compl       = compl, 
                               useDirac    = self.conf["CAT_dirac"],
                               catalogname = self.conf["CAT_name"],
                               groupname   = "Galaxy_Group_Catalogue.csv",
                               band        = self.conf["CAT_band"],
                               Lcut        = self.conf["CAT_Lcut"],
                               pixels      = conf_pixels,
                               nside       = self.conf["NSIDE"][0],
                               z_range     = self.z_limits[e]
                               )

            gal_completed.add_cat(cat)

            z_grid      = np.linspace(*self.z_limits[e], 1000)

            # Galaxy catalog
            dict_gal    = cat.pixelize_region(conf_pixels)
            pCAT, pCATw = cat.compute_pCAT_pixelized(z_grid, dict_gal, do_post_norm=True)

            # Plot 2D+"coring"
            fig, ax     = plt.subplots(1,2,figsize=(13,5))
            fig.suptitle(self.event_list[e])
            plotting.plot_2Dcoring(ax, cat.data.ra, cat.data.dec, cat.data.z,
                                     conf_pixels,
                                     self.conf["NSIDE"],
                                     self.z_limits[e],
                                     nest          = False,
                                     do_boundaries = True,
                                     do_KDE        = False,
                                     do_pGW        = True,
                                     pGW_KDE       = self.gw.eventKDEs[e],
                                     do_zGal       = True,
                                     do_pCAT       = True,
                                     pCAT_x        = z_grid,
                                     pCAT_y        = pCAT,
                                     norm_pCAT     = False, # cfr with `do_post_norm`
                                     )
            if self.savefig: fig.savefig(self.conf["PLOT_dir"]+self.event_list[e]+"_0_2D.jpg", dpi=250)


            # Setup integration grids

            H0_grid  = np.linspace(*self.conf["H0_prior"], self.conf["H0_grid_points"])
            cmapH0   = plt.cm.cool(np.linspace(0.,.8,len(H0_grid)))
            like_H0  = np.zeros((len(H0_grid),len(conf_pixels)))

            # Setup pGW pCAT

            fig, ax = plt.subplots(2,1, sharex=True)
            fig.suptitle(self.event_list[e])
            ax[1].set_xlabel("z")
            ax[0].set_ylabel("$p_{GW}(z | H_0)$")
            ax[1].set_ylabel("$p_{GAL}(z | H_0)$")

            # H0 likelihood 

            for k in tqdm(range(len(H0_grid))):
                lambda_cosmo          = {"H0":H0_grid[k],"Om0":0.3}
                GW_KDEs, GW_KDE_norms = self.gw.like(lambda_cosmo, self.conf["lambda_mass"])
                GW_KDE                = GW_KDEs[e]
                GW_KDE_norm           = GW_KDE_norms[e]

                for ipix, pix in enumerate(conf_pixels):
                    
                    p_gw         = GW_KDE(np.vstack([z_grid, 
                                                     np.full_like(z_grid, self.gw.ra_conf[e][pix]), 
                                                     np.full_like(z_grid, self.gw.dec_conf[e][pix])]))
                    p_cat        = pCAT[ipix,:]

                    ax[0].plot(z_grid, p_gw, c=cmapH0[k], lw=0.5)
                    ax[1].plot(z_grid, p_cat, c=cmapH0[k], lw=0.5)

                    like_H0[k,ipix] = np.trapz(p_gw*p_cat,  z_grid)

            if self.savefig: fig.savefig(self.conf["PLOT_dir"]+self.event_list[e]+"_1_pGWpCAT.jpg", dpi=250)


        self.res.append(like_H0)

        print("Event done.\n")


    def save(self, dir_res):
        
        with h5py.File(dir_res+'0_results.hdf5', 'a') as f:
            for e in range(len(self.event_list)):
                if not self.event_list[e] in f.keys():
                    # Create a group for the array
                    group = f.create_group(self.event_list[e])
                    # Add the array to the group
                    group.create_dataset('likeH0', data=self.res[e])
                else:
                    print(self.event_list[e]+" already present in result file. Nothing done.")
