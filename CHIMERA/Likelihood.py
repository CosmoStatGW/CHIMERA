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


class Likelihood():
    attrs_baseline = ["model_cosmo", "model_mass", "model_rate", 
                      "data_GW_smooth", "data_GAL_dir", "data_GAL_weights"
                     "nside_list", "npix_event", "sky_conf", 
                     "z_int_H0_prior", "z_int_sigma", "z_int_res", "z_det_range", 'neff_data_min']

    attrs_basesave = ["model_cosmo_name", "model_mass_name", "model_rate_name", 
                     "data_GW_names", "data_GW_smooth", "data_GAL_dir",
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

        # Compute like for each event
        for e in range(self.nevents):
            z_grid    = self.z_grids[e]
            p_cat     = self.p_cat_all[e]
            p_cat_vol = np.array(self.gw.model_cosmo.dV_dz(z_grid, {"H0":70, "Om0":0.3}))[:,np.newaxis]

            norm1     = np.trapz(p_cat, z_grid, axis=0)
            norm2     = np.trapz(p_cat_vol, z_grid, axis=0)

            compl     = np.where(z_grid<1.3, 1., 0.)[:,np.newaxis]

            p_gal     = (compl * (norm2/norm1)*p_cat/p_cat_vol + (1.-compl))#*np.array(self.gw.model_cosmo.dV_dz(z_grid, lambda_cosmo))[:,np.newaxis]
            p_rate    = self.gw.model_rate(z_grid, lambda_rate)/(1.+z_grid)
            p_z       = (p_rate/p_z_norm)[:,np.newaxis] * p_gal

            p_gw      = self.gw.compute_event(e, z_grid, lambda_cosmo, lambda_mass)
            
            like_pix = np.trapz(p_gw*p_z, z_grid, axis=0)

            # p_z  /= np.trapz(p_z, self.z_grids[e], axis=0)
            # p_gal /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega
            # dOmega = hp.pixelfunc.nside2pixarea(self.gw.nside[e],degrees=False)

            like_events[e] = np.nanmean(like_pix, axis=0) #/dOmega

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
        self.gw         = GW(data=data_GW, data_names=self.data_GW_names,  data_smooth=self.data_GW_smooth,
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
            self.p_bkg = self.gal.p_bkg()
        else:
            self.p_cat_all = [np.ones((self.z_int_res,self.npix_event[e])) for e in range(self.nevents)]        
        


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
                 data_GAL_weights=None,

                 # Other parameters
                 neff_data_min = True
                 ):

        self.pixelize = pixelize
        self.data_GAL_weights = data_GAL_weights
        self.check_Neff = check_Neff
        self.z_det_range = z_det_range

        self.gw      = GW(data=data_GW, data_names=data_GW_names,  data_smooth=data_GW_smooth,
                          model_mass=model_mass, model_rate=model_rate, model_spin="", model_cosmo=model_cosmo, 
                          npix_event=npix_event, nside_list=nside_list, sky_conf=sky_conf, data_Neff=neff_data_min)

        self.nside      = self.gw.nside
        self.pix_conf   = self.gw.pix_conf
        self.npix_event = np.array([len(a) for a in self.pix_conf])
        self.z_grids = self.gw.compute_z_grids(z_int_H0_prior, z_int_sigma, z_int_res)
        
        self.gal     = GLADEPlus(data_GAL_dir, nside = self.gw.nside, Lcut=Lcut, band=band)
        
        self.p_cat_all, self.ngal_pix = self.gal.precompute(self.nside, self.pix_conf, self.z_grids, self.data_names, self.data_GAL_weights)

        # if self.pixelize is False:
        #     for e in range(self.gw.Nevents):
        #         self.p_gal = np.mean(self.p_gal[e], axis=1, keepdims=True)





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






class LikeLVKOld():

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





# class IndividualLikelihoods_old():

#     def __init__(self, event_list, obs_run, conf, kind, savefig=False):

#         self.event_list = event_list
#         self.Nevents    = len(event_list)
#         self.conf       = conf
#         self.savefig    = savefig


#         if kind == "BBH":
#             self.dataGW = misc.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=True, SNR_th=11)
        
#         elif kind == "other":
#             self.dataGW = misc.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=False)
#         else:
#             print("ERROR")
    

#     def load_gw(self):
#         self.gw = GW(data=self.dataGW, 
#                      data_names=self.event_list, 
#                      data_smooth=self.conf["GW_dirBWs"],
#                      model_mass=self.conf["mass_dist"],
#                      model_rate=self.conf["rate_evol"],
#                      model_spin="",
#                      model_cosmo=self.conf["cosmo"],

#                      nside_list = self.conf["NSIDE"],
#                      sky_conf  = self.conf["GW_perc_conf_Omega"]
#                      )

#         # For plotting purposes
#         # self.pGW70, _ = self.gw.like(self.conf["lambda_cosmo"], self.conf["lambda_mass"])

#         self.z_limits = self.gw.compute_z_ranges(H0_prior_range=self.conf["H0_prior"], z_conf_range=self.conf["GW_perc_conf_z"])


#     def load_cat(self, e=0):
#         self.cat = GLADEPlus_v2(foldername  = self.conf["CAT_dir"], 
#                             compl       = SkipCompleteness(), 
#                             useDirac    = self.conf["CAT_dirac"],
#                             catalogname = self.conf["CAT_name"],
#                             groupname   = "Galaxy_Group_Catalogue.csv",
#                             band        = self.conf["CAT_band"],
#                             Lcut        = self.conf["CAT_Lcut"],
#                             pixels      = self.gw.pix_conf_finite[e],
#                             nside       = self.conf["NSIDE"],
#                             z_range     = self.z_limits[e]
#                             )



#     def analysis(self):
        
#         self.res = []

#         # Event by event
#         for e in range(self.Nevents):
#             print("\n##################")
#             print("################## {:s}".format(self.event_list[e]))
#             print("##################")

#             # Select Healpix pixels in the confidence region
#             conf_pixels = self.gw.pix_conf[e]

#             # Load galaxy catalog and completeness in this region
#             gal_completed = GalCompleted(completionType=self.conf["COMPL_complType"])
#             compl         = MaskCompleteness_v2(comovingDensityGoal=self.conf["COMPL_nbar"], zRes=30, nMasks=9)

#             cat = GLADEPlus_v2(foldername  = self.conf["CAT_dir"], 
#                                compl       = compl, 
#                                useDirac    = self.conf["CAT_dirac"],
#                                catalogname = self.conf["CAT_name"],
#                                groupname   = "Galaxy_Group_Catalogue.csv",
#                                band        = self.conf["CAT_band"],
#                                Lcut        = self.conf["CAT_Lcut"],
#                                pixels      = conf_pixels,
#                                nside       = self.conf["NSIDE"][0],
#                                z_range     = self.z_limits[e]
#                                )

#             gal_completed.add_cat(cat)

#             z_grid      = np.linspace(*self.z_limits[e], 1000)

#             # Galaxy catalog
#             dict_gal    = cat.pixelize_region(conf_pixels)
#             pCAT, pCATw = cat.compute_pCAT_pixelized(z_grid, dict_gal, do_post_norm=True)

#             # Plot 2D+"coring"
#             fig, ax     = plt.subplots(1,2,figsize=(13,5))
#             fig.suptitle(self.event_list[e])
#             plotting.plot_2Dcoring(ax, cat.data.ra, cat.data.dec, cat.data.z,
#                                      conf_pixels,
#                                      self.conf["NSIDE"],
#                                      self.z_limits[e],
#                                      nest          = False,
#                                      do_boundaries = True,
#                                      do_KDE        = False,
#                                      do_pGW        = True,
#                                      pGW_KDE       = self.gw.eventKDEs[e],
#                                      do_zGal       = True,
#                                      do_pCAT       = True,
#                                      pCAT_x        = z_grid,
#                                      pCAT_y        = pCAT,
#                                      norm_pCAT     = False, # cfr with `do_post_norm`
#                                      )
#             if self.savefig: fig.savefig(self.conf["PLOT_dir"]+self.event_list[e]+"_0_2D.jpg", dpi=250)


#             # Setup integration grids

#             H0_grid  = np.linspace(*self.conf["H0_prior"], self.conf["H0_grid_points"])
#             cmapH0   = plt.cm.cool(np.linspace(0.,.8,len(H0_grid)))
#             like_H0  = np.zeros((len(H0_grid),len(conf_pixels)))

#             # Setup pGW pCAT

#             fig, ax = plt.subplots(2,1, sharex=True)
#             fig.suptitle(self.event_list[e])
#             ax[1].set_xlabel("z")
#             ax[0].set_ylabel("$p_{GW}(z | H_0)$")
#             ax[1].set_ylabel("$p_{GAL}(z | H_0)$")

#             # H0 likelihood 

#             for k in tqdm(range(len(H0_grid))):
#                 lambda_cosmo          = {"H0":H0_grid[k],"Om0":0.3}
#                 GW_KDEs, GW_KDE_norms = self.gw.like(lambda_cosmo, self.conf["lambda_mass"])
#                 GW_KDE                = GW_KDEs[e]
#                 GW_KDE_norm           = GW_KDE_norms[e]

#                 for ipix, pix in enumerate(conf_pixels):
                    
#                     p_gw         = GW_KDE(np.vstack([z_grid, 
#                                                      np.full_like(z_grid, self.gw.ra_conf[e][pix]), 
#                                                      np.full_like(z_grid, self.gw.dec_conf[e][pix])]))
#                     p_cat        = pCAT[ipix,:]

#                     ax[0].plot(z_grid, p_gw, c=cmapH0[k], lw=0.5)
#                     ax[1].plot(z_grid, p_cat, c=cmapH0[k], lw=0.5)

#                     like_H0[k,ipix] = np.trapz(p_gw*p_cat,  z_grid)

#             if self.savefig: fig.savefig(self.conf["PLOT_dir"]+self.event_list[e]+"_1_pGWpCAT.jpg", dpi=250)


#         self.res.append(like_H0)

#         print("Event done.\n")


#     def save(self, dir_res):
        
#         with h5py.File(dir_res+'0_results.hdf5', 'a') as f:
#             for e in range(len(self.event_list)):
#                 if not self.event_list[e] in f.keys():
#                     # Create a group for the array
#                     group = f.create_group(self.event_list[e])
#                     # Add the array to the group
#                     group.create_dataset('likeH0', data=self.res[e])
#                 else:
#                     print(self.event_list[e]+" already present in result file. Nothing done.")
