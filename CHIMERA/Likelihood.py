import h5py
import numpy as np
from tqdm import tqdm
from scipy.integrate import quad

import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 

import DSutils, DSplotting

from GW import GW
from GLADE import GLADEPlus_v2
from Completeness import SkipCompleteness, MaskCompleteness_v2

from Mock import MockGalaxiesMICEv2

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

                 # Parameters for integration in redshift
                 z_int_H0_prior,
                 z_int_sigma,
                 z_int_res,

                 # Backend
                 dir_out,
                 ):
        
            

        self.gw      = GW(data=data_GW, data_names=data_GW_names,  data_smooth=data_GW_smooth,
                          model_mass=model_mass, model_rate=model_rate, model_spin="", model_cosmo=model_cosmo, 
                          npix_event=npix_event, nside_list=nside_list, sky_conf=sky_conf)
        
        self.z_grids = self.gw.compute_z_grids(z_int_H0_prior, z_int_sigma, z_int_res)

        self.gal     = MockGalaxiesMICEv2(data_GAL_dir, 
                                          z_err = data_GAL_zerr,
                                          nside = self.gw.nside)
        
        self.p_gal, self.p_gal_w = self.gal.precompute(self.gw.nside, self.gw.pix_conf, self.z_grids, self.gw.data_names)
        # why not functions?

        self.dir_out = dir_out



    def compute(self, lambda_cosmo, lambda_mass, lambda_rate, return_full=False):

        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

        like_event = np.empty(self.gw.Nevents)
        
        if return_full:
            like_allpix_ev = []
            p_gw_ev = []
            p_rate_ev = []
        
        for e in range(self.gw.Nevents):
            p_gw     = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass)
            
            # p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e])
            p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e])*self.gw.model_cosmo.dV_dz(self.z_grids[e], lambda_cosmo)
            p_rate  /= np.trapz(p_rate, self.z_grids[e], axis=0)


            p_z      = p_rate[:,np.newaxis]*self.p_gal[e]
            p_z     /= np.trapz(p_z, self.z_grids[e], axis=0)

            like_pix      = np.trapz(p_gw*p_z, self.z_grids[e], axis=0)
            like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)
        
            if return_full:
                like_allpix_ev.append(like_pix)
                p_gw_ev.append(p_gw)
                p_rate_ev.append(p_rate)
        
        if return_full:
            return like_event, like_allpix_ev, p_gw_ev, p_rate
        else:
            return like_event



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

    def compute_w_Beta(self, lambda_cosmo, lambda_mass, lambda_rate, beta, return_full=False):

        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis) /((~np.isnan(A))*weights).sum(axis=axis)

        like_event = np.empty(self.gw.Nevents)
        
        if return_full:
            like_allpix_ev = []
            p_gw_ev = []
            p_rate_ev = []
        
        for e in range(self.gw.Nevents):
            # print(e)
            p_gw     = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass)
            # print("GW")

            p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e]) #*self.gw.model_cosmo.dV_dz(self.z_grids[e], lambda_cosmo)
            # p_rate  /= np.trapz(p_rate, self.z_grids[e], axis=0)
            # print("rate")


            # p_gal    = self.p_gal[e]
            # p_gal   /= np.trapz(p_gal, self.z_grids[e], axis=0)
            # print("p_gal")


            p_z      = p_rate[:,np.newaxis]#*self.p_gal[e]
            # p_z     /= np.trapz(p_z, self.z_grids[e], axis=0)
            # print("p_z")

            log_p    = np.log(p_gw) + np.log(p_z)

            # \int dz, then dOmega
            like_pix      = np.trapz(np.exp(log_p), self.z_grids[e], axis=0)#/beta
            like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)
            # print("like_pix")
        
            # \int dOmega, then dz
            # l1 = nanaverage(p_gw*p_z, weights=self.p_gal_w[e], axis=1)
            # like_event[e] = np.trapz(l1, self.z_grids[e], axis=0)/beta

            if return_full:
                like_allpix_ev.append(like_pix)
                p_gw_ev.append(p_gw)
                p_rate_ev.append(p_rate)
        
        if return_full:
            return like_event, like_allpix_ev, p_gw_ev, p_rate
        else:
            return like_event



    def compute_w_Beta_log(self, lambda_cosmo, lambda_mass, lambda_rate, beta, return_full=False):

        def nanaverage(A,weights,axis):
            return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

        like_event = np.empty(self.gw.Nevents)
        
        if return_full:
            like_allpix_ev = []
            p_gw_ev = []
            p_rate_ev = []
        
        for e in range(self.gw.Nevents):
            p_gw     = self.gw.compute_event(e, self.z_grids[e], lambda_cosmo, lambda_mass)
            
            p_rate   = self.gw.model_rate(self.z_grids[e], lambda_rate)/(1.+self.z_grids[e])
            p_rate  /= np.trapz(p_rate, self.z_grids[e], axis=0)
            p_z      = p_rate[:,np.newaxis]*self.p_gal[e]
            p_z     /= np.trapz(p_z, self.z_grids[e], axis=0)

            like_pix      = np.trapz(p_gw*p_z, self.z_grids[e], axis=0)/beta
            like_event[e] = nanaverage(like_pix, weights=self.p_gal_w[e], axis=0)
        
            if return_full:
                like_allpix_ev.append(like_pix)
                p_gw_ev.append(p_gw)
                p_rate_ev.append(p_rate)
        
        if return_full:
            return like_event, like_allpix_ev, p_gw_ev, p_rate
        else:
            return like_event



        # pass
            # np.save(self.dir_out+"t_"+str(e)+"results.npy", np.array(like))

    # def compute_event(self, event, z_grid, lambda_cosmo, lambda_mass, lambda_rate):
        # return np.trapz(p_gw * p_gal, z_grid, axis=0)


###




class IndividualLikelihoods():

    def __init__(self, event_list, obs_run, conf, kind, savefig=False):

        self.event_list = event_list
        self.Nevents    = len(event_list)
        self.conf       = conf
        self.savefig    = savefig


        if kind == "BBH":
            self.dataGW = DSutils.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=True, SNR_th=11)
        
        elif kind == "other":
            self.dataGW = DSutils.load_data(self.event_list, obs_run, conf["GW_Nsamples"], BBH_only=False)
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
            DSplotting.plot_2Dcoring(ax, cat.data.ra, cat.data.dec, cat.data.z,
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
