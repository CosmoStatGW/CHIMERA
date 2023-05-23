#
#   This module handles the gravitational-wave (GW) events analysis.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import logging
from copy import deepcopy

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import gaussian_kde
from tqdm import tqdm

import CHIMERA.chimeraUtils as chimeraUtils

log = logging.getLogger(__name__)

__all__ = ['GW']



class GW(object):
        
    def __init__(self,
                 
                 # Data
                 data,
                 data_names, 
                 data_smooth,

                 # Hypotheses
                 model_mass,
                 model_rate,
                 model_spin,
                 model_cosmo,

                 # Healpix parameters
                 pixelize   = True,
                 npix_event = 15,
                 nside_list = 32,
                 nest       = False,
                 sky_conf   = 0.9):

        """Class for handling GW events posteriors

        Args:
            data (list): list fwrom MGCosmopop (TBD) with following keys: ["m1z", "m2z", "dL", "ra", "dec"]
            names (list): events names
            model_mass (MGCosmoPop.population.astro.astromassfuncribution): TYPE population from MGCosmoPop
            model_cosmo (MGCosmoPop.cosmology.cosmo): cosmo : TYPE cosmo from MGCosmoPop
            pixelize (bool, optional): Defaults to True.
            nside (int, optional): nside parameter for Healpix. Defaults to 2**8.
            nest (bool, optional): nest parameter for Healpix. Defaults to False.
            sky_conf (float, optional): confidence interval threshold for the pixels to include. Defaults to 0.9.
            file_bws (string, optional): Path to the file containing pre-computed KDE bandwitdhs. Defaults to None.
        """        

        keys_check = ["m1z", "m2z", "dL", "ra", "dec"]

        if not np.all([i in data for i in keys_check]):
            raise ValueError("'data' dictionary must contain the following keys: "+", ".join(keys_check))

        self.data         = deepcopy(data)
        self.data_names   = data_names
        self.data_smooth  = data_smooth

        self.Nevents      = self.data["dL"].shape[0]
        self.Nsamples     = self.data["dL"].shape[1]

        self.model_mass   = model_mass
        self.model_rate   = model_rate
        self.model_cosmo  = model_cosmo

        self.nest         = nest
        self.sky_conf     = sky_conf     

        if pixelize:
            self.nside, self.pix_conf, self.ra_conf, self.dec_conf = self.prepixelize(nside_list, npix_event)




    def prepixelize(self, nside_list, npix_event):
        """
        Pre-compute columns of corresponding Healpix indices for all the provided pixelization parameters.
        """
        for n in nside_list:
            log.info("Precomputing Healpixels for the GW events (NSIDE={:d}, NEST={})".format(n,self.nest))
            self.data.update({"pix"+str(n) : chimeraUtils.find_pix_RAdec(self.data["ra"],self.data["dec"],n,self.nest)})

        log.info("Finding optimal pixelization for each event (~{:d} pix/event)".format(npix_event))

        mat   = np.array([[len(self.compute_sky_conf_event(e,n)) for n in nside_list] for e in range(self.Nevents)])
        ind   = np.argmin(np.abs(mat - npix_event), axis=1)
        nside = np.array(nside_list)[ind]
        u, c  = np.unique(nside, return_counts=True)

        log.info(" > NSIDE: "+" ".join("{:4d}".format(x) for x in u))
        log.info(" > counts:"+" ".join("{:4d}".format(x) for x in c))

        pixels  = [self.compute_sky_conf_event(e,nside[e]) for e in range(self.Nevents)]
        ra, dec = zip(*(chimeraUtils.find_ra_dec(pixels[e], nside=nside[e]) for e in range(self.Nevents)))

        return nside, pixels, ra, dec


    def compute_sky_conf_event(self, event, nside):
        """Return all the Healpix indices of the skymap where the probability of an event is above a given threshold.

        Args:
            event (int): number of the event
            nside (int): nside parameter for Healpix

        Returns:
            np.ndarray: Healpix indices of the skymap where the probability of an event is above a given threshold.
        """

        unique, counts = np.unique(self.data["pix"+str(nside)][event], return_counts=True)
        p              = np.zeros(hp.nside2npix(nside), dtype=float)
        p[unique]      = counts/self.data["pix"+str(nside)][event].shape[0]

        return np.argwhere(p>=self._get_threshold(p, level=self.sky_conf)).reshape(-1)



    def compute_event(self, event, z_grid, lambda_cosmo, lambda_mass):
        """ Compute the pixelized GW probability for one event.

        Args:
            event (int):  number of the event
            z_grid (np.ndarray): redshift grid
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            np.ndarray: pixelized GW probability for one event.
        """

        kde_gw, kde_norm = self.kde_event(event, lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass)
        Npix = len(self.pix_conf[event])
        args =  np.array([np.tile(z_grid, Npix),
                          np.hstack([ np.full_like(z_grid, x) for x in self.ra_conf[event] ]),
                          np.hstack([ np.full_like(z_grid, x) for x in self.dec_conf[event] ])])

        p_gw =  kde_gw(args).reshape(Npix,len(z_grid)).T * kde_norm

        return p_gw


    def kde_event(self, event, lambda_cosmo, lambda_mass):
        """Compute the KDE for one event.

        Args:
            event (int): number of the event
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            [gaussian_kde, norm]: KDE for one event and its normalization factor.
        """

        dL  = self.data["dL"][event]
        z   = self.model_cosmo.z_from_dL(dL*1000, lambda_cosmo)
        ra  = self.data["ra"][event]
        dec = self.data["dec"][event]
        m1  = self.data["m1z"][event]/(1+z)
        m2  = self.data["m2z"][event]/(1+z)

        weight = self.model_mass(m1, m2, lambda_mass) * (1 + z)**-2 * self.model_cosmo.ddL_dz(z, lambda_cosmo, dL)**-1

        norm   = np.mean(weight, axis=0)

        return gaussian_kde(np.array([z,ra,dec]), bw_method=self.data_smooth, weights=weight), norm
        # return gaussian_kde(np.array([z,ra,dec]), bw_method=self.data_smooth), norm


    

    # OLD, to be removed 

    def old_like(self, lambda_cosmo, lambda_mass, **kwargs):
        self.update_cosmology_quantities(lambda_cosmo)
        self.update_KDEgw_weights(lambda_mass)
        self.compute_all_KDEs(keys=["z", "ra", "dec"], **kwargs)

        return self.eventKDEs, self.KDEnorms


    def update_cosmology_quantities(self, lambda_cosmo):
        """Helper function to update cosmological quantities

        Args:
            lambda_cosmo (dictionary): cosmology parameters
        """

        self.lambda_cosmo = lambda_cosmo

        z = self.model_cosmo.z_from_dL(self.data["dL"]*1000, self.lambda_cosmo)
        
        self.data.update({"z" : z})
        self.data.update({"m1" : self.data["m1z"]/(1+z)})
        self.data.update({"m2" : self.data["m2z"]/(1+z)})
        # print("Now, z mean = ", np.mean(self.data["z"]))


    def update_KDEgw_weights(self, lambda_mass):
        """Helper function to update KDE's weigths

        Args:
            lambda_mass (dictionary): mass function parameters
        """
        self.lambda_mass = lambda_mass

        # Compute weigths if needed w=p_m1m2/(1+z)
        self.KDEweights = self.model_mass(self.data["m1"], self.data["m2"], lambda_mass) \
                          * (self.data["dL"] * (1 + self.data["z"]))**-2 \
                          * (self.model_cosmo.ddL_dz(self.data["z"], self.lambda_cosmo, self.data["dL"]))**-1
        
        self.KDEnorms = np.mean(self.KDEweights, axis=0)




    def compute_sky_localization(self, nside):
        """ Compute credible credible sky location (in term of Healpixels) and associated probabilites
        for all the events using the (RA, DEC) posteriors from GW data. 

        Parameters
            credible_level (float): Threshold above which pixels are considered "confident."
        Returns
            ra_conf (np.ndarray): Right ascensions for the confident pixels.
            dec_conf (np.ndarray): Declinations for the confident pixels.
            pix_conf (np.ndarray): Healpixel indices for the confident pixels.
            p (np.ndarray): Probabilities for each pixel.

        TBD. Including reasonable weigths does not affect the results.
        """        

        ra_conf      = np.full((self.Nevents, self.npix), np.nan)
        dec_conf     = np.full((self.Nevents, self.npix), np.nan)
        pix_conf     = np.full((self.Nevents, self.npix), -1, dtype=np.int32)
        p            = np.full((self.Nevents, self.npix), np.nan)

        for i in range(self.Nevents):
            pix_all                 = chimeraUtils.find_pix_RAdec(ra=self.data["ra"][i], dec=self.data["dec"][i], nside=nside)
            unique, counts          = np.unique(pix_all, return_counts=True)
            p[i]                    = np.zeros(self.npix, dtype=float)
            p[i][unique]            = counts/pix_all.shape[0]
            is_conf                 = p[i]>=self._get_threshold(p[i], level=self.sky_conf)
            pix_conf[i,:][is_conf]  = np.arange(0, self.npix)[is_conf]
            ra_conf[i,:][is_conf], dec_conf[i,:][is_conf] = chimeraUtils.find_ra_dec(pix=pix_conf[i,:][is_conf], nside=nside)

        self.ra_conf  = ra_conf
        self.dec_conf = dec_conf 
        self.pix_conf = pix_conf
        self.pix_prob = p

        self.ra_conf_finite   = [np.array(ra_conf[i, np.isfinite(ra_conf[i])]) for i in range(self.Nevents)]
        self.dec_conf_finite  = [np.array(dec_conf[i, np.isfinite(dec_conf[i])]) for i in range(self.Nevents)]
        self.pix_conf_finite  = [np.array(pix_conf[i, pix_conf[i]!= -1]) for i in range(self.Nevents)]
        self.Npix_conf_finite = np.array([np.sum(np.isfinite(dec_conf[i])) for i in range(self.Nevents)])

        # self.pix_conf_finite = find_pix_RAdec(ra=self.ra_conf_finite, dec=self.dec_conf_finite, nside=self.nside)
        # self.pix_conf2        = np.full_like(self.ra_conf, np.nan)
        # self.pix_conf2[finite]= self.pix_conf_finite



    def compute_all_KDEs(self, 
                         keys=["z", "ra", "dec"],
                         weights=False, 
                         bw_file=None, 
                         bw_method=None, 
                         factor=1):

        # # Compute weigths if needed w=p_m1m2/(1+z)
        # if weights:
        #     # w = self.model_mass(self.data["m1"], self.data["m2"]) / (1 + self.data["z"])
        #     w = self.model_mass(self.data["m1"], self.data["m2"]) #\
        #         # * (self.data["dL"] * (1 + self.data["z"]))**-2 \
        #         # * (self.cosmology.ddL_dz(self.data["z"], self.lambda_cosmo, self.data["dL"]))**-1
        #     self.eventKDEs_norm = np.mean(w, axis=0)
                
        # else:
        #     w = [None]*self.Nevents
        #     self.eventKDEs_norm = np.ones((self.Nevents, self.Nsamples))


        # If not given, compute KDE bandwidths using `bw_method`
        if bw_file is None:

            if isinstance(bw_method, str):
                if bw_method == "1Dscott":
                    # Here we use keys[0] (that should be dL or z)
                    bws = self._find_bandwidths_1Dscott(keys[0])
                elif bw_method == "3Dscott":
                    # Here we use the full 3D distrbution
                    bws = self._find_bandwidths_3Dscott(keys)
                elif bw_method == "gridseach":
                    bws = self._find_bandwidths_gridseach(keys, grid=np.logspace(-4, 1, 100))
                else:
                    raise ValueError("Invalid bw_method string") 

                Nnan = np.sum(~np.isfinite(bws))
                if Nnan>0:
                    print("WARNING:", Nnan, " nan bws")

                # Save pre-computed bandwidths
                # ts      = time.strftime("%Y%m%d-%H%M%S")
                # bw_file = "DarkSirensStatV1/data/bws_{:s}_Nsamples_{:d}.ecsv".format(ts, self.Nsamples)
                # table   = Table((self.names, bws), names=["name", "bw"])
                # table.write(bw_file, format="ascii.ecsv")

            elif isinstance(bw_method, list):
                assert len(bw_method) == self.Nevents
                bws = np.array(bw_method).astype(float)

            else:
                raise ValueError("Invalid bw_method type. Allowed tipes are: str, list") 
        
        # If table is given, load KDE bandwidths from it
        else: 
            bw_table = Table.read(bw_file, format="ascii.ecsv")
            bw_table.add_index('name')  # name indexing
            
            bws = np.array([bw_table.loc[n]["bw"] for n in self.names]).astype(float)

        bws = np.array(bws)
 
        self.eventKDEs = self.compute_KDEs(keys, bws=factor*bws, weights=self.KDEweights)



    def compute_z_grids(self, H0_prior_range, z_conf_range, z_res):
        """Computes [z_min, z_max] of the redshift probability distribution of each event.

        Args:
            H0_prior_range (list): range of the H0 values that will be explored.
            z_confidence_range (list, float): if list: percentile range, if number: median \pm number * MAD

        Returns:
            np.ndarray: z_ranges
        """
        
        if isinstance(z_conf_range, list):
            dL_min, dL_max = np.percentile(self.data["dL"], z_conf_range, axis=1)
        elif isinstance(z_conf_range, int):
            mu     = np.mean(self.data["dL"], axis=1)
            sig    = np.std(self.data["dL"], axis=1)
            dL_min = mu - z_conf_range * sig
            dL_max = mu + z_conf_range * sig
        else:
            med = np.median(self.data["dL"])
            mad = np.median(np.abs(self.data["dL"] - med))
            dL_min = med - z_conf_range * mad
            dL_max = med + z_conf_range * mad

        # dL_min[dL_min<0.] = 0.
        dL_min[dL_min<0.073] = 0.073

        z_min  = self.model_cosmo.z_from_dL(dL_min*1000, {"H0":H0_prior_range[0], "Om0":0.3})
        z_max  = self.model_cosmo.z_from_dL(dL_max*1000, {"H0":H0_prior_range[1], "Om0":0.3})

        return np.linspace(z_min, z_max, z_res, axis=1)




    def compute_z_ranges(self, H0_prior_range, z_conf_range):
        """Computes [z_min, z_max] of the redshift probability distribution of each event.

        Args:
            H0_prior_range (list): range of the H0 values that will be explored.
            z_confidence_range (list, float): if list: percentile range, if number: median \pm number * MAD

        Returns:
            np.ndarray: z_ranges
        """
        
        if isinstance(z_conf_range, list):
            dL_min, dL_max = np.percentile(self.data["dL"], z_conf_range, axis=1)
        elif isinstance(z_conf_range, int):
            mu     = np.mean(self.data["dL"], axis=1)
            sig    = np.std(self.data["dL"], axis=1)
            dL_min = mu - z_conf_range * sig
            dL_max = mu + z_conf_range * sig
        else:
            med = np.median(self.data["dL"])
            mad = np.median(np.abs(self.data["dL"] - med))
            dL_min = med - z_conf_range * mad
            dL_max = med + z_conf_range * mad

        dL_min[dL_min<0.] = 0.
        z_min  = self.model_cosmo.z_from_dL(dL_min*1000, {"H0":H0_prior_range[0], "Om0":0.3})
        z_max  = self.model_cosmo.z_from_dL(dL_max*1000, {"H0":H0_prior_range[1], "Om0":0.3})

        return np.array([z_min,z_max]).T



    def get_loc_meas(self, H0, dL_conf_range=[0,90]):
        """Function to get both sky and volume localization given a GW event and an underlying cosmology.
        TODO: compute directly from dL

        Args:
            H0 (_type_): _description_
            z_conf_range (list, optional): _description_. Defaults to [0,90].

        Returns:
            _type_: _description_
        """

        dL_min, dL_max = np.percentile(self.data["dL"], dL_conf_range, axis=1)
        vol = 4/3*np.pi * (dL_max**3 - dL_min**3)

        loc_area = self.Npix_conf_finite * hp.nside2pixarea(self.nside, degrees = True)

        return loc_area, loc_area/41253. * vol * 1e-9 # Gpc^-3


    def _find_bandwidths_1Dscott(self, key):
        """
        Scotts rule. Scott, D.W. (1992) Multivariate Density Estimation. [...]. New York: Wiley.
        """

        print("Computing 3D KDE bandwidths using 1D Scott's rule")

        bws = []
        for i in tqdm(range(self.Nevents)):
            data1D = self.data[key][i]
            sigma  = min(np.std(data1D, ddof=1), 
                        (np.percentile(data1D, q=75) - np.percentile(data1D, q=25)) / 1.348979500392163)
            bws.append(sigma * np.power(len(data1D), -1.0 / 5))

        return bws

    def _find_bandwidths_3Dscott(self, keys=["z", "ra", "dec"]):
        """
        Scotts rule. Scott, D.W. (1992) Multivariate Density Estimation. [...]. New York: Wiley.
        """

        print("Computing 3D KDE bandwidths using 3D Scott's rule")

        bws = []
        for i in tqdm(range(self.Nevents)):
            data = np.array([self.data[keys[0]][i],
                            self.data[keys[1]][i],
                            self.data[keys[2]][i]]) 
            kde = gaussian_kde(data)
            bws.append(kde.covariance_factor() * data.std())


        return bws

    def _find_bandwidths_gridseach(self, keys=["z", "ra", "dec"], grid=np.logspace(-2, 1, 100)):
        """TBD. Including reasonable weigths does not affect the results.
            Default grid np.logspace(-2, 1, 100)
        """        

        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KernelDensity

        print("Computing 3D KDE bandwidths using sklearn GridSearch")

        bws = []
        for i in tqdm(range(self.Nevents)):
            # print(i)
            gridbw = GridSearchCV(KernelDensity(), {'bandwidth': grid})
            gridbw.fit(np.array([self.data[keys[0]][i],
                                 self.data[keys[1]][i],
                                 self.data[keys[2]][i]]).T, sample_weight=None)
            bws.append(gridbw.best_estimator_.bandwidth)

        return bws


    def compute_KDEs(self, keys, bws, weights):

        KDEs = []

        for i in range(self.Nevents):
            data   = np.array( [self.data[keys[0]][i],
                                self.data[keys[1]][i],
                                self.data[keys[2]][i]] )

            KDEs.append(gaussian_kde(data, bw_method=bws[i], weights=weights[i]))


        return KDEs


            


    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    def plot_p_z_marginal_map(self, ra_point, dec_point, z, ax=None):
        """ to be moved"""
        if ax is None: 
            fig = plt.figure(figsize=(15,8))
            ax = fig.add_subplot(111, projection='mollweide')

        ax.scatter(np.deg2rad(ra_point), np.deg2rad(dec_point) )


    def plot_p_z_marginal(self, event_i, ra_point, dec_point, z, ax=None):
        """ to be moved"""
        if ax is None: 
            fig = plt.figure(figsize=(15,8))
            ax = fig.add_subplot(111, projection='mollweide')


        print("Plotting event {:s}".format(self.names[event_i]))
        ax.scatter(np.deg2rad(ra_point), np.deg2rad(dec_point))


    def plot_hists_KDE_3x(self, KDEs, keys, Nresample = 10**5, median_coring=False):
        """ to be moved"""
        assert len(KDEs) == self.Nevents

        for i in range(self.Nevents):
            fig, ax = plt.subplots(1, 3, figsize=(10,3),dpi=100)
            resampled = KDEs[i].resample(Nresample)


            for jpar in range(3):
                ax[jpar].hist(self.data[keys[jpar]][i], bins=30, histtype="stepfilled", density=True, color="silver", label="data")
                ax[jpar].hist(resampled[jpar], bins=30, histtype="step", density=True, color="k", ls="--", label="resampled KDE")
                ax[jpar].set_xlabel(keys[jpar])


            if median_coring:
                dmu, dsigma       = np.mean(self.data[keys[0]][i]), np.std(self.data[keys[0]][i])
                xgrid             = np.linspace(max(0, dmu-4*dsigma), dmu+4*dsigma, 200) # evaluated in dL or z \pm 4sigma
                ra_todo, dec_todo = list(np.median([self.data["ra"][i], self.data["dec"][i]], axis=1)) # evaluated at median (ra, dec)
                prob_x            = np.array([KDEs[i]([xi, ra_todo, dec_todo])[0] for xi in xgrid])
                prob_x            = prob_x/np.trapz(prob_x,xgrid)

                ax[0].plot(xgrid, prob_x, color="orange")
                ax[1].axvline(ra_todo, color='orange', ls='-', label="$p({:s}|ra_{{med}},dec_{{med}})$".format(keys[0]))
                ax[2].axvline(dec_todo, color='orange', ls='-')

            ax[0].set_ylabel("Density")
            ax[1].legend()
            plt.suptitle(self.names[i])
            plt.show()


    def plot_confpix_KDEs(self, z_grid, ax=None):
        if ax is None: 
            fig, ax = plt.subplots(1, 3, figsize=(10,3),dpi=100)




        
    def _get_threshold(self, norm_counts, level=0.9):
        '''
        Finds value mincount of normalized number counts norm_counts that bouds the x% credible region , with x=level
        Then to select pixels in that region: all_pixels[norm_count>mincount]
        '''
        prob_sorted     = np.sort(norm_counts)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        idx             = np.searchsorted(prob_sorted_cum, level) # find index of array which bounds the confidence interval
        mincount        = prob_sorted[idx] 
        return mincount
    
    
