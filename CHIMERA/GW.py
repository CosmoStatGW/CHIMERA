#
#   This module handles the gravitational-wave (GW) events analysis.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import healpy as hp
import numpy as np

# from scipy.stats import gaussian_kde

from jax.scipy.stats import gaussian_kde
from jax.config import config
config.update("jax_enable_x64", True)

import logging
log = logging.getLogger(__name__)


from CHIMERA.utils import (angles, misc)


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
                 nside_list = [32],
                 nest       = False,
                 sky_conf   = 0.9,
                 check_Neff = True,
                 data_Neff  = 5,
                 ):

        """Class for handling GW events posteriors

        Args:
            data (list): list with following keys: ["m1det", "m2det", "dL", "ra", "dec"]
            names (list): events names
            model_mass : TYPE population from 
            model_cosmo cosmo : TYPE cosmo from
            pixelize (bool, optional): Defaults to True.
            nside (int, optional): nside parameter for Healpix. Defaults to 2**8.
            nest (bool, optional): nest parameter for Healpix. Defaults to False.
            sky_conf (float, optional): confidence interval threshold for the pixels to include. Defaults to 0.9.
            file_bws (string, optional): Path to the file containing pre-computed KDE bandwitdhs. Defaults to None.
        """        

        keys_check = ["m1det", "m2det", "dL", "ra", "dec"]

        if not np.all([i in data for i in keys_check]):
            raise ValueError("'data' dictionary must contain the following keys: "+", ".join(keys_check))

        self.data         = data.copy()
        self.data_names   = np.atleast_1d(data_names)
        self.data_smooth  = data_smooth

        self.Nevents      = self.data["dL"].shape[0]
        self.Nsamples     = self.data["dL"].shape[1]

        self.model_mass   = model_mass
        self.model_rate   = model_rate
        self.modefl_spin  = model_spin
        self.model_cosmo  = model_cosmo

        self.nest         = nest
        self.sky_conf     = sky_conf   
        self.check_Neff   = check_Neff  
        self.data_Neff    = data_Neff

        if pixelize:
            self.nside, self.pix_conf, self.ra_conf, self.dec_conf = self.prepixelize(nside_list, npix_event)


    def prepixelize(self, nside_list, npix_event):
        """Pre-compute columns of corresponding Healpix indices for all the provided `nside_list` pixelization parameters.

        Args:
            nside_list (list): list of nside parameters for Healpix
            npix_event (number): approximate number of desired pixels per event

        Returns:
            np.ndarray: optimized nside parameter for each event
            np.ndarray: pixels in the sky_conf area for each event
            np.ndarray: ra of the pixels in the sky_conf area for each event
            np.ndarray: dec of the pixels in the sky_conf area for each event
        """

        for n in nside_list:
            log.info(f"Precomputing Healpix pixels for the GW events (NSIDE={n}, NEST={self.nest})")
            self.data[f"pix{n}"] = angles.find_pix_RAdec(self.data["ra"], self.data["dec"], n, self.nest)

        log.info(f"Finding optimal pixelization for each event (~{npix_event} pix/event)")

        mat   = np.array([[len(self.compute_sky_conf_event(e,n)) for n in nside_list] for e in range(self.Nevents)])
        ind   = np.argmin(np.abs(mat - npix_event), axis=1)
        nside = np.array(nside_list)[ind]
        # print(nside)
        u, c  = np.unique(nside, return_counts=True)

        log.info(" > NSIDE: " + " ".join(f"{x:4d}" for x in u))
        log.info(" > counts:" + " ".join(f"{x:4d}" for x in c))

        pixels  = [self.compute_sky_conf_event(e,nside[e]) for e in range(self.Nevents)]
        ra, dec = zip(*[angles.find_ra_dec(pixels[e], nside=nside[e]) for e in range(self.Nevents)])

        return nside, pixels, ra, dec


    def compute_sky_conf_event(self, event, nside):
        """Return all the Healpix pixel indices where the probability of an event is above a given threshold.

        Args:
            event (int): number of the event
            nside (int): nside parameter for Healpix

        Returns:
            np.ndarray: Healpix indices of the skymap where the probability of an event is above a given threshold.
        """

        pixel_key      = f"pix{nside}"
        unique, counts = np.unique(self.data[pixel_key][event], return_counts=True)
        p              = np.zeros(hp.nside2npix(nside), dtype=float)
        p[unique]      = counts/self.data[pixel_key][event].shape[0]

        return np.argwhere(p >= self._get_threshold(p, level=self.sky_conf)).flatten()


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
    




    def compute_event_log(self, event, z_grid, lambda_cosmo, lambda_mass, lambda_rate=None):
        """GW probability for one event pixelized.

        Args:
            event (int):  number of the event
            z_grid (np.ndarray): redshift grid
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            np.ndarray: pixelized GW probability for one event.
        """

        Npix = len(self.pix_conf[event])
        Nz   = len(z_grid)
        kde_gw, kde_norm_log = self._kde_event_log(event, lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)

        if kde_gw is None:
            return np.full((Nz, Npix), -np.inf)

        vals =  np.array([np.tile(z_grid, Npix),
                          np.hstack([ np.full_like(z_grid, x) for x in self.ra_conf[event] ]),
                          np.hstack([ np.full_like(z_grid, x) for x in self.dec_conf[event] ])])

        return kde_gw.logpdf(vals).reshape(Npix,Nz).T + kde_norm_log


    def _kde_event_log(self, event, lambda_cosmo, lambda_mass, lambda_rate):
        """Compute the KDE for one event (log approach)

        Args:
            event (int): number of the event
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            [gaussian_kde, norm]: KDE for one event and its normalization factor.
        """

        dL     = self.data["dL"][event]
        z      = np.array(self.model_cosmo.z_from_dL(dL, lambda_cosmo))
        loc3D  = np.array([z, self.data["ra"][event], self.data["dec"][event]])
        m1, m2 = self.data["m1det"][event]/(1.+z), self.data["m2det"][event]/(1.+z)

        models = self.model_mass(m1, m2, lambda_mass)
        jacD2S = 2*np.log1p(z) + self.model_cosmo.log_ddL_dz(z, lambda_cosmo, dL)
        priors = 2*np.log(dL)

        log_weights = models - jacD2S - priors # np.nan_to_num(, nan=-np.inf)
        log_norm    = np.logaddexp.reduce(log_weights) - np.log(len(log_weights))
        Neff        = misc.get_Neff_log(log_weights, log_norm)

        if (np.isfinite(log_weights).sum()<5) or (self.check_Neff & (Neff < self.data_Neff)):
            log.warning(f"Neff = {Neff:.1f} < {self.data_Neff} for event {event}: returning -inf logprob")
            return None, log_norm

        return gaussian_kde(loc3D, bw_method=self.data_smooth, weights=np.exp(log_weights)), log_norm



    def compute_event(self, event, z_grid, lambda_cosmo, lambda_mass, lambda_rate=None):
        """GW probability for one event pixelized.

        Args:
            event (int):  number of the event
            z_grid (np.ndarray): redshift grid
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            np.ndarray: pixelized GW probability for one event.
        """

        Npix = len(self.pix_conf[event])
        Nz   = len(z_grid)
        kde_gw, kde_norm = self._kde_event(event, lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)

        if kde_gw is None:
            return np.zeros((Nz, Npix))

        vals =  np.array([np.tile(z_grid, Npix),
                          np.hstack([ np.full_like(z_grid, x) for x in self.ra_conf[event] ]),
                          np.hstack([ np.full_like(z_grid, x) for x in self.dec_conf[event] ])])

        return kde_gw.pdf(vals).reshape(Npix,Nz).T * kde_norm



    def _kde_event(self, event, lambda_cosmo, lambda_mass, lambda_rate=None):
        """Compute the KDE for one event.

        Args:
            event (int): number of the event
            lambda_cosmo (dict): cosmology hyperparameters
            lambda_mass (dict): mass hyperparameters

        Returns:
            [gaussian_kde, norm]: KDE for one event and its normalization factor.
        """

        dL      = self.data["dL"][event]
        z       = np.array(self.model_cosmo.z_from_dL(dL, lambda_cosmo))
        loc3D   = np.array([z, self.data["ra"][event], self.data["dec"][event]])
        m1, m2  = self.data["m1det"][event]/(1.+z), self.data["m2det"][event]/(1.+z)

        weights  = self.model_mass(m1, m2, lambda_mass) *\
                   np.power(1+z, -2) * np.power(self.model_cosmo.ddL_dz(z, lambda_cosmo, dL), -1) *\
                   np.power(dL, -2)

        norm    = np.mean(weights, axis=0)
        Neff    = misc.get_Neff(weights, norm)

        if ((weights>0).sum()<5) or (self.check_Neff & (Neff < self.data_Neff)):
            log.warning(f"Neff = {Neff:.1f} < {self.data_Neff} for event {event}: returning zero probability")
            return None, norm
        
        return gaussian_kde(np.array(loc3D), bw_method=self.data_smooth, weights=weights), norm



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

        dL_min[dL_min<1.e-6] = 1.e-6
        # dL_min[dL_min<0.073] = 0.073

        z_min  = self.model_cosmo.z_from_dL(dL_min, {"H0":H0_prior_range[0], "Om0":0.3})
        z_max  = self.model_cosmo.z_from_dL(dL_max, {"H0":H0_prior_range[1], "Om0":0.3})

        return np.linspace(z_min, z_max, z_res, axis=1)






    # def compute_catalog(self, lambda_cosmo, lambda_mass, lambda_rate):
    #     """TBD. Compute the KDE for the entire catalog.

    #     Args:
    #         event (int): number of the event
    #         lambda_cosmo (dict): cosmology hyperparameters
    #         lambda_mass (dict): mass hyperparameters

    #     Returns:
    #         [gaussian_kde, norm]: KDE for one event and its normalization factor.
    #     """

    #     dL  = self.data["dL"]
    #     z   = self.model_cosmo.z_from_dL(dL, lambda_cosmo)
    #     ra  = self.data["ra"]
    #     dec = self.data["dec"]
    #     m1  = self.data["m1det"]/(1.+z)
    #     m2  = self.data["m2det"]/(1.+z)

    #     models = self.model_mass(m1, m2, lambda_mass)
    #     jacD2S = 2*np.log1p(z) + self.model_cosmo.log_ddL_dz(z, lambda_cosmo, dL)
    #     priors = 2*np.log(dL)

    #     # models = self.model_rate(z, lambda_rate)/(1.+z) * self.model_mass(m1, m2, lambda_mass) * self.model_cosmo.dV_dz(z, lambda_cosmo)

    #     log_weights = np.nan_to_num(models - jacD2S - priors, nan=-np.inf)
        
    #     # log_norm    = np.logaddexp.reduce(log_weights) - np.log(len(log_weights))
    #     # log_s2      = np.logaddexp.reduce(2.*log_weights) - 2.*np.log(len(log_weights)) 
    #     # log_sig2    = chimeraUtils.logdiffexp(log_s2, 2.*log_norm-np.log(len(log_weights)))
    #     # Neff        = np.exp(2.*log_norm - log_sig2)

    #     weights     = np.exp(log_weights)

    #     # if (Neff < 5) or ((weights!=0).sum() < 5):
    #     #     log.warning(f"Neff={Neff:.2f} < 5.0 for event {event}.")
    #     #     return None, log_norm

        

    #     return gaussian_kde(np.array([z,ra,dec]), bw_method=self.data_smooth, weights=weights), log_norm