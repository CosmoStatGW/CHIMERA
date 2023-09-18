#
#   This module contains classes to handle galaxy catalogs.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

from abc import ABC, abstractmethod
from copy import deepcopy

import pickle
import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d

from CHIMERA.utils import angles
from CHIMERA.cosmo import fLCDM

import logging
log = logging.getLogger(__name__)


__all__ = ["Galaxies",
           "sum_Gaussians_UCV",
           "sum_Gaussians",
]


    



class Galaxies(ABC):
    
    def __init__(self, 

                 # Pixelization
                 nside = None,
                 nest  = False,
    
                 **kwargs,
                 ):
        

        self.nside = np.atleast_1d(nside)
        self.nest  = nest

        self.load(**kwargs)

        if nside is not None:
            self.prepixelize()
        


    @abstractmethod
    def load(self):
        pass

    def prepixelize(self):
        """
        Pre-compute columns of corresponding Healpix indices for all the pixelization parameters provided.
        """
        for n in np.unique(self.nside):
            log.info(f"Precomputing Healpixels for the galaxy catalog (NSIDE={n}, NEST={self.nest})")
            self.data[f"pix{n}"] = angles.find_pix_RAdec(self.data['ra'], self.data['dec'], n, self.nest)

    def precompute(self, nside, pix_todo, z_grid, names=None, weights=None, lambda_cosmo={"H0": 70, "Om0": 0.3}):
        """Pre-compute pixelized p_cat for many events.

        Args:
            nside (np.ndarray): nside Healpix parameter for each event
            pix_todo (list): arrays of pixels for each event
            z_grid (np.ndarray): 2D array of the redshift grids with shape (Nevents, z_int_res)
            names (list, optional): names of GW events, for logging purposes. Defaults to None.
            weights_kind (str, optional): weights_kind for p_cat. Defaults to "N".

        Returns:
            [p_cat], [N_gal]: probability p_cat and associated weigths
        """        
        Nevents = len(nside)
        assert Nevents == len(pix_todo) == len(z_grid)
        if names is None: names = [f"{i}/{Nevents}" for i in range(Nevents)]

        log.info(f"Precomputing p_cat for {Nevents} events...")

        if weights is None:
            log.info(f"Setting uniform weights")
            self.data["w"] = np.ones_like(self.data["z"])
        else:
            log.info(f"Setting weights to {weights}")
            self.data["w"] = self.data[weights]/np.mean(self.data[weights])

        p_cat, N_gal = zip(*[self.precompute_event(nside[e], pix_todo[e], z_grid[e], names[e]) for e in range(Nevents)])

        return p_cat, N_gal 


    def precompute_event(self, nside, pix_todo, z_grid, name):
        """Pre-compute pixelized p_cat for one event.

        Args:
            nside (int): nside Healpix parameter
            pix_todo (np.ndarray): arrays of pixels for each event
            z_grid (np.ndarray): redshift grid
            name (str): name of the GW event, for logging purposes.
            weights_kind (str, optional): weights_kind for p_cat. Defaults to "N".

        Returns:
            p_cat, p_cat_w: probability p_cat and associated weigths
        """
        log.info(f"### {name} ###")

        data   = self.select_event_region(z_grid[0], z_grid[-1], pix_todo, nside)
        pixels = data[f"pix{nside}"]
        
        p_cat  = np.vstack([sum_Gaussians_UCV(z_grid, data["z"][pixels == p], 
                                                  data["z_err"][pixels == p],
                                                  weights=data["w"][pixels == p]) for p in pix_todo]).T

        # p_cat /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega

        p_cat[~np.isfinite(p_cat)] = 0.  # pb. if falls into an empty pixel

        N_gal = np.sum(data[f"pix{nside}"][None, :] == np.array(pix_todo)[:, None], axis=1)

        return p_cat, N_gal


    def select_event_region(self, z_min, z_max, pixels, nside):
        """Select a sub-catalog containing the galaxies inside the given redshift interval and Healpix pixels.

        Args:
            z_min (float): minimum redshift
            z_max (float): maximum redshift
            pixels (np.ndarray, int): Helpix pixels to select.
        """

        log.info(f"Setting sky area to {pixels.shape[0]} confident pixels (nside={nside:d})")
        
        pixn   = f"pix{nside}"

        mask        = np.isin(self.data[pixn], pixels)
        selected    = {k: v[mask] for k, v in self.data.items()}

        log.info(f" > kept {selected['z'].shape[0]} galaxies")
        log.info(f"Setting z range: {z_min:.3f} < z < {z_max:.3f}")

        z_mask      = (selected['z'] > z_min) & (selected['z'] < z_max)
        selected    = {k: v[z_mask] for k, v in selected.items()}

        log.info(f" > kept {selected['z'].shape[0]} galaxies")

        N_gal_pix = np.sum(selected[pixn][None, :] == pixels[:, None], axis=1)
        log.info(f" > mean {np.mean(N_gal_pix):.1f} galaxies per pixel")

        return selected
    
        # keep_from_dataGW(dataGAL, dataGWi, init_sky_cut=4, dL_cut=4, H0_prior_range=[20,200])

    def keep_in_dataGW(self, dataGWi, conf_level_KDE=0.9):

        dataGWi   = np.array([dataGWi[key] for key in ["ra", "dec"]])
        dataGAL   = np.array([self.data[key] for key in ["ra", "dec"]])

        prob  = gaussian_kde(dataGWi).evaluate(dataGAL)

        # 1. Sort the points based on their values
        sorted_idx  = np.argsort(prob)
        sorted_data = dataGAL[:,sorted_idx]
        sorted_prob = prob[sorted_idx]

        # 2. Find the most probable points
        cdf = np.cumsum(sorted_prob, axis=0)
        cdf /= cdf[-1]
        sorted_mask = (cdf >= 1-conf_level_KDE)
        
        # 3. Apply the mask to the original unsorted data
        mask = np.zeros_like(prob, dtype=bool)
        mask[sorted_idx[sorted_mask]] = True

        log.info(" > KDE-GW cut with threshold {:.1f}: kept {:d} galaxies".format(conf_level_KDE, mask.sum()))

        self.data = {k : self.data[k][mask] for k in self.data.keys()}

    # def set_completeness(self, completeness, **kwargs):
    #     self.completeness = completeness(self.data, **kwargs)

    def compute_completeness(self, **kwargs):
        compl = self._completeness(**kwargs)
        compl.compute()
        
        self.P_compl = compl.P_compl()

    def get_interpolant(self, dir_interp=None, z_range=[0.073, 1.3]):

        def _generic_interpolant(z):
            p_cat_int = np.where((z>z_range[0])&(z<z_range[1]), fLCDM.dV_dz(z, {"H0": 70, "Om0": 0.3}), 0)
            p_cat_int /= _fR({"H0": 70, "Om0": 0.3})
            return p_cat_int

        if dir_interp is not None:
            log.info(f"Loading catalog interpolant in {dir_interp}")
            
            with open(dir_interp, "rb") as f:
                p_cat_int = pickle.load(f)
        else:
            log.info("No catalog interpolant provided, using dVdz(70,0.3)")
            p_cat_int = _generic_interpolant
    
        # def _fR_integrand(zz, lambda_cosmo):
        #     return np.where(zz<1.3, 1, 0)*np.array(fLCDM.dV_dz(zz, lambda_cosmo))

        # def _fR_integrated(lambda_cosmo):
        #     return quad(_fR_integrand, 0, 10, args=({"H0": 70, "Om0": 0.3}))[0]  # general

        def _fR(lambda_cosmo):
            return float(fLCDM.V(z_range[1], lambda_cosmo)-fLCDM.V(z_range[0], lambda_cosmo))
        
        def _p_bkg_fcn(z, lambda_cosmo):
            return _fR(lambda_cosmo)*p_cat_int(z) + (1-self.P_compl(z))*np.array(fLCDM.dV_dz(z, lambda_cosmo))  

        return _p_bkg_fcn
    

    def select_upper_cut(self, key, cut):
        """Select galaxies with key > cut

        Args:
            key (str): key of the galaxy catalog
            cut (float): cut value

        Returns:
            np.ndarray: mask
        """
        mask = self.data[key] > cut
        log.info(f" > applying cut to {key}>{cut}: kept {np.sum(mask)} galaxies")
        
        return {k : self.data[k][mask] for k in self.data.keys()}




import jax
import jax.numpy as jnp

@jax.jit
def Gaussian(x, mu, sigma):
    return jnp.power(2 * jnp.pi * (sigma ** 2), -0.5) * jnp.exp(-0.5 * jnp.power((x - mu) / sigma, 2.))



@jax.jit
def sum_Gaussians_UCV(z_grid, mu, sigma, weights=None, lambda_cosmo={"H0": 70, "Om0": 0.3}):
    """ Vectorized sum of multiple Gaussians on z_grid each one with its own weight, mean and standard deviation.
    Each Gaussian is weighted by the volume element dV/dz.

    Args:
        z_grid (np.ndarray): redshift grid
        mu (np.ndarray): mean redshifts
        sigma (np.ndarray): redshifts standard deviations
        weights (np.ndarray, optional): weigths. Defaults to np.ones(len(mu)).

    Returns:
        np.ndarray: sum of Gaussians
    """

    if len(mu) == 0:
        return jnp.zeros_like(z_grid)

    z_grid = jnp.array(z_grid)[:, jnp.newaxis]
    mu = jnp.array(mu)
    sigma = jnp.array(sigma)

    if weights is None:
        weights = jnp.ones(len(mu))

    gauss = Gaussian(z_grid, mu, sigma)*fLCDM.dV_dz(z_grid, lambda_cosmo)
    norm  = jnp.trapz(gauss, z_grid, axis=0)

    return jnp.sum(weights * gauss/norm, axis=1) / jnp.sum(weights)


def deconvolve_volume_factor(z_grid, p_cat, lambda_cosmo={"H0": 70, "Om0": 0.3}):
    """ TBD
    """
    z_grid = np.array(z_grid)[:, np.newaxis]
    p_cat  = np.array(p_cat)

    return np.trapz(p_cat * fLCDM.dV_dz(z_grid, lambda_cosmo), z_grid, axis=0)
                    

    

 
@jax.jit
def sum_Gaussians(z_grid, mu, sigma, weights=None, lambda_cosmo=None):
    """ Vectorized sum of multiple Gaussians on z_grid each one with its own weight, mean and standard deviation.

    Args:
        z_grid (np.ndarray): redshift grid
        mu (np.ndarray): mean redshifts
        sigma (np.ndarray): redshifts standard deviations
        weights (np.ndarray, optional): weigths. Defaults to np.ones(len(mu)).

    Returns:
        np.ndarray: sum of Gaussians
    """

    if len(mu) == 0:
        return jnp.zeros_like(z_grid)

    z_grid  = jnp.array(z_grid)[:, jnp.newaxis]
    mu      = jnp.array(mu)
    sigma   = jnp.array(sigma)
    weights = jnp.array(weights) if weights is not None else jnp.ones(len(mu))

    gauss   = Gaussian(z_grid, mu, sigma)

    return jnp.sum(weights * gauss, axis=1) / jnp.sum(weights)



# def sum_Gaussians(z_grid, mu, sigma, weights=None, lambda_cosmo=None):
#     """ Vectorized sum of multiple Gaussians on z_grid each one with its own weight, mean and standard deviation.

#     Args:
#         z_grid (np.ndarray): redshift grid
#         mu (np.ndarray): mean redshifts
#         sigma (np.ndarray): redshifts standard deviations
#         weights (np.ndarray, optional): weigths. Defaults to np.ones(len(mu)).

#     Returns:
#         np.ndarray: sum of Gaussians
#     """

#     if len(mu)==0:
#         return np.array([])
    
#     z_grid = np.array(z_grid)[:, np.newaxis]
#     mu     = np.array(mu)
#     sigma  = np.array(sigma)

#     if weights is None:
#         weights = np.ones(len(mu))

#     num  = weights * Gaussian(z_grid, mu, sigma)
#     den = np.trapz(num, z_grid, axis=0)

#     return np.sum(num/den, axis=1)


def UVC_distribution(z_grid, lambda_cosmo={"H0":70,"Om0":0.3}):
    """ TBD
    """
    norm = fLCDM.V(np.max(z_grid), lambda_cosmo) - fLCDM.V(np.min(z_grid), lambda_cosmo)
    return 4*np.pi*fLCDM.dV_dz(z_grid, lambda_cosmo) / norm


