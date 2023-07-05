#
#   This module contains classes to handle galaxy catalogs.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde, norm

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
                 
                 # Completeness
                 completeness = None,
                 useDirac = None,
                 
                 **kwargs,
                 ):
        

        self.nside = np.atleast_1d(nside)
        self.nest  = nest

        self.load(**kwargs)

        if completeness is not None:
            self._completeness = deepcopy(completeness)
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
        for n in np.unique(self.nside):
            log.info(f"Precomputing Healpixels for the galaxy catalog (NSIDE={n}, NEST={self.nest})")
            self.data[f"pix{n}"] = angles.find_pix_RAdec(self.data['ra'], self.data['dec'], n, self.nest)

    def precompute(self, nside, pix_todo, z_grid, names=None, weights=None):
        """Pre-compute pixelized p_gal for many events.

        Args:
            nside (np.ndarray): nside Healpix parameter for each event
            pix_todo (list): arrays of pixels for each event
            z_grid (np.ndarray): 2D array of the redshift grids with shape (Nevents, z_int_res)
            names (list, optional): names of GW events, for logging purposes. Defaults to None.
            weights_kind (str, optional): weights_kind for p_gal. Defaults to "N".

        Returns:
            [p_gal], [N_gal]: probability p_gal and associated weigths
        """        
        Nevents = len(nside)
        assert Nevents == len(pix_todo) == len(z_grid)
        if names is None: names = [f"{i}/{Nevents}" for i in range(Nevents)]

        log.info(f"Precomputing p_GAL for {Nevents} events...")

        if weights is None:
            log.info(f"Setting uniform weights")
            self.data["w"] = np.ones_like(self.data["z"])
        else:
            log.info(f"Setting weights to {weights}")
            self.data["w"] = 1/(self.data[weights]/np.mean(self.data[weights]))

        p_gal, N_gal = zip(*[self.precompute_event(nside[e], pix_todo[e], z_grid[e], names[e]) for e in range(Nevents)])

        return p_gal, N_gal 


    def precompute_event(self, nside, pix_todo, z_grid, name):
        """Pre-compute pixelized p_gal for one event.

        Args:
            nside (int): nside Healpix parameter
            pix_todo (np.ndarray): arrays of pixels for each event
            z_grid (np.ndarray): redshift grid
            name (str): name of the GW event, for logging purposes.
            weights_kind (str, optional): weights_kind for p_gal. Defaults to "N".

        Returns:
            p_gal, p_gal_w: probability p_gal and associated weigths
        """
        log.info(f"### {name} ###")

        data   = self.select_event_region(z_grid[0], z_grid[-1], pix_todo, nside)
        pixels = data[f"pix{nside}"]
        
        p_gal  = np.vstack([sum_Gaussians_UCV(z_grid, data["z"][pixels == p], 
                                                      data["z_err"][pixels == p],
                                                      weights=data["w"][pixels == p]) for p in pix_todo]).T

        # p_gal /= hp.pixelfunc.nside2pixarea(nside,degrees=False) # for the integral in dOmega

        p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls into an empty pixel

        N_gal = np.sum(data[f"pix{nside}"][None, :] == np.array(pix_todo)[:, None], axis=1)

        return p_gal, N_gal


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









def Gaussian(x,mu,sigma):
    return np.power(2*np.pi*(sigma**2), -0.5) * np.exp(-0.5*np.power((x-mu)/sigma,2.))

def sum_Gaussians_UCV(z_grid, mu, sigma, weights=None, lambda_cosmo={"H0":70,"Om0":0.3}):
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

    if len(mu)==0:
        return np.zeros_like(z_grid)
    
    z_grid = np.array(z_grid)[:, np.newaxis]
    mu     = np.array(mu)
    sigma  = np.array(sigma)

    if weights is None: weights = np.ones(len(mu))

    num  = Gaussian(z_grid, mu, sigma) * fLCDM.dV_dz(z_grid, lambda_cosmo)
    den  = np.trapz(num, z_grid, axis=0)

    # print(np.sum(num/den, axis=1))

    return np.sum(weights*num/den, axis=1)/np.sum(weights)



def sum_Gaussians(z_grid, mu, sigma, weights=None):
    """ Vectorized sum of multiple Gaussians on z_grid each one with its own weight, mean and standard deviation.

    Args:
        z_grid (np.ndarray): redshift grid
        mu (np.ndarray): mean redshifts
        sigma (np.ndarray): redshifts standard deviations
        weights (np.ndarray, optional): weigths. Defaults to np.ones(len(mu)).

    Returns:
        np.ndarray: sum of Gaussians
    """

    if len(mu)==0:
        return np.array([])
    
    z_grid = np.array(z_grid)[:, np.newaxis]
    mu     = np.array(mu)
    sigma  = np.array(sigma)

    if weights is None:
        weights = np.ones(len(mu))

    num  = weights * Gaussian(z_grid, mu, sigma)
    den = np.trapz(num, z_grid, axis=0)

    return np.sum(num/den, axis=1)


def UVC_distribution(z_grid, lambda_cosmo={"H0":70,"Om0":0.3}):
    """ TBD
    """
    norm = fLCDM.V(np.max(z_grid), lambda_cosmo) - fLCDM.V(np.min(z_grid), lambda_cosmo)
    return 4*np.pi*fLCDM.dV_dz(z_grid, lambda_cosmo) / norm


