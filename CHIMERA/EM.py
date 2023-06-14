#
#   This module contains classes to handle galaxy catalogs.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy

import healpy as hp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

import CHIMERA.chimeraUtils as chimeraUtils
from CHIMERA.cosmo import fLCDM
from scipy.stats import norm, gaussian_kde


def sum_of_gauss_UVC_prior(z_grid, mu, sigma, weights=None):
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

    z_grid = np.array(z_grid)[:, np.newaxis]
    mu     = np.array(mu)
    sigma  = np.array(sigma)

    if weights is None:
        weights = np.ones(len(mu))

    num  = weights * norm.pdf(z_grid, mu, sigma) * fLCDM.dV_dz(z_grid, {"H0":70,"Om0":0.3})
    den  = np.trapz(num, z_grid, axis=0)

    return np.sum(num/den, axis=1)



def sum_of_gauss(z_grid, mu, sigma, weights=None):
    """ Vectorized sum of multiple Gaussians on z_grid each one with its own weight, mean and standard deviation.

    Args:
        z_grid (np.ndarray): redshift grid
        mu (np.ndarray): mean redshifts
        sigma (np.ndarray): redshifts standard deviations
        weights (np.ndarray, optional): weigths. Defaults to np.ones(len(mu)).

    Returns:
        np.ndarray: sum of Gaussians
    """

    z_grid = np.array(z_grid)[:, np.newaxis]
    mu     = np.array(mu)
    sigma  = np.array(sigma)

    if weights is None:
        weights = np.ones(len(mu))

    num  = weights * norm.pdf(z_grid, mu, sigma)
    den = np.trapz(num, z_grid, axis=0)

    return np.sum(num/den, axis=1)


def UVC_distribution(z_grid):
    """ TBD
    """
    norm = fLCDM.V(np.max(z_grid), {"H0":70,"Om0":0.3}) - fLCDM.V(np.min(z_grid), {"H0":70,"Om0":0.3})
    return 4*np.pi*fLCDM.dV_dz(z_grid, {"H0":70,"Om0":0.3}) / norm
    



class GalCat(ABC):
    
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

        self.data  = pd.DataFrame()
        self.load(**kwargs)
        self.selectedData = self.data

        if completeness is not None:
            self._completeness = deepcopy(completeness)
            self._useDirac     = useDirac
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
            log.info(f"Precomputing Healpixels for the galaxy catalog (NSIDE={n:d}, NEST={self.nest})")
            self.data[f"pix{n}"] = chimeraUtils.find_pix_RAdec(self.data['ra'], self.data['dec'], n, self.nest)


    def precompute(self, nside, pix_todo, z_grid, names=None, weights_kind="N"):
        """Pre-compute pixelized p_gal for many events.

        Args:
            nside (np.ndarray): nside Healpix parameter for each event
            pix_todo (list): arrays of pixels for each event
            z_grid (np.ndarray): 2D array of the redshift grids with shape (Nevents, z_int_res)
            names (list, optional): names of GW events, for logging purposes. Defaults to None.
            weights_kind (str, optional): weights_kind for p_gal. Defaults to "N".

        Returns:
            [p_gal], [p_gal_w]: probability p_gal and associated weigths
        """        
        Nevents = len(nside)
        assert Nevents == len(pix_todo) == len(z_grid)
        if names is None: names = [f"{i}/{Nevents}" for i in range(Nevents)]

        log.info(f"Precomputing p_GAL for {Nevents} events...")

        p_gal, p_gal_w = zip(*[self.precompute_event(nside[e], pix_todo[e], z_grid[e], names[e], weights_kind) for e in range(Nevents)])

        return p_gal, p_gal_w 


    def precompute_event(self, nside, pix_todo, z_grid, name, weights_kind="N"):
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
        p_gal  = np.vstack([sum_of_gauss_UVC_prior(z_grid, data["z"][pixels == p], data["z_err"][pixels == p]) for p in pix_todo]).T

        p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls into an empty pixel

        if weights_kind == "N":
            p_gal_w = np.sum(data[f"pix{nside}"][None, :] == np.array(pix_todo)[:, None], axis=1)
        else:
            p_gal_w = np.ones_like(p_gal)

        return p_gal, p_gal_w


    def select_event_region(self, z_min, z_max, pixels, nside):
        """Select a sub-catalog containing the galaxies inside the given redshift interval and Healpix pixels.

        Args:
            z_min (float): minimum redshift
            z_max (float): maximum redshift
            pixels (np.ndarray, int): Helpix pixels to select.
        """

        log.info(f"Setting sky area to {pixels.shape[0]} confident pixels (nside={nside:d})")
        
        pixel_key   = f"pix{nside}"
        mask        = np.isin(self.data[pixel_key], pixels)
        selected    = {k: v[mask] for k, v in self.data.items()}

        log.info(f" > kept {selected['z'].shape[0]} galaxies")
        log.info(f"Setting z range: {z_min:.3f} < z < {z_max:.3f}")

        z_mask      = (selected['z'] > z_min) & (selected['z'] < z_max)
        selected    = {k: v[z_mask] for k, v in selected.items()}

        log.info(f" > kept {selected['z'].shape[0]} galaxies")

        N_gal_pix = np.sum(selected[pixel_key][None, :] == pixels[:, None], axis=1)
        log.info(f" > mean {np.mean(N_gal_pix):.1f} galaxies per pixel")

        return selected
    

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

        print(" > KDE-GW cut with threshold {:.1f}: kept {:d} galaxies".format(conf_level_KDE, mask.sum()))

        self.data = {k : self.data[k][mask] for k in self.data.keys()}






def cutout_given_dataGW(dataGAL, dataGWi, init_sky_cut=4, dL_cut=None, H0_prior_range=None, conf_level_KDE=None, add_cat_keys=[]):
    """Function that cuts the volume of a galaxy catalog according to the posteriors in (z, RA, DEC) of a GW event

    Args:
        dataGAL (dict): galaxy catalog dictionary with "ra" and "dec" in [rad]
        dataGWi (dict): GW posterior dictionary with "ra" and "dec" in [rad]
        init_sky_cut (list or int, optional): percentile cut if `list`, mean \pm sigma cut if `int`. Defaults to 4.
        dL_cut (list or int, optional): activates dL cut if `list`, mean \pm sigma cut if `int`. Defaults to 4.
        H0_prior_range (list, optional): H0 prior range. Defaults to None.
        conf_level_KDE (float, optional): activates the sky KDE cut; credible level threshold. Defaults to None.

    Returns:
        dict: reduced galaxy catalog
    """   

    keys_GWi  = ["ra", "dec"] if H0_prior_range is None else ["ra", "dec", "dL"]
    keys_GAL  = ["ra", "dec"] if H0_prior_range is None else ["ra", "dec", "z"]
    keys_GAL.extend(add_cat_keys)

    dataGWi   = np.array([dataGWi[key] for key in keys_GWi])
    dataGAL   = np.array([dataGAL[key] for key in keys_GAL])

    # First of all remove the points outside of the square area
    if isinstance(init_sky_cut, list):
        sky     = np.percentile(dataGWi[[0,1],:], init_sky_cut, axis=1)
    elif isinstance(init_sky_cut, int):
        mu, sig = np.mean(dataGWi[[0,1],:], axis=1), np.std(dataGWi[[0,1],:], axis=1)
        sky     = np.array([mu-init_sky_cut*sig, mu+init_sky_cut*sig])
    else:
        ValueError("ERROR, init_sky_cut must be a list of percentiles or an integer")

    mask    = (np.all(dataGAL[[0,1],:]>=sky[0][:,np.newaxis], axis=0) & np.all(dataGAL[[0,1],:]<=sky[1][:,np.newaxis], axis=0) )
    dataGAL = dataGAL[:,mask]
    mask    = (np.all(dataGWi[[0,1],:]>=sky[0][:,np.newaxis], axis=0) & np.all(dataGWi[[0,1],:]<=sky[1][:,np.newaxis], axis=0) )
    dataGWi = dataGWi[:,mask]

    print(" > sky_cut: kept {:d} galaxies".format(dataGAL.shape[1]))
    
    # Then, remove points in the dL range, (in z for the galaxies, taking into account all the possible H0)
    if dL_cut is not None:
        if H0_prior_range is None:
            ValueError("ERROR, H0_prior_range must be provided if dL_cut is not None")
        if isinstance(dL_cut, list):
            dmin, dmax = np.percentile(dataGWi[2,:], dL_cut)
        elif isinstance(dL_cut, int):
            mu, sig    = np.mean(dataGWi[2,:]), np.std(dataGWi[2,:])
            dmin, dmax = mu-dL_cut*sig, mu+dL_cut*sig
        else:
            ValueError("ERROR, dL_cut must be a list of percentiles or an integer")

        if dmin<0: dmin=0

        zmin = fLCDM.z_from_dL(dmin, {"H0":H0_prior_range[0], "Om0":0.3})
        zmax = fLCDM.z_from_dL(dmax, {"H0":H0_prior_range[1], "Om0":0.3})

        dataGAL = dataGAL[:,(dataGAL[2,:]>=zmin) & (dataGAL[2,:]<=zmax)]
        dataGWi = dataGWi[:,(dataGWi[2,:]>=dmin) & (dataGWi[2,:]<=dmax)]

        print(" > z_cut for H0 in {:s}: kept {:d} galaxies".format(str(H0_prior_range), dataGAL.shape[1]))

    if conf_level_KDE is None:
        return {k:dataGAL[i] for i, k in enumerate(keys_GAL)}

    # Then, refine the sky cut with a KDE
    prob  = gaussian_kde(dataGWi[[0,1],:]).evaluate(dataGAL[[0,1],:])

    # 1. Sort the points based on their values
    sorted_idx  = np.argsort(prob)
    sorted_data = dataGAL[:,sorted_idx]
    sorted_prob = prob[sorted_idx]

    # 2. Find the most probable points
    cdf = np.cumsum(sorted_prob, axis=0)
    cdf /= cdf[-1]
    mask = (cdf >= 1-conf_level_KDE)
    
    print(" > KDE-GW cut with threshold {:.1f}: kept {:d} galaxies".format(conf_level_KDE, mask.sum()))

    return {k:sorted_data[i,mask] for i, k in enumerate(keys_GAL)}
