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
from CHIMERA.cosmologies import fLCDM
from scipy.stats import norm


def sum_of_gaussians(z_grid, mu, sigma, weights=None):
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

    dVdz     = fLCDM.dV_dz(z_grid, {"H0":70,"Om0":0.3})
    gauss    = norm.pdf(z_grid, mu, sigma)
    integral = np.trapz(dVdz*gauss, z_grid, axis=0)

    return np.sum(weights * dVdz * gauss/integral, axis=1)
    # return np.sum(weights * dVdz * gauss, axis=1)


class GalCat(ABC):
    
    def __init__(self, 
                 fname,

                 # Pixellization
                 nside = None,
                 nest  = False,
                 
                 # Completeness
                 completeness = None,
                 useDirac = None,

                 **kwargs,
                 ):
        
        self._path = fname

        self.nside = np.atleast_1d(nside)
        self.nest  = nest

        self.data  = pd.DataFrame()
        self.load(**kwargs)
        self.selectedData = self.data

        if completeness is not None:
            self._completeness          = deepcopy(completeness)
            self._useDirac              = useDirac
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
        for nsid in np.unique(self.nside):
            log.info("Precomputing Healpixels for the galaxy catalog (NSIDE={:d}, NEST={})".format(nsid,self.nest))

            self.data.loc[:,"pix"+str(nsid)] = chimeraUtils.find_pix_RAdec(self.data.ra.to_numpy(),self.data.dec.to_numpy(),nsid,self.nest)
        
        # coln  = "pix"+str(nside)
        # N_gal_pix = np.array([sum(self.selectedData["pix"+str(nside)] == pix) for pix in pixels])


    def precompute(self, nside, pix_todo, z_grid, names=None, weights="N"):

        Nevents  = len(nside)
        assert Nevents == len(pix_todo) == len(z_grid)
        log.info("Precomputing p_GAL for {:d} events...".format(Nevents))

        # To be improved
        res = []
        wei = []

        for e in range(Nevents):

            if names is None:
                log.info("### Event {:d}/{:d} ###".format(e+1, Nevents))
            else:
                log.info("### {:s} ###".format(names[e]))

            r, data = self.compute_event(nside[e], pix_todo[e], z_grid[e])
            
            if weights == "N":
                w = np.array([sum(data["pix"+str(nside[e])] == p) for p in pix_todo[e]])
            else:
                w = np.ones_like(r)

            res.append(r)
            wei.append(w)
            
        return res, wei


    def compute_event(self, nside, pix_todo, z_grid):

        data   = self.select_event_region(z_grid[0], z_grid[-1], pix_todo, nside)
        pixels = data["pix"+str(nside)]
        p_gal  = np.vstack([sum_of_gaussians(z_grid, data.z[pixels == p], data.z_err[pixels == p]) for p in pix_todo]).T
        # p_gal /= np.trapz(p_gal, z_grid, axis=0) 
        p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls into an empty pixel

        return p_gal, data


    def select_event_region(self, z_min, z_max, pixels, nside):
        """Select a sub-catalog containing the galaxies inside the given redshift interval and Healpix pixels.

        Args:
            z_min (float): minimum redshift
            z_max (float): maximum redshift
            pixels (np.ndarray, int): Helpix pixels to select.
        """
        
        log.info("Setting sky area to {:d} confident pixels (nside={:d})".format(pixels.shape[0], nside))
        mask              = self.data.isin({"pix"+str(nside): pixels}).any(1)
        selected          = self.data[mask]
        log.info(" > kept {:d} galaxies".format(selected.shape[0]))
        log.info("Setting z range: {:.3f} < z < {:.3f}".format(z_min, z_max,3))
        selected          = selected[(selected.z >= z_min) & (selected.z < z_max)]
        log.info(" > kept {:d} galaxies".format(selected.shape[0]))
        N_gal_pix         = np.array([sum(selected["pix"+str(nside)] == pix) for pix in pixels])
        log.info(" > mean {:.1f} galaxies per pixel".format(np.mean(N_gal_pix)))
        
        return selected
