#
#   This module handles I/O and computations related to the mock catalogs.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import logging

log = logging.getLogger(__name__)

import h5py
import numpy as np
import pandas as pd

from CHIMERA.Galaxies import GalCat
import CHIMERA.chimeraUtils as chimeraUtils

# class MockGalaxiesMICEv2(GalCat):

#     def __init__(self, 
#                  file_catalog,
#                  nside,
#                  **kwargs):
        
#         self._file = file_catalog
#         super().__init__(nside=nside, **kwargs)


#     def load(self,
#              z_err          = 5, 
#              units_ra_dec   = "deg",
#              columns        = ["ra_gal", "dec_gal", "z_cgal"]):

#         log.info("Loading mock galaxy catalog...")
#         df = pd.read_hdf(self._file, columns=columns)

#         if units_ra_dec == "deg":
#             df["ra_gal"]  = np.deg2rad(df["ra_gal"])
#             df["dec_gal"] = np.deg2rad(df["dec_gal"])

#         if z_err is not None:
#             log.info(" > setting galaxies' z uncertainties to {:.1e} x (1+z)".format(z_err))
#             df.loc[:,"z_err"] = z_err * (1. + df["z_cgal"])

#         df = df.rename(columns={"ra_gal":"ra", 
#                                 "dec_gal":"dec",
#                                 "z_cgal":"z"})
        
#         self.data = self.data.append(df, ignore_index=True)



class MockGalaxiesMICEv2(GalCat):

    def __init__(self, 
                 dir_catalog, 
                 nside, 
                 **kwargs):
        
        self._file = dir_catalog

        super().__init__(nside=nside, **kwargs)


    def load(self,
             z_err          = 1,
             units_ra_dec   = "deg",
             keys_load      = ["ra_gal", "dec_gal", "z_cgal"]):

        log.info("Loading mock galaxy catalog...")

        cat={}

        with h5py.File(self._file, 'r') as file:
            for k in keys_load:
                cat[k] = np.array(file[k][()])

        if units_ra_dec == "deg":
            log.info(" > converting (RA,DEC) to [rad]")
            cat["ra_gal"]  = np.deg2rad(cat["ra_gal"])
            cat["dec_gal"] = np.deg2rad(cat["dec_gal"])

        if z_err is not None:
            log.info(" > setting galaxies' z uncertainties to {:.1e} * (1+z)".format(z_err))
            cat["z_err"] = z_err * (1. + cat["z_cgal"])

        cat["ra"]  = cat.pop("ra_gal")
        cat["dec"] = cat.pop("dec_gal")
        cat["z"]   = cat.pop("z_cgal")

        self.data = cat




class MockGWData():

    """ 
        Class to handle mock GW data.
    """
    
    def __init__(self,
                 dir_catalog):
        
        self._file = dir_catalog


    def load(self, 
             Nevents   = None,
             Nsamples  = None,
             keys_load = ["m1det", "m2det", "chi1z", "phi", "dL", "theta", "chi2z", "iota"],
             ):
        
        """Load mock GW catalog calculating (RA,DEC) values.

        Args:
            Nevents (int, optional): number of events to load. Defaults to all events.
            Nsamples (int, optional): number of samples to load for each event. Defaults to all samples.
            keys_load (list, optional): list of keys to be loaded. Defaults to ["m1det", "m2det", "phi", "dL", "theta"].

        Returns:
            dict: dictionary of arrays containing the mock GW data.
        """

        data = {}

        log.info("Loading mock GW catalog...")

        with h5py.File(self._file, 'r') as f:

            Nevents_max, Nsamples_max = f["posteriors"][keys_load[0]].shape

            if Nevents is None: Nevents = Nevents_max
            if Nsamples is None: Nsamples = Nsamples_max

            for f_key in keys_load:
                if f_key not in f['posteriors'].keys(): 
                    log.warning(" > key {:s} not available".format(f_key))
                else:
                    data[f_key] = np.array(f["posteriors"][f_key])[:Nevents, :Nsamples]

        self.Nevents  = Nevents
        self.Nsamples = Nsamples

        log.info(" > loaded {:d} events with {:d} posterior samples each".format(self.Nevents, self.Nsamples))
        log.info(" > converting (theta,phi) to (RA,DEC) [rad] and dL [Gpc]")

        ra, dec      = chimeraUtils.ra_dec_from_th_phi(data["theta"], data["phi"])
        data["ra"]    = ra
        data["dec"]   = dec

        self.data = data

        return data
    

    def medians(self):

        """Return the median values of the parameters."""

        return {k:np.median(v, axis=1) for k,v in self.data.items()}