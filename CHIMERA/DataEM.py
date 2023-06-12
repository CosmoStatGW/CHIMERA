#
#   This module handles I/O and computations related to the mock catalogs.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#


import h5py
import numpy as np


from CHIMERA.Galaxies import GalCat
import CHIMERA.chimeraUtils as chimeraUtils

import logging
log = logging.getLogger(__name__)
from CHIMERA.cosmo import fLCDM




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




class GLADEPlus_v3(GalCat):

    def __init__(self,
                 dir_cat,
                 nside=None,
                 **kwargs):
        
        self.dir_cat  = dir_cat
        self.all_keys = ['ra', 'dec', 'z', 'sigmaz', 'm_B', 'm_K', 'm_W1', 'm_bJ']
        
        GalCat.__init__(self, nside, **kwargs)


    def load(self, 
             keys=None,
             Lcut=None,
             band=None,
             ):

        if keys is None:
            keys = self.all_keys


        # Load data from a hdf5 file in the most efficient and less memory consuming way (do not use pandas)
        with h5py.File(self.dir_cat, 'r') as f:
            data = {key: np.array(f[key]) for key in keys}

        if Lcut is not None:
            mask = self.get_mask_Lcut(band, Lcut)
            for key in keys:
                data[key] = data[key][mask]

        
        if "sigmaz" in data: data["z_err"] = data.pop("sigmaz")

        self.data = data

       

    def get_mask_Lcut(self, band, level):

        colm = "m_"+band
        colL = "L_"+band

        if colm not in self.data.keys():
            ValueError("ERROR, band not present in the catalog")

        if colL not in self.data.keys():

            dL   = fLCDM.dL(self.data["z"], chimeraUtils.lambda_cosmo_GLADE)
            Mabs = self.data[colm] - 5 * np.log10(dL) - 25  # dL in [Mpc]
            L    = chimeraUtils.Mag2lum(Mabs, band)
            self.data.update({colL: L})

 
        L_th = level * chimeraUtils.Lstar_default(band)

        mask = self.data[colL] > L_th
        # log.info(" > applied Lcut on band {} at L > {:.1e}".format(band, L_th))
        print(" > L_{} cut, L > {:.1e}Lo: kept {:d} galaxies ({:.1f}%)".format(band, L_th, mask.sum(), 100*mask.sum()/len(mask)))

        return mask