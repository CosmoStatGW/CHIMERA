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




