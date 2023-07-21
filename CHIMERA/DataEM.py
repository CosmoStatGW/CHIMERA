#
#   This module handles I/O and preliminary computations related to the EM data.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import h5py
import numpy as np
import logging

log = logging.getLogger(__name__)

from CHIMERA.EM import Galaxies
from CHIMERA.utils import (presets, mags)
from CHIMERA.cosmo import fLCDM
from CHIMERA.Completeness import CompletenessMICEv2

__all__ = [
    "MockGalaxiesMICEv2",
    "GLADEPlus",
]




class MockGalaxiesMICEv2(Galaxies):

    def __init__(self, 
                 dir_catalog, 
                 dir_interpolant = None,
                 **kwargs):
        
        self._file        = dir_catalog
        self._file_interp = dir_interpolant

        super().__init__(**kwargs)


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

        self._completeness = CompletenessMICEv2
        

class GLADEPlus(Galaxies):

    def __init__(self,
                 dir_cat,
                 nside=None,
                 **kwargs):
        
        self.dir_cat  = dir_cat
        self.all_keys = ['ra', 'dec', 'z', 'sigmaz', 'm_B', 'm_K', 'm_W1', 'm_bJ']
        
        Galaxies.__init__(self, nside=nside, **kwargs)


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
            log.info("Applying luminosity cut...")
            colm = "m_"+band
            colL = "L_"+band

            if colm not in data.keys():
                ValueError("ERROR, band not present in the catalog")

            if colL not in data.keys():

                dL   = fLCDM.dL(data["z"], presets.lambda_cosmo_GLADE)
                Mabs = data[colm] - 5 * np.log10(dL) - 40  # dL in [Gpc]
                L    = mags.Mag2lum(Mabs, band)
                data[colL] = L
                keys.append(colL)

    
            L_th = Lcut * mags.Lstar_default(band)

            mask = data[colL] > L_th
            for key in keys:
                data[key] = data[key][mask]

            log.info(" > L_{} cut, L > {:.1e}Lo: kept {:d} galaxies ({:.1f}%)".format(band, L_th, mask.sum(), 100*mask.sum()/len(mask)))

        
        if "sigmaz" in data: data["z_err"] = data.pop("sigmaz")

        self.data = data

       

    # def get_mask_Lcut(self, band, level):


        # return mask






def cut_dataEM_given_dataGW(dataGAL, dataGWi, init_sky_cut=4, dL_cut=None, H0_prior_range=None, conf_level_KDE=None, add_cat_keys=[]):
    """Function that cuts the volume of a galaxy catalog according to the posteriors in (z, RA, DEC) of a GW event

    Args:
        dataGAL (dict): galaxy catalog dictionary with "ra" and "dec" in [rad]
        dataGWi (dict): GW posterior dictionary with "ra" and "dec" in [rad]
        init_sky_cut (list or int, optional): percentile cut if `list`, mean \pm sigma cut if `int`. Defaults to 4.
        dL_cut (list or int, optional): activates dL cut if `list`, mean \pm sigma cut if `int`. Defaults to 4.
        H0_prior_range (list, optional): H0 prior range. Defaults to None.
        conf_level_KDE (float, optional): activates the sky KDE cut; credible level threshold. Defaults to None.

    Returns:
        dict: reduced EM catalog
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