import logging
log = logging.getLogger(__name__)

from Galaxies import GalCat
from Completeness import SkipCompleteness

import h5py
import numpy as np
import pandas as pd

import DSutils


class MockGalaxiesMICEv2(GalCat):

    def __init__(self, 
                 file_catalog,
                 **kwargs):
        
        self._file = file_catalog
        GalCat.__init__(self, file_catalog, completeness=None, **kwargs)


    def load(self,
             z_err          = 5, 
             units_ra_dec   = "deg",
             columns        = ["ra_gal", "dec_gal", "z_cgal"]):

        log.info("Loading mock galaxy catalog...")
        df = pd.read_hdf(self._file, columns=columns)

        if units_ra_dec == "deg":
            df["ra_gal"]  = np.deg2rad(df["ra_gal"])
            df["dec_gal"] = np.deg2rad(df["dec_gal"])

        if z_err is not None:
            log.info(" > setting galaxies' z uncertainties as {}% of z".format(z_err))
            df.loc[:,"z_err"] = float(z_err)/100 * df["z_cgal"]

        df = df.rename(columns={"ra_gal":"ra", 
                                "dec_gal":"dec",
                                "z_cgal":"z"})
        
        self.data = self.data.append(df, ignore_index=True)



class MockGWInjectionsGWFAST():
    
    def __init__(self, 
                  injections_df, 
                  N_gen,
                 #DeltaOm=None,
                  #nInjUse=None,  
                  dist_unit="Mpc", Tobs=1, 
                  #snr_th=12,
                 #DelOm_th=100,
                  
                 ):
        
        self.N_gen=N_gen
        #self.snr_th=snr_th
        #self.DelOm_th=DelOm_th
        
        
        self.dist_unit=dist_unit
        self.m1z, self.m2z, self.dL, self.log_weights_sel  = self._load_data(injections_df )
        self.logN_gen = np.log(self.N_gen)
        assert (self.m1z > 0).all()
        assert (self.m2z > 0).all()
        assert (self.dL > 0).all()
        assert(self.m2z<=self.m1z).all()

        
        self.Tobs=Tobs
        self.spins = []
        print('Obs time: %s' %self.Tobs )
        
        self.condition = np.full(self.m1z.shape, True)
        
    def get_theta(self):
        return np.array( [self.m1z, self.m2z, self.dL  ] ) 
    
    
    def _load_data(self, injections_df,):
    
        m1_sel = injections_df['m1_det']
        m2_sel = injections_df['m2_det']
        dl_sel = injections_df['dL'] * 1000 # Mpc
        log_weights_sel = injections_df['log_p_draw_nospin']
        
        self.detected = np.full(len(m1_sel), True)
        print('Total detected injections: %s' %self.detected.sum())
       
        self.keep = self.detected
        return m1_sel[self.keep], m2_sel[self.keep], dl_sel[self.keep],  log_weights_sel[self.keep] 



















class MockGW():
    
    def __init__(self,
                 file_catalog):
        
        self._file = file_catalog


    def load(self, 
             Nevents,
             Nsamples,
             keys_skip = ['Phicoal', 'chi1z', 'chi2z', 'iota','psi', 'tcoal']
             ):
        log.info("Loading mock GW catalog...")

        evs={}
        with h5py.File(self._file, 'r') as phi: 
            #observations.h5 has to be in the same folder as this code

            log.info(" > N_events: {:d}".format(phi['posteriors'][keys_skip[0]].shape[0]))

            for pn in phi['posteriors'].keys():
                if pn not in keys_skip:
                    log.info(' > loading %s' %pn)
                    pn_last=pn
                    if pn!='rho':
                        if Nsamples is not None:
                            evs[pn] = np.array(phi['posteriors'][pn])[:Nevents, :Nsamples]
                        else:
                            evs[pn] = np.array(phi['posteriors'][pn])[:Nevents, :]
                    else:
                        evs[pn] = np.array(phi['posteriors'][pn])[:Nevents]
        
        log.info(" > converting (theta,phi) to (ra,dec) [rad] and dL [Mpc]")
        ra, dec   = DSutils.ra_dec_from_th_phi(evs["theta"], evs["phi"])
        del evs["phi"]
        del evs["theta"]
        evs["ra"]   = ra
        evs["dec"]  = dec
        evs["dL"]   = 1000*evs["dL"] # Mpc

        self.data = evs

        return evs
    

    def medians(self):
        """Return the median values of the parameters"""
        return {k:np.median(v, axis=1) for k,v in self.data.items()}