#
#   This Temporary module contains classes for the completeness correction.
#   Makes intensive use of MGCosmoPop/SelectionBias module <https://github.com/CosmoStatGW/MGCosmoPop> (Mancarella+2021).
#

import os, sys
import numpy as np
import CHIMERA.chimeraUtils as chimeraUtils


import logging
log = logging.getLogger(__name__)



class MockGWInjectionsGWFAST():
    
    def __init__(self, 
                 injections_df, 
                 N_gen,
                 dist_unit="Gpc", 
                 snr_th=None,
                 Tobs=1, 
                 
                #DeltaOm=None,
                 #nInjUse=None,  
                 ):
        
        self.N_gen      = N_gen
        self.dist_unit  = dist_unit
        self.m1z, self.m2z, self.dL, self.log_weights_sel  = self._load_data(injections_df, snr_th)
        self.logN_gen   = np.log(self.N_gen)

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
    
    
    def _load_data(self, injections_df, snr_th=None):
        
        log_weights_sel = injections_df['log_p_draw_nospin']

        self.keep = np.full(len(log_weights_sel), True)

        if snr_th is not None:
            self.keep = self.keep & (injections_df['SNR_net'] > snr_th)
            print('Injections SNR threshold set to %s' %snr_th)

        print('Total detected injections: %s' %self.keep.sum())

        if not "m1_det" in injections_df.keys():
            dL   = injections_df['dL'] # Gpc
            z    = injections_df['z']
            m1z  = injections_df['m1src']*(1.+z)
            m2z  = injections_df['m2src']*(1.+z)
        else:
            m1z = injections_df['m1_det']
            m2z = injections_df['m2_det']
            dL  = injections_df['dL'] # Gpc


        return m1z[self.keep], m2z[self.keep], dL[self.keep],  log_weights_sel[self.keep] 


class SelectionBias():

    def __init__(self, MGCP_dir):

        # Check if MGCosmoPop is present
        if not os.path.isdir(MGCP_dir):
            raise Exception("MGCosmoPop not found in default folder. Please provide correct path.")

        self.MGCP_dir = MGCP_dir

        # Dictionary to convert CHIMERA parameter names to MGCP parameter names
        self.key2MGCP = {'H0':'H0','Om0':'Om','gamma':'alphaRedshift','kappa':'betaRedshift', 
                'zp':'zp','R0':'R0','lambdaPeak':'lambdaPeak','beta':'beta','alpha':'alpha',
                'deltam':'deltam','ml':'ml','mh':'mh','muMass':'muMass','sigmaMass':'sigmaMass'}


    def load_pop(self, fname_inj, lambda_cosmo, lambda_rate, lambda_mass,
                 dist_unit='Gpc', snr_th=None, N_gen_tot=50000000):

        sys.path.append(self.MGCP_dir)
        from MGCosmoPop.population.astro.astroMassDistribution import PowerLawPlusPeakMass
        from MGCosmoPop.population.astro.rateEvolution import AstroPhRateEvolution
        from MGCosmoPop.population.astro.astroSpinDistribution import DummySpinDist
        from MGCosmoPop.population.astro.astroPopulation import AstroPopulation
        from MGCosmoPop.cosmology.cosmo import Cosmo
        from MGCosmoPop.population.allPopulations import AllPopulations
        from MGCosmoPop.posteriors.selectionBias import SelectionBiasInjections

        self.SelectionBiasInjections = SelectionBiasInjections

        injections_load = chimeraUtils.load_data_h5(fname_inj)
        self.injMock    = MockGWInjectionsGWFAST(injections_load,  N_gen = N_gen_tot, snr_th=snr_th)

        self.allPopsMD  = AllPopulations(Cosmo(dist_unit))

        self.allPopsMD.add_pop(AstroPopulation(AstroPhRateEvolution(normalized=True), 
                                               PowerLawPlusPeakMass(), 
                                               DummySpinDist()))
        jointpop = {}
        [jointpop.update(d) for d in [lambda_cosmo, lambda_rate, lambda_mass]]
        jointpopMGCP = {self.key2MGCP.get(key, key): value for key, value in jointpop.items()}
        self.allPopsMD.set_values( jointpopMGCP )


    def update_bias_fcn(self, param_todo, zmax, normalized):
        """Compute the bias function for a given parameter/s.

        Args:
            param_todo (str): parameter to compute the bias for
            lambda_cosmo (dict): dictionary of cosmological hyperparameters
            lambda_rate (dict): dictionary of rate hyperparameters
            lambda_mass (dict): dictionary of mass hyperparameters

        Returns:
            function: bias function
        """
        log.info("\nComputing bias function with MGCosmoPop for parameter {}...".format(param_todo))

        self.fcn1D = self.SelectionBiasInjections(self.allPopsMD, 
                                                  [self.injMock, ],
                                                  [self.key2MGCP[param_todo], ], 
                                                  zmax = zmax,
                                                  normalized=normalized)

    def get_bias_fcn(self, x):
        """Return the bias function for a given parameter/s.

        Args:
            param_todo (str): parameter to compute the bias for
            lambda_cosmo (dict): dictionary of cosmological hyperparameters
            lambda_rate (dict): dictionary of rate hyperparameters
            lambda_mass (dict): dictionary of mass hyperparameters

        Returns:
            function: bias function
        """

        return self.fcn1D.Ndet(x)



    def update_bias_fcn_ND(self, param_todo, zmax, normalized):
        """Compute the bias function for a given parameter/s.

        Args:
            param_todo (str): parameter to compute the bias for
            lambda_cosmo (dict): dictionary of cosmological hyperparameters
            lambda_rate (dict): dictionary of rate hyperparameters
            lambda_mass (dict): dictionary of mass hyperparameters

        Returns:
            function: bias function
        """
        log.info("\nComputing bias function with MGCosmoPop for parameter {}...".format(param_todo))

        self.fcnND = self.SelectionBiasInjections(self.allPopsMD, 
                                                  [self.injMock, ],
                                                  param_todo, 
                                                  zmax = zmax,
                                                  normalized=normalized)

    def get_bias_fcn_ND(self, x):
        """Return the bias function for a given parameter/s.

        Args:
            param_todo (str): parameter to compute the bias for
            lambda_cosmo (dict): dictionary of cosmological hyperparameters
            lambda_rate (dict): dictionary of rate hyperparameters
            lambda_mass (dict): dictionary of mass hyperparameters

        Returns:
            function: bias function
        """

        return self.fcnND.Ndet(x)