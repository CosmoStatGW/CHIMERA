#
#   This Temporary module contains classes for the completeness correction.
#   Makes intensive use of MGCosmoPop/SelectionBias module <https://github.com/CosmoStatGW/MGCosmoPop> (Mancarella+2021).
#

import os, sys
import CHIMERA.chimeraUtils as chimeraUtils

from CHIMERA.Mock import MockGWInjectionsGWFAST

import logging
log = logging.getLogger(__name__)

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


    def load_pop(self, fname_inj, lambda_cosmo, lambda_rate, lambda_mass, dist_unit='Gpc'):

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
        self.injMock    = MockGWInjectionsGWFAST(injections_load,  N_gen = 50000000)

        self.allPopsMD  = AllPopulations(Cosmo(dist_unit))

        self.allPopsMD.add_pop(AstroPopulation(AstroPhRateEvolution(normalized=True), 
                                               PowerLawPlusPeakMass(), 
                                               DummySpinDist()))
        jointpop = {}
        [jointpop.update(d) for d in [lambda_cosmo, lambda_rate, lambda_mass]]
        jointpopMGCP = {self.key2MGCP.get(key, key): value for key, value in jointpop.items()}
        self.allPopsMD.set_values( jointpopMGCP )


    def update_bias_fcn(self, param_todo, zmax):
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
                                                  normalized=True)

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

        return self.fcn1D.Ndet(x)[0][0]


    