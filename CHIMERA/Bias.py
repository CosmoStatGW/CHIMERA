import os, sys
import numpy as np
import logging
log = logging.getLogger(__name__)

from CHIMERA.utils import misc

__all__ = ['Bias']


class Bias():

    def __init__(self, 
                 dir_file, 
                 N_inj, 
                 model_mass,
                 model_rate,
                 model_cosmo,
                 snr_th       = None, 
                 normalized   = False, 
                 Tobs         = 1,
                 z_int_res    = 1000,
                 z_det_range  = [0,1.3],
                 check_Neff   = False):

        self.dir_file    = dir_file
        self.N_inj       = N_inj
        self.snr_th      = snr_th  

        self.model_mass  = model_mass
        self.model_rate  = model_rate
        self.model_cosmo = model_cosmo

        self.normalized  = normalized
        self.Tobs        = Tobs if ~normalized else 1.
        self.z_int_res   = z_int_res
        self.z_det_range = z_det_range
        self.check_Neff  = check_Neff

        self.load()


    def load(self):
        
        data_inj = misc.load_data_h5(self.dir_file)

        log_weights_sel = data_inj['log_p_draw_nospin']

        self.keep = np.full(len(log_weights_sel), True)

        if self.snr_th is not None:
            self.keep = self.keep & (data_inj['SNR_net'] > self.snr_th)
            print('Injections SNR threshold set to %s' %self.snr_th)

        if not "m1_det" in data_inj.keys():
            dL   = data_inj['dL'] # Gpc
            z    = data_inj['z']
            m1z  = data_inj['m1src']*(1.+z)
            m2z  = data_inj['m2src']*(1.+z)
        else:
            m1z = data_inj['m1_det']
            m2z = data_inj['m2_det']
            dL  = data_inj['dL'] # Gpc

        assert (m1z > 0).all()
        assert (m2z > 0).all()
        assert (dL > 0).all()
        assert (m2z<=m1z).all()

        print('Total detected injections: %s' %self.keep.shape)
        print('Total used injections: %s' %self.keep.sum())

        self.data_inj = {"m1det":m1z[self.keep], "m2det":m2z[self.keep], "dL":dL[self.keep],  "log_w":log_weights_sel[self.keep]}

        return self.data_inj
    

    def get_likelihood(self, lambda_mass, lambda_cosmo, lambda_rate):

        z      = self.model_cosmo.z_from_dL(self.data_inj["dL"]*1000., lambda_cosmo)
        m1, m2 = self.data_inj["m1det"]/(1.+z),  self.data_inj["m2det"]/(1.+z)
        p_draw = np.exp(self.data_inj["log_w"])

        dN_dm1dm2dz    = self.Tobs * self.model_mass(m1, m2, lambda_mass) *\
                         1e-9*self.model_cosmo.dV_dz(z, lambda_cosmo) * self.model_rate(z, lambda_rate)/(1.+z)
        
        dN_dm1zdm2zddL = dN_dm1dm2dz / ((1.+z)**2 * 1e-3*self.model_cosmo.ddL_dz(z, lambda_cosmo, self.data_inj["dL"]*1000.))

        dN             = dN_dm1zdm2zddL/p_draw

        if self.normalized: 
            norm   = self.Nexp(lambda_cosmo, lambda_rate)
            dN    /= norm
        
        return dN
        


    def Nexp(self, lambda_cosmo, lambda_rate):
        """Expected number of events in the Universe given the hyperparameters

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: Nexp
        """

        zz    = np.linspace(*self.z_det_range, self.z_int_res)
        dN_dz = 1e-9*self.model_cosmo.dV_dz(zz, lambda_cosmo) *  self.model_rate(zz, lambda_rate)/(1.+zz) 

        return np.trapz(dN_dz, zz)

    
    def Ndet(self, lambda_mass, lambda_cosmo, lambda_rate):

        dN   = self.get_likelihood(lambda_mass, lambda_cosmo, lambda_rate)

        mu   = np.sum(dN) / self.N_inj

        s2   = np.sum(dN**2) / self.N_inj**2
        sig2 = s2 - mu**2 / self.N_inj
        Neff = mu**2 / sig2

        if (Neff < 5) and self.check_Neff:
            log.warning(f"Neff = {Neff:.1f} < 5 for injections. Returning zero prob.")
            return 0.
        
        return mu

    


    def get_loglikelihood(self, lambda_mass, lambda_cosmo, lambda_rate):

        z          = self.model_cosmo.z_from_dL(self.data_inj["dL"]*1000., lambda_cosmo)
        m1, m2     = self.data_inj["m1det"]/(1.+z), self.data_inj["m2det"]/(1.+z)
        log_p_draw = self.data_inj["log_w"]

        dN_dm1dm2dz    = np.log(self.Tobs) + self.model_mass(m1, m2, lambda_mass) +\
                         self.model_cosmo.log_dV_dz(z, lambda_cosmo)-9*np.log(10) + self.model_rate(z, lambda_rate) - np.log(1.+z)
        
        dN_dm1zdm2zddL = dN_dm1dm2dz - 2*np.log(1+z) - self.model_cosmo.log_ddL_dz(z, lambda_cosmo, self.data_inj["dL"]*1000.)+3*np.log(10)

        dN             = dN_dm1zdm2zddL - log_p_draw

        if self.normalized: 
            norm   = self.logNexp(lambda_cosmo, lambda_rate)
            dN     = dN - norm
        
        return dN


    def logNexp(self, lambda_cosmo, lambda_rate):
        """Expected number of events in the Universe given the hyperparameters

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: Nexp
        """

        zz    = np.linspace(*self.z_det_range, self.z_int_res)
        dN_dz = self.model_cosmo.log_dV_dz(zz, lambda_cosmo)-9*np.log(10) + self.model_rate(zz, lambda_rate) - np.log(1.+zz) 

        return np.log(np.trapz(np.exp(dN_dz), zz))


    def logNdet(self, lambda_mass, lambda_cosmo, lambda_rate):

        log_dN   = self.get_loglikelihood(lambda_mass, lambda_cosmo, lambda_rate)

        log_mu   = np.logaddexp.reduce(log_dN) - np.log(self.N_inj)

        log_s2   = np.logaddexp.reduce(2*log_dN) - 2*np.log(self.N_inj)
        log_sig2 = misc.logdiffexp(log_s2, 2.0*log_mu-np.log(self.N_inj))
        Neff     = np.exp(2.0*log_mu - log_sig2)

        #if Nobs is not None:# and verbose:
            #muSq = np.exp(2*logMu)
            #SigmaSq = np.exp(logSigmaSq)

        
        # s2   = np.sum(dN**2) / self.N_inj**2
        # sig2 = s2 - mu**2 / self.N_inj
        # Neff = mu**2 / sig2

        if (Neff < 5) and self.check_Neff:
            log.warning(f"Neff = {Neff:.1f} < 5 for injections. Returning zero prob.")
            return 0.
        
        return np.exp(log_mu)


#####################################################################################
##################################################################################### OLD (to be removed)
#####################################################################################


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

        injections_load = misc.load_data_h5(fname_inj)
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
    