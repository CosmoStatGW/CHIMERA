import os, sys, logging
import numpy as np
log = logging.getLogger(__name__)

from .utils import misc

__all__ = ['Bias']



class Bias():

    """Class to compute the selection bias term. 

    >>> from CHIMERA.Bias import Bias
    >>> bias = Bias(...)
    >>> xi   = bias.compute(lambda_cosmo, lambda_mass, lambda_rate)

    Args:
        model_cosmo (CHIMERA.cosmo): cosmological model
        model_mass (CHIMERA.astro.mass): mass model
        model_rate (CHIMERA.astro.rate): rate model
        file_inj (str): path to the injection file
        N_inj (int, optional): number of injections. Defaults to None (i.e., the length of the injection catalog).
        snr_th (float, optional): SNR threshold. Defaults to None (i.e., no SNR cut applied).
        p_bkg (callable, optional): background probability function. Defaults to None (in this case, it uses `model_cosmo.dV_dz`).
        z_det_range (list, optional): redshift range for the redshift probability normalization. Defaults to None (i.e., no redshift normalization applied).
        z_int_res (int, optional): resolution for the redshift integral. Defaults to 1000.
        neff_inj_min (int, optional): minimum number of effective injections. Defaults to 5.
        Tobs (float, optional): observation time. Defaults to 1.
    """       


    def __init__(self,
                 model_cosmo,
                 model_mass,
                 model_rate,
                 file_inj, 
                 N_inj        = None, 
                 snr_th       = None, 
                 p_bkg        = None,
                 z_det_range  = None,
                 z_int_res    = 1000,
                 neff_inj_min = 5,
                 Tobs         = 1,
                 ):

        self.dir_file     = file_inj
        self.N_inj        = N_inj
        self.snr_th       = snr_th  

        self.model_cosmo  = model_cosmo
        self.model_mass   = model_mass
        self.model_rate   = model_rate

        self.p_bkg        = p_bkg if (p_bkg is not None) else model_cosmo.dV_dz

        self.normalized   = False if z_det_range is None else True
        self.Tobs         = Tobs if ~self.normalized else 1.
        self.z_int_res    = z_int_res
        self.z_det_range  = z_det_range

        self.neff_inj_min = neff_inj_min
        self.check_Neff   = True if self.neff_inj_min is not None else False  

        self._load()



    def _load(self):
        """Helper function to load the injection data
        """
        
        data_inj = misc.load_data_h5(self.dir_file)

        log_weights_sel = data_inj['log_p_draw_nospin']

        self.keep = np.full(len(log_weights_sel), True)

        log.info("Loading injections...")

        if self.N_inj is None:
            log.warning("  > N_inj not given. Assuming the injection files contains all the injections (no SNR cut applied)!")
            self.N_inj = len(data_inj['dL'])

        if self.snr_th is not None:
            self.keep = self.keep & (data_inj['SNR_net'] > self.snr_th)
            log.info(f"  > Injections SNR threshold set to {self.snr_th}")

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

        log.info(f"  > Total detected injections: {self.keep.shape}")
        log.info(f"  > Total used injections: {self.keep.sum()}")

        self.data_inj = {"m1det":m1z[self.keep], "m2det":m2z[self.keep], "dL":dL[self.keep],  "log_w":log_weights_sel[self.keep]}
        self.data_inj["w"] = np.exp(self.data_inj["log_w"])

        return self.data_inj
    


    def compute(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Compute the bias via Monte Carlo integration given the hyperparameters. Checks for N effective.

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters
        
        Returns:
            float: bias value
        """

        dN   = self._get_likelihood(lambda_cosmo, lambda_mass, lambda_rate)
        xi   = np.sum(dN) / self.N_inj

        neff = misc.get_Neff(dN, xi, Ndraw=self.N_inj)

        if (neff < self.neff_inj_min) and self.check_Neff:
            log.warning(f"Neff = {neff:.1f} < 5 for injections. Returning zero prob.")
            return 0.
        
        return xi



    def _get_likelihood(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Likelihood of the injection data given the hyperparameters 

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: likelihood
        """

        z      = np.array(self.model_cosmo.z_from_dL(self.data_inj["dL"], lambda_cosmo))
        m1, m2 = self.data_inj["m1det"]/(1.+z),  self.data_inj["m2det"]/(1.+z)

        dN_dm1dm2dz    = self.Tobs * self.model_mass(m1, m2, lambda_mass) *\
                         np.array(self.p_bkg(z, lambda_cosmo)) * self.model_rate(z, lambda_rate)/(1.+z)

        dN_dm1zdm2zddL = dN_dm1dm2dz / ((1.+z)**2 * np.array(self.model_cosmo.ddL_dz(z, lambda_cosmo, self.data_inj["dL"])))

        dN             = dN_dm1zdm2zddL/self.data_inj["w"] 

        if self.normalized: 
            norm   = self._Nexp(lambda_cosmo, lambda_rate)
            dN    /= norm
        
        return dN
        


    def _Nexp(self, lambda_cosmo, lambda_rate):
        """Expected number of events in the Universe given the hyperparameters

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: Nexp
        """

        zz    = np.linspace(*self.z_det_range, self.z_int_res)
        dN_dz = self.model_rate(zz, lambda_rate)/(1.+zz) * np.array(self.p_bkg(zz, lambda_cosmo))
        res   = np.trapz(dN_dz, zz)

        return res

    

    def _get_loglikelihood(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Likelihood of the injection data given the hyperparameters (log version)

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: likelihood
        """

        z          = self.model_cosmo.z_from_dL(self.data_inj["dL"], lambda_cosmo)
        m1, m2     = self.data_inj["m1det"]/(1.+z), self.data_inj["m2det"]/(1.+z)
        log_p_draw = self.data_inj["log_w"]

        dN_dm1dm2dz    = np.log(self.Tobs) + self.model_mass(m1, m2, lambda_mass) +\
                         self.model_cosmo.log_dV_dz(z, lambda_cosmo) + self.model_rate(z, lambda_rate) - np.log(1.+z)
        
        dN_dm1zdm2zddL = dN_dm1dm2dz - 2*np.log(1+z) - self.model_cosmo.log_ddL_dz(z, lambda_cosmo, self.data_inj["dL"])

        dN             = dN_dm1zdm2zddL - log_p_draw

        if self.normalized: 
            norm   = self.logNexp(lambda_cosmo, lambda_rate)
            dN     = dN - norm
        
        return dN



    def _logNexp(self, lambda_cosmo, lambda_rate):
        """Expected number of events in the Universe given the hyperparameters (log version)

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_rate (dict): rate hyperparameters

        Returns:
            float: Nexp
        """

        zz    = np.linspace(*self.z_det_range, self.z_int_res)
        dN_dz = self.model_cosmo.log_dV_dz(zz, lambda_cosmo) + self.model_rate(zz, lambda_rate) - np.log(1.+zz) 

        return np.log(np.trapz(np.exp(dN_dz), zz))



    def _compute_log(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Compute the bias via Monte Carlo integration given the hyperparameters. Checks for N effective (log version).

        Args:
            lambda_cosmo (dict): cosmological hyperparameters
            lambda_mass (dict): mass hyperparameters
            lambda_rate (dict): rate hyperparameters
        
        Returns:
            float: bias value
        """

        log_dN   = self.get_loglikelihood(lambda_cosmo, lambda_mass, lambda_rate)

        log_mu   = np.logaddexp.reduce(log_dN) - np.log(self.N_inj)

        log_s2   = np.logaddexp.reduce(2*log_dN) - 2*np.log(self.N_inj)
        log_sig2 = misc.logdiffexp(log_s2, 2.0*log_mu-np.log(self.N_inj))
        Neff     = np.exp(2.0*log_mu - log_sig2)

        if (Neff < self.inj_Neff) and self.check_Neff:
            log.warning(f"Neff = {Neff:.1f} < {self.inj_Neff} for injections. Returning 0 logprob")
            return 0
        
        return log_mu
