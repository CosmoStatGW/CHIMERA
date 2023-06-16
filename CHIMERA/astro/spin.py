import numpy as np
from scipy.special import erfc


def logpdf_truncGauss(x, mu=1, sigma=1, lower=0, upper=100):
    Phialpha = 0.5 * erfc(-(lower - mu) / (np.sqrt(2) * sigma))
    Phibeta = 0.5 * erfc(-(upper - mu) / (np.sqrt(2) * sigma))
    
    return np.where((x > lower) & (x < upper),
                   -np.log(2 * np.pi) / 2 - np.log(sigma) - np.log(Phibeta - Phialpha) - (x - mu) ** 2 / (2 * sigma ** 2),
                   -np.inf)



######################################################
###################################################### Dummy
######################################################


def logpdf_dummy(chi1, chi2, lambda_spin={}):
    """Dummy spin distribution

    Args:
        chi1 (np.ndarray): spin1
        chi2 (np.ndarray): spin2
        lambda_spin (dict): parameters of the spin function with keys: ["muEff", "sigmaEff", "muP", "sigmaP", "rho"]

    Returns:
        np.ndarray: log(pdf)
    """
    return np.zeros(chi1.shape[0])



######################################################
###################################################### Uniform
######################################################

def logpdf_U(chi1z, chi2z, lambda_spin):

    chi_min, chi_max = lambda_spin["chi_min"], lambda_spin["chi_max"]
    logN             = -np.log(chi_max - chi_min)

    return np.where((chi1z > chi_min) & (chi1z < chi_max) & (chi2z > chi_min) & (chi2z < chi_max), logN, -np.inf)





######################################################
###################################################### Gauss
######################################################



def logpdf_G(chi_eff, chi_p, lambda_s):
    # """Gaussian spin distribution

    lpar = ["mu_eff", "sigma_eff", "mu_p", "sigma_p"]
    mu_eff, sigma_eff, mu_p, sigma_p = [lambda_s[p] for p in lpar]

    logpdf_eff = logpdf_truncGauss(chi_eff, mu=mu_eff, sigma=sigma_eff, lower=-1, upper=1)
    logpdf_p   = logpdf_truncGauss(chi_p, mu=mu_p, sigma=sigma_p, lower=0, upper=1)

        
    return logpdf_eff + logpdf_p














class UniformSpinDistChiz(BBHDistFunction): 
    
    '''
    Uniform distribution in chi1z, chi2z, uncorrelated
    '''
    
    def __init__(self, ):
        BBHDistFunction.__init__(self)
        self.params = ['chiMin', 'chiMax', ] #'rho' ] # For the moment we ignore correlation
        
        self.baseValues = {
                           
                           'chiMin': -0.75,
                           'chiMax':0.75, 
                           }
        
        self.names = {
                           'chiMin':r'$\chi_{ z, Min}$',
                           'chiMax':r'$\chi_{ z, Max}$', }
         
        self.n_params = len(self.params)
    
        self.maxChi = 1
        self.minChi = -1
        self.nVars = 2
        

        
        print('Uniform spin distribution in (chi1z, chi2z) base values: %s' %self.baseValues)
    

    