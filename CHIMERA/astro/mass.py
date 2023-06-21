import numpy as np
from scipy.special import erf



def _logpdf_TPL(x, alpha, mmin, mmax):
    norm_const = (1 - alpha) / (mmax**(1 - alpha) - mmin**(1 - alpha))
    return -alpha*np.log(x) + np.log(norm_const)

def _logpdf_G(x, mu, sigma):
    return -0.5*np.log(2 * np.pi) -np.log(sigma) - (x-mu)**2/(2. *sigma**2)

def _logSmoothing(m, delta_m, ml):
    # Smoothing function
    maskL      = m <= ml
    maskU      = m >= (ml + delta_m)
    conditions = [maskL, maskU, ~(maskL | maskU)]
    functions  = [-np.inf, 0., lambda x: -np.logaddexp(0., (delta_m/(x-ml) + delta_m/(x-ml-delta_m)))]
    return np.piecewise(m, conditions, functions)


######################################################
###################################################### Dummy
######################################################

def dummy_mass(m1, m2, lambda_m):
    return np.ones_like(m1)

######################################################
###################################################### PL
######################################################

def _logpdfm2_PL(m2, beta, ml):
    # Conditional distribution p(m2 | m1)
    return np.where(m2 >= ml, beta*np.log(m2), -np.inf)

def _logC_PL(m1, beta, ml):
    return np.log((1+beta) / (m1**(1+beta) - ml**(1+beta)))

def logpdf_PL(m1, m2, lambda_m):
    """Power-law mass distribution, p(m1,m2|lambda_m) normalized

    Args:
        m1 (np.ndarray): primary mass
        m2 (np.ndarray): secondary mass
        lambda_m (dict): parameters of the mass function with keys: 
                         ["alpha", "beta", "ml", "mh"]
    """
    # Unpack parameters
    lpar = ["alpha", "beta", "ml", "mh"]
    alpha, beta, ml, mh = [lambda_m[p] for p in lpar]

    return np.where((ml < m2) & (m2 < m1) & (m1 < mh),
                    
                    # compute logprob
                    _logpdf_TPL(m1, alpha, ml, mh) + _logpdfm2_PL(m2, beta, ml) + _logC_PL(m1, beta, ml),

                    # return zero probability
                    -np.inf)

def pdf_PL(m1, m2, lambda_m):
    return np.exp(logpdf_PL(m1, m2, lambda_m))


######################################################
###################################################### Broken PL
######################################################

# TBD



######################################################
###################################################### Smooth PL
######################################################


def _logpdfm2_SPL(m2, beta, delta_m, ml):
    # Conditional distribution p(m2 | m1)

    return np.where(m2 >= ml, 
                    beta*np.log(m2) + _logSmoothing(m2, delta_m, ml), 
                    -np.inf)

def _logC_SPL(m1, beta, delta_m, ml, res=200):
    # Inverse log integral of PL p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation)

    mmid = ml + delta_m + delta_m/10.
    mm   = np.concatenate([np.linspace(ml, mmid, 200),
                           np.linspace(mmid + 1e-1, np.max(m1), res)])
    mm   = np.sort(mm)
    p2   = np.exp(_logpdfm2_SPL(mm, beta, delta_m, ml))
    cdf  = np.cumsum(0.5*(p2[:-1] + p2[1:]) * np.diff(mm))

    return -np.log(np.interp(m1, mm[1:], cdf))


# TBD logpdf


######################################################
###################################################### PL + Peak
######################################################

def _logpdfm1_PLP(m1, lambda_peak, alpha, delta_m, ml, mh, mu_g, sigma_g):
    # Marginal distribution p(m1), not normalised
    P = np.exp(_logpdf_TPL(m1, alpha, ml, mh))
    G = np.exp(_logpdf_G(m1, mu_g, sigma_g))

    return np.where((m1 >= ml) & (m1 <= max(mh, mu_g+10*sigma_g)), 
                    np.log((1-lambda_peak)*P + lambda_peak*G) + _logSmoothing(m1, delta_m, ml), 
                    -np.inf)

def _logN_PLP(lambda_peak, alpha, delta_m, ml, mh, mu_g, sigma_g, res=200):
    # log integral of PLP p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    if lambda_peak==0:
        mm = np.linspace(ml, mh, res) 
    else:
        mmax = max(mh, mu_g+10*sigma_g)
        mmid = ml + delta_m + delta_m/10.

        mm   = np.concatenate([np.linspace(1., mmid, 200),                                    # lower edge                                                    
                               np.linspace(mmid+1e-1, mu_g-5*sigma_g, int(res/2)),            # before gaussian peak
                               np.linspace(mu_g-5*sigma_g+1e-1, mu_g+5*sigma_g, int(res/2)),  # gaussian peak
                               np.linspace(mu_g+5*sigma_g+1e-1, mmax+mmax/2, int(res/2)) ])   # after gaussian peak
        
        mm   = np.sort(mm) # sort to avoid interpolation errors (e.g. when mu-5sigma < ml)

        p1   = np.exp(_logpdfm1_PLP(mm, lambda_peak, alpha, delta_m, ml, mh, mu_g, sigma_g))
        
    return np.log(np.trapz(p1,mm))


def logpdf_PLP(m1, m2, lambda_m):
    """Power-law+Peak mass distribution, p(m1,m2|lambda_m) normalized

    Args:
        m1 (np.ndarray): primary mass
        m2 (np.ndarray): secondary mass
        lambda_m (dict): parameters of the mass function with keys: 
                         ["lambda_peak", "alpha", "beta", "delta_m", "ml", "mh", "mu_g", "sigma_g"]
    """

    # Unpack parameters
    lpar = ["lambda_peak", "alpha", "beta", "delta_m", "ml", "mh", "mu_g", "sigma_g"]
    lambda_peak, alpha, beta, delta_m, ml, mh, mu_g, sigma_g = [lambda_m[p] for p in lpar]

    return np.where((ml < m2) & (m2 < m1) & (m1 < max(mh, mu_g+10*sigma_g)),
                    
                    # compute logprob
                    _logpdfm1_PLP(m1, lambda_peak, alpha, delta_m, ml, mh, mu_g, sigma_g) + \
                    _logpdfm2_SPL(m2, beta, delta_m, ml) + _logC_SPL(m1, beta, delta_m, ml) - \
                    _logN_PLP(lambda_peak, alpha, delta_m, ml, mh, mu_g, sigma_g) , 

                    # return zero probability
                    -np.inf)


def pdf_PLP(m1, m2, lambda_m):
    return np.exp(logpdf_PLP(m1, m2, lambda_m))


######################################################
###################################################### PL + 2 Peaks
######################################################

def _logpdfm1_PL2P(m1, lp, l1, alpha, dm, ml, mh, mu1, s1, mu2, s2):
    # Marginal distribution p(m1), not normalised
    P  = np.exp(_logpdf_TPL(m1, alpha, ml, mh))
    G1 = np.exp(_logpdf_G(m1, mu1, s1))
    G2 = np.exp(_logpdf_G(m1, mu2, s2))

    return np.where((m1 >= ml) & (m1 <= max(mh, mu2+10*s2)), 
                    np.log((1.-lp)*P + lp*l1*G1 + lp*(1.-l1)*G2) + _logSmoothing(m1, dm, ml), 
                    -np.inf)

def _logN_PL2P(lp, l1, alpha, dm, ml, mh, mu1, s1, mu2, s2, res=200):
    # log integral of PL2P p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    if lp==0 and l1==0:
        mm = np.linspace(ml, mh, res) 
    else:
        mmax = max(mh, mu2+10*s2)
        mmid = ml + dm + dm/10.

        mm   = np.concatenate([np.linspace(1., mmid, 200),                              # lower edge                                                    
                               np.linspace(mmid+1e-1, mu1-5*s1, int(res/2)),            # before 1st gaussian peak
                               np.linspace(mu1-5*s1+1e-1, mu1+5*s1, int(res/2)),        # 1st gaussian peak
                               np.linspace(mu1+5*s1+1e-1, mu2-5*s2, int(res/2)),        # after 1st gaussian peak
                               np.linspace(mu2-5*s2+1e-1, mu2+5*s2, int(res/2)),        # 2nd gaussian peak
                               np.linspace(mu2+5*s2+1e-1, mmax+mmax/2, int(res/2)) ])   # after 1st gaussian peak
    
        mm   = np.sort(mm) # sort to avoid interpolation errors (e.g. when mu1-5s1 < ml)

        p1   = np.exp(_logpdfm1_PL2P(mm, lp, l1, alpha, dm, ml, mh, mu1, s1, mu2, s2))
        
    return np.log(np.trapz(p1,mm))


def logpdf_PL2P(m1, m2, lambda_m):
    """Power-law+Peak mass distribution, p(m1,m2|lambda_m) normalized

    Args:
        m1 (np.ndarray): primary mass
        m2 (np.ndarray): secondary mass
        lambda_m (dict): parameters of the mass function with keys: 
                         ["lambda_peak", "lambda1", "alpha", "beta", "delta_m", "ml", "mh", "mu1", "s1", "mu2", "s2"]
    """

    # Unpack parameters
    lpar = ["lambda_peak", "lambda1", "alpha", "beta", "delta_m", "ml", "mh", "mu1", "s1",  "mu2", "s2"]
    lp, l1, alpha, beta, dm, ml, mh, mu1, s1, mu2, s2 = [lambda_m[p] for p in lpar]

    return np.where((ml < m2) & (m2 < m1) & (m1 < max(mh, mu2+10*s2)),
                    
                    # compute logprob
                    _logpdfm1_PL2P(m1, lp, l1, alpha, dm, ml, mh, mu1, s1, mu2, s2) + \
                    _logpdfm2_SPL(m2, beta, dm, ml) + _logC_SPL(m1, beta, dm, ml) - \
                    _logN_PL2P(lp, l1, alpha, dm, ml, mh, mu1, s1, mu2, s2) , 

                    # return zero probability
                    -np.inf)


def pdf_PL2P(m1, m2, lambda_m):
    return np.exp(logpdf_PL2P(m1, m2, lambda_m))


######################################################
###################################################### FLAT (for BNSs)
######################################################


def logpdf_FLAT(m1, m2, lambda_m):
    
    '''p(m1, m2 | Lambda ) = p(m1)*p(m2), normalized to one'''
    
    ml, mh  = lambda_m["ml"], lambda_m["mh"]

    return np.where( (ml<m2) & (m2<m1) & (m1<mh), -np.log(mh-ml), -np.inf )


def pdf_FLAT(m1, m2, lambda_m):
    return np.exp(logpdf_FLAT(m1, m2, lambda_m))


######################################################
###################################################### G(m1)G(m2) for BNSs
######################################################

def pdf_truncGausslower(x, xmin, loc=0., scale=1.):

    phi = 0.5*(1.+erf((xmin-loc)/(np.sqrt(2.)*scale)))

    return np.where(x>xmin, 1./(np.sqrt(2.*np.pi)*scale)/(1.-phi) * np.exp(-(x-loc)**2/(2*scale**2)) ,0.)



def logpdf_Gm1Gm2(m1,m2, lambda_m):
    
    '''p(m1, m2 | Lambda ) = p(m1)*p(m2), normalized to one'''
    
    mu_m, sigma_m = lambda_m["mu_m"], lambda_m["sigma_m"]
    
    logpdfm1 = np.log(pdf_truncGausslower(m1, 0., loc=mu_m, scale=sigma_m))
    logpdfm2 = np.log(pdf_truncGausslower(m2, 0., loc=mu_m, scale=sigma_m))
    
    return np.where(m2<m1, logpdfm1+logpdfm2, -np.inf)



######################################################
###################################################### PL + Peak NSBH
######################################################


def logpdf_PLP_NSBH(m1, m2, lambda_m):
    """Power-law+Peak mass distribution, p(m1,m2|lambda_m) normalized

    Args:
        m1 (np.ndarray): primary mass
        m2 (np.ndarray): secondary mass
        lambda_m (dict): parameters of the mass function with keys: 
                         ["lambda_peak", "alpha", "beta", "delta_m", "ml1", "mh1", "ml2", "mh2", "mu_g", "sigma_g"]
    """

    # Unpack parameters
    lpar = ["lambda_peak", "alpha", "beta", "delta_m", "ml1", "mh1", "ml2", "mh2", "mu_g", "sigma_g"]
    lambda_peak, alpha, beta, delta_m, ml1, mh1, ml2, mh2, mu_g, sigma_g = [lambda_m[p] for p in lpar]

    return np.where((ml2 < m2) & (m2 < mh2) & (m2 < m1) & (ml1 < m1) & (m1 < max(mh1, mu_g+10*sigma_g)),
                    
                    # compute logprob
                    _logpdfm1_PLP(m1, lambda_peak, alpha, delta_m, ml1, mh1, mu_g, sigma_g) + \
                    _logpdfm2_SPL(m2, beta, delta_m, ml2) + _logC_SPL(m1, beta, delta_m, ml2) - \
                    _logN_PLP(lambda_peak, alpha, delta_m, ml1, mh1, mu_g, sigma_g) , 

                    # return zero probability
                    -np.inf)


def pdf_PLP_NSBH(m1, m2, lambda_m):
    # print(np.exp(logpdf_PLP_NSBH(m1, m2, lambda_m)))
    # print()
    return np.exp(logpdf_PLP_NSBH(m1, m2, lambda_m))