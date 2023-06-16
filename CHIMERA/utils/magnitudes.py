
###################
# Magnitudes
###################

def Mag2lum(M, band='K'):
    """Converts magnitudes to solar luminosities

    Args:
        M (float): magnitude
        band (str, optional): obs. magnitude. K corr not implemented. Defaults to 'K'.

    Returns:
        float: luminositu in units of solar luminosity
    """
    
    if band == 'bol':
        M_sun = 4.83
    elif band == 'B':
        M_sun = 4.72
    elif band == 'W1':
        M_sun = 3.24
    elif band == 'K':
        M_sun = 3.28
    else:
        ValueError("Not supported")

    return np.power(10, 0.4*(M_sun-M))


def lum2Mag(L, band='K'):
    """Converts magnitudes to solar luminosities

    Args:
        M (float): magnitude
        band (str, optional): obs. magnitude. K corr not implemented. Defaults to 'K'.

    Returns:
        float: luminositu in units of solar luminosity
    """
    
    if band == 'bol':
        M_sun = 4.83
    elif band == 'W1':
        M_sun = 3.24
    else:
        ValueError("Not supported")

    return -2,5*np.log10(L) + M_sun





######################
# Schechter functions
######################


import numpy as np
from scipy.integrate import quad

def Lstar_default(band):
    if band=="B":
        Lstar = 2.45e10
    elif band=="K":
        Lstar = 1.1e11 
    return Lstar


def lambda_default(band):
    if band=="B":
        # GWCOSMO
        return {"alpha":-1.21, "M_star":-19.70, "phi_star":5.5e-3} # "Mmin":-22.96, "Mmax":-12.96
    elif band=="K":
        # GWCOSMO
        # https://iopscience.iop.org/article/10.1086/322488/pdf
        # Mmax from  Fig 3 of the above reference 
        return {"alpha":-1.09, "M_star":-23.39, "phi_star":5.5e-3}  # , "Mmin":-27.0, "Mmax":-19.0
    elif band=="B_Gehrels_Arcavi_17":
        # used in Fischbach 2019
        return {"alpha":-1.07, "M_star":-20.47, "phi_star":5.5e-3} 
    elif band=="K_GLADE_2.4":
        # used in Fischbach 2019
        return {"alpha":-1.02, "M_star":-23.55, "phi_star":5.5e-3} 

    else:
        raise ValueError("{band} band not implemented")




# phistar does not affect the analysis, (cancels in numerator and denominator of the main analysis, as for H0)


def log_phiM(M, args_Sch, args_cosmo):
    """Compute Schechter function (in log)

    Args:
        M (np.array, float): detector frame absolute magnitudes
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        _type_: _description_
    """    
    alpha, M_star, phi_star = args_Sch.values()
    h                       = args_cosmo["H0"]/100.

    M_star   = M_star + 5.*np.log10(h)
    phi_star = phi_star * h**3
    factor   = np.power(10., 0.4*(M-M_star))

    return 0.4 * np.log(10.0) * phi_star * factor**(1.+alpha) * np.exp(-factor)
    


def log_phiM_normalized(M, Mmin, Mmax, args_Sch, args_cosmo):
    """Compute Schechter function normalized in [Mmin, Mmax]

    Args:
        M (np.array, float): detector frame absolute magnitudes
        Mmin (float): lower limit of integration
        Mmax (float): upper limit of integration
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        np.array: Normalized Schechter function 
    """
    h     = args_cosmo["H0"]/100.
    Mmin  = Mmin + 5.*np.log10(h)
    Mmax  = Mmax + 5.*np.log10(h)

    return phiM(M,args_Sch,args_cosmo)/quad(phiM, Mmin, Mmax, args=(args_Sch, args_cosmo))[0]


def get_SchNorm(phistar, Lstar, alpha, Lcut):
        '''
        
        Input:  - Schechter function parameters L_*, phi_*, alpha
                - Lilit of integration L_cut in units of 10^10 solar lum.
        
        Output: integrated Schechter function up to L_cut in units of 10^10 solar lum.
        '''
        from scipy.special import gammaincc
        from scipy.special import gamma
                
        norm= phistar*Lstar*gamma(alpha+2)*gammaincc(alpha+2, Lcut)
        return norm