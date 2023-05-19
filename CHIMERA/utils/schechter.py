import numpy as np
from scipy.integrate import quad



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




# phistar does not affect the analysis, (cancels in numerator and denominator of the main analysis, and for H0)





def phiM(M, args_Sch, args_cosmo):
    """Compute Schechter function (in log)

    Args:
        M (np.array, float): detector frame absolute magnitudes
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        _type_: _description_
    """    
    alpha, M_star, phi_star = args_Sch.values()
    h07                     = args_cosmo["H0"]/70.

    M_star   = M_star + 5.*np.log10(h07)
    phi_star = phi_star * h07**3
    factor   = 10**(0.4*(M-M_star))

    return 0.4 * np.log(10.0) * phi_star * factor**(1.+alpha) * np.exp(-factor)
    



def phi_norm(M, Mmin, Mmax, args_Sch, args_cosmo):
    """Compute normalized Schechter function

    Args:
        M (np.array, float): detector frame absolute magnitudes
        Mmin (float): lower limit of integration
        Mmax (float): upper limit of integration
        args_Sch (dict): Schechter function parameters
        args_cosmo (dict): cosmology parameters

    Returns:
        np.array: Normalized Schechter function 
    """
    return phi(M,args_Sch,args_cosmo)/quad(phi, Mmin, Mmax, args=(args_Sch, args_cosmo))[0]



