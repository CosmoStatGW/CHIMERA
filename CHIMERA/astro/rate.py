import numpy as np

######################################################
###################################################### PL
######################################################

def phi_PL(z,lambda_rate):
    """PowerLaw CBCs rate density in [Gpc-3 yr-1].
    If gamma=0, merger rate density is uniform in comoving volume and source-frame time"""
    return (1. + z)**lambda_rate["gamma"]


######################################################
###################################################### Callister+20/Madau+14
######################################################


def phi_MD(z, lambda_rate):
    """Madau-Dickinson-like star formation rate density in [Gpc-3 yr-1]

    Args:
        z (np.ndarray): redshift
        lambda_rate (dict): parameters of the rate function with keys: ["gamma", "kappa", "zp"]

    Returns:
        np.ndarray: SFRD (not normalized!)
    """
    gamma, kappa, zp = lambda_rate["gamma"], lambda_rate["kappa"], lambda_rate["zp"]

    return (z+1.)**gamma / ( 1. + ((z+1.)/(zp+1.))**(gamma+kappa) )


def phi_MD_norm(z, lambda_rate):
    """Normalized Madau-Dickinson-like star formation rate density, R(z=0)=R0

    Args:
        z (np.ndarray): redshift
        lambda_rate (dict): parameters of the rate function with keys: ["R0", "gamma", "kappa", "zp"]

    Returns:
        np.ndarray: SFRD
    """
    R0, gamma, kappa, zp = lambda_rate["R0"], lambda_rate["gamma"], lambda_rate["kappa"], lambda_rate["zp"]

    return (1. + (zp + 1.)**(-gamma-kappa)) * R0 * phi_MD(z, lambda_rate)


def logphi_MD(z, lambda_rate):  # slower than non-log
    """Logarithm of Madau-Dickinson-like star formation rate density in [Mpc-3 yr-1]

    Args:
        z (np.ndarray): redshift
        lambda_rate (dict): parameters of the rate function with keys: ["gamma", "kappa", "zp"]

    Returns:
        np.ndarray: log(SFRD) (not normalized!)
    """
    gamma, kappa, zp = lambda_rate["gamma"], lambda_rate["kappa"], lambda_rate["zp"]

    return gamma*np.log1p(z) - np.log1p( ((z+1.)/(zp+1.))**(gamma+kappa) )




