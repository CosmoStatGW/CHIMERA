import numpy as np 
# import functools
from scipy.interpolate import interp1d
from scipy.special import hyp2f1

from astropy.cosmology import scalar_inv_efuncs
E_inv = np.vectorize(scalar_inv_efuncs.flcdm_inv_efunc_norel, otypes=[np.float64])

c_light = 299792.458 # km/s


def _T_hypergeometric(x):
    r"""TAKEN FROM ASTROPY 
    Compute value using Gauss Hypergeometric function 2F1.
    T(x) = 2 \sqrt(x) _{2}F_{1}\left(\frac{1}{6}, \frac{1}{2}; \frac{7}{6}; -x^3 \right)

    Ref. [1] Baes, Camps, & Van De Putte (2017). MNRAS, 468(1), 927-930.
    """
    return 2 * np.sqrt(x) * hyp2f1(1./6, 1./2, 7./6, -x**3)

def _int_dC_hyperbolic(z, Om0):
    s = ((1 - Om0) / Om0) ** (1./3)
    prefactor = (s * Om0)**(-0.5)
    return prefactor * (_T_hypergeometric(s) - _T_hypergeometric(s / (z + 1.0)))


##########################
####### Distances ########
##########################

def dC(z, args):
    """Comoving distance [Mpc] at redshift ``z``."""
    return c_light/args["H0"] * _int_dC_hyperbolic(z, args["Om0"])

def dL(z, args):
    """Luminosity distance [Mpc] at redshift ``z``."""
    return (z+1.0)*dC(z, args)

def dA(z, args):
    """Angular diameter distance [Mpc] at redshift ``z``."""
    return dC(z, args)/(z+1.0)

def ddL_dz(z, args, dL=None):
    """Differential luminosity distance [Mpc] at redshift ``z``."""

    if dL is not None:
        # raise Exception("Not implemented")
        return dL/(1+z) + c_light/args["H0"] * (1+z) * E_inv(z, args["Om0"], 1.-args["Om0"])

    return dC(z, args) + c_light/args["H0"] * (1+z) * E_inv(z, args["Om0"], 1.-args["Om0"])

def log_ddL_dz(z, args, dL=None):
    """log of the differential luminosity distance [Mpc] at redshift ``z``."""
    return np.log(ddL_dz(z, args, dL=dL))


##########################
#######  Volumes  ########
##########################

def V(z, args):
    """Comoving volume in [Mpc^3] at redshift ``z``."""
    return 4.0 / 3.0 * np.pi * dC(z, args)**3

def dV_dz(z, args):
    """Differential comoving volume at redshift ``z``."""
    return 4*np.pi*c_light/args["H0"] * (dC(z, args) ** 2) * E_inv(z, args["Om0"], 1.-args["Om0"])


#########################
### Solver for z(dL)  ###
#########################

def z_from_dL(dL_vec, args):
    z_grid  = np.concatenate([np.logspace(-15, np.log10(9.99e-09), base=10, num=10), 
                              np.logspace(-8,  np.log10(7.99), base=10, num=1000),
                              np.logspace(np.log10(8), 5, base=10, num=100)])
    f_intp  = interp1d(dL(z_grid, args), z_grid, 
                       kind='cubic', bounds_error=False, 
                       fill_value=(0,np.NaN), assume_sorted=True)
    return f_intp(dL_vec) 
