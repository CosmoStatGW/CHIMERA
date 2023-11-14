import numpy as onp
from scipy.special import hyp2f1

import jax.numpy as np
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)
import jax.experimental.host_callback as hcb

c_light = 299792.458 # km/s

@jit
def E_inv_norel(z, Om0, Ode0):
    return np.power((1. + z)**3 * Om0 + Ode0, -0.5)

def _hyp2f1_lcdm(x):
    return hyp2f1(1./6, 1./2, 7./6, -x**3)

@jit
def hyp2f1_lcdm(x):
  return hcb.call(_hyp2f1_lcdm, x, result_shape=x)

@jit
def _T_hypergeometric(x):
    r"""Compute value using Gauss Hypergeometric function 2F1. Taken from astropy.
    T(x) = 2 \sqrt(x) _{2}F_{1}\left(\frac{1}{6}, \frac{1}{2}; \frac{7}{6}; -x^3 \right)

    Ref. [1] Baes, Camps, & Van De Putte (2017). MNRAS, 468(1), 927-930.
    """
    return 2 * np.sqrt(x) * hyp2f1_lcdm(x)

@jit
def _int_dC_hyperbolic(z, Om0):
    s = ((1 - Om0) / Om0) ** (1./3)
    prefactor = (s * Om0)**(-0.5)
    return prefactor * (_T_hypergeometric(s) - _T_hypergeometric(s / (z + 1.0)))


##########################
####### Distances ########
##########################


def dC(z, args):
    """Comoving distance [Gpc] at redshift ``z``."""
    return c_light/(1e3*args["H0"]) * _int_dC_hyperbolic(z, args["Om0"])


def dL(z, args):
    """Luminosity distance [Gpc] at redshift ``z``."""
    return (z+1.0)*dC(z, args)


def dA(z, args):
    """Angular diameter distance [Gpc] at redshift ``z``."""
    return dC(z, args)/(z+1.0)

def ddL_dz(z, args, dL=None):
    """Differential luminosity distance [Gpc] at redshift ``z``."""

    if dL is not None:
        # raise Exception("Not implemented")
        return dL/(1+z) + c_light/(1e3*args["H0"]) * (1+z) * E_inv_norel(z, args["Om0"], 1.-args["Om0"])

    return dC(z, args) + c_light/(1e3*args["H0"]) * (1+z) * E_inv_norel(z, args["Om0"], 1.-args["Om0"])

def log_ddL_dz(z, args, dL=None):
    """log of the differential luminosity distance [Gpc] at redshift ``z``."""
    return np.log(ddL_dz(z, args, dL=dL))



##########################
#######  Volumes  ########
##########################

def V(z, args):
    """Comoving volume in [Gpc^3] at redshift ``z``."""
    return 4.0 / 3.0 * np.pi * dC(z, args)**3

def dV_dz(z, args):
    """Differential comoving volume at redshift ``z``."""
    return 4*np.pi*c_light/(1e3*args["H0"]) * (dC(z, args) ** 2) * E_inv_norel(z, args["Om0"], 1.-args["Om0"])

def log_dV_dz(z, args):
    return np.log(dV_dz(z, args))


#########################
### Solver for z(dL)  ###
#########################

def z_from_dL(dL_vec, args):
    z_grid  = np.concatenate([np.logspace(-15, np.log10(9.99e-09), base=10, num=100), 
                              np.logspace(-8, np.log10(7.99), base=10, num=8000),
                              np.logspace(np.log10(8), 5, base=10, num=5000)])
    dL_grid = dL(z_grid, args)

    # return cubic_spline_interpolation(dL_grid, z_grid, dL_vec)
    return  np.interp(dL_vec, dL_grid, z_grid)


def H0_approx(dL, z):
    return c_light/(1e3*dL) * z




# @jit
# def cubic_spline_interpolation(x_new, x, y):
#     dx = np.diff(x)
#     dy = np.diff(y)

#     # Spline matrix
#     A = np.diag(np.concatenate((np.array([dx[1]]), 2 * (dx[:-1] + dx[1:]), np.array([dx[-2]])))) + \
#         np.diag(np.concatenate((np.array([-dx[0] - dx[1]]), dx[1:])), k=1) + \
#         np.diag(np.concatenate((np.array([dx[0]]), np.zeros(len(x) - 3))), k=2) + \
#         np.diag(np.concatenate((dx[:-1], np.array([-dx[-2] - dx[-1]]))), k=-1) + \
#         np.diag(np.concatenate((np.zeros(len(x) - 3), np.array([dx[-1]]))), k=-2)

#     # Spline coefficients
#     coeff = np.linalg.solve(A, np.pad(3 * (dy[1:]/dx[1:]-dy[:-1]/dx[:-1]), (1, 1), mode='constant'))

#     # Compute polynomials
#     ind = np.clip(np.digitize(x_new, x)-1, 0, len(x) - 2)
#     t   = x_new - x[ind]
#     dx  = dx[ind]

#     result = y[ind] + ((y[ind + 1] - y[ind]) / dx - (2 * coeff[ind] + coeff[ind + 1]) * dx / 3.0) * t + \
#              coeff[ind] * t ** 2 + ((coeff[ind + 1] - coeff[ind]) / (3 * dx)) * t ** 3

#     return result



