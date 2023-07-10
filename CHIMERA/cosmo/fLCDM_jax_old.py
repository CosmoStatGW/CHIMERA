import jax.numpy as jnp
# from jax.scipy.special import hyp2f1
# from jax.scipy.interpolate import interp1d
from jax import jit, vmap
import numpy as np  # only for data generation, not for computation

from scipy.special import hyp2f1
import jax.numpy as jnp

hyp2f1_vectorized = jnp.vectorize(hyp2f1)



# No direct JAX equivalent for these astropy functions. I'm using simple function equivalents here,
# but for the actual scientific calculation, you might need to implement or find JAX compatible versions
def flcdm_inv_efunc_norel(z, Om0, Ode0):
    Omz = Om0 * (1. + z)**3
    return 1./jnp.sqrt(Omz + Ode0)

E_inv = jit(vmap(flcdm_inv_efunc_norel, in_axes=(0, None, None)))  # vectorize and jit for efficiency

c_light = 299792.458  # km/s

def _T_hypergeometric(x):
    return 2 * jnp.sqrt(x) * hyp2f1(1./6, 1./2, 7./6, -x**3)

def _int_dC_hyperbolic(z, Om0):
    s = ((1 - Om0) / Om0) ** (1./3)
    prefactor = (s * Om0)**(-0.5)
    return prefactor * (_T_hypergeometric(s) - _T_hypergeometric(s / (z + 1.0)))

# Distances

def dC(z, args):
    return c_light/args["H0"] * _int_dC_hyperbolic(z, args["Om0"])

def dL(z, args):
    return (z+1.0)*dC(z, args)

def dA(z, args):
    return dC(z, args)/(z+1.0)

def ddL_dz(z, args, dL=None):
    if dL is not None:
        return dL/(1+z) + c_light/args["H0"] * (1+z) * E_inv(z, args["Om0"], 1.-args["Om0"])

    return dC(z, args) + c_light/args["H0"] * (1+z) * E_inv(z, args["Om0"], 1.-args["Om0"])

def log_ddL_dz(z, args, dL=None):
    return jnp.log(ddL_dz(z, args, dL=dL))

# Volumes

def V(z, args):
    return 4.0 / 3.0 * np.pi * dC(z, args)**3

def dV_dz(z, args):
    return 4*np.pi*c_light/args["H0"] * (dC(z, args) ** 2) * E_inv(z, args["Om0"], 1.-args["Om0"])

def log_dV_dz(z, args):
    return jnp.log(dV_dz(z, args))

# Solver for z(dL)

# JAX doesn't support all scipy functions, so here we generate data with numpy (CPU)
# and use those for the jax interpolation.
def z_from_dL(dL_vec, args):
    z_grid  = np.concatenate([np.logspace(-15, np.log10(9.99e-09), base=10, num=10),
                              np.logspace(-8,  np.log10(7.99), base=10, num=1000),
                              np.logspace(np.log10(8), 5, base=10, num=100)])
    # Generate data with numpy on CPU
    y = [dL(z, args) for z in z_grid]
    f_intp = interp1d(y, z_grid, kind='linear', fill_value=(0,jnp.nan), assume_sorted=True)
    return f_intp(dL_vec)
