import numpy as np 
from scipy.interpolate import interp1d


from astropy.cosmology import scalar_inv_efuncs
E_inv = np.vectorize(scalar_inv_efuncs.flcdm_inv_efunc_norel, otypes=[np.float64])

c_light = 299792.458 # km/s



from .fLCDM import dC


##########################
####### Distances ########
##########################


def dLEM(z, args):
    """Luminosity distance [Mpc] at redshift ``z``."""
    return (z+1.0)*dC(z, args)


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
## MG Distances (Xi,n) ##
#########################

def Xi(z, args):
    """Modified gravity parameter Xi at redshift ``z`` in the (Xi, n) parametrization. 
    Ref. [1] Belgacem, Dirian, Foffa, & Maggiore, Phys. Rev. D98 (2018) 023510
    """
    return args["Xi0"] + (1-args["Xi0"]) * (1+z)**(-args["n"])

def sPrime(z, args):
    return Xi(z, args)-args["n"]*(1-args["Xi0"])/(1+z)**args["n"]

def dL(z, args):
    """Luminosity distance [Mpc] at redshift ``z`` in (Xi, n) modified gravity.
    """
    return dL(z, args) if ((args["Xi0"]==1) | (args["n"]==0)) else dL(z, args)*Xi(z, args)


# def ddL_dz_MG(z):
#     """Differential luminosity distance [Mpc] at redshift ``z`` in (Xi, n) modified gravity."""
#     return self.dH*( self.sPrime(z)*self.dC_scalar(z) + (1+z)*self.Xi(z)*self.E_inv_scalar(z) ) 

# def log_ddL_dz_MG(z):
#     """log differential luminosity distance [Mpc] at redshift ``z`` in (Xi, n) modified gravity."""
#     return np.log(self.ddL_dz_MG(z))




#########################
### Solver for z(dL)  ###
#########################

def z_from_dLEM(dL_vec, args):
    z_grid  = np.concatenate([np.logspace(-15, np.log10(9.99e-09), base=10, num=10), 
                              np.logspace(-8,  np.log10(7.99), base=10, num=1000),
                              np.logspace(np.log10(8), 5, base=10, num=100)])
    f_intp  = interp1d(dL(z_grid, args), z_grid, 
                       kind='cubic', bounds_error=False, 
                       fill_value=(0,np.NaN), assume_sorted=True)
    return f_intp(dL_vec) 

def z_from_dLGW(self, dL):
    z_grid  = np.concatenate([np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), 
                                np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000),
                                np.logspace(start=np.log10(8), stop=5, base=10, num=100)])
    f_intp  = interp1d(self.dL(z_grid)*self.Xi(z_grid), z_grid, 
                        kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
    return f_intp(dL) 


















# def vectorize_redshift_method(func=None, nin=1):
#     """Taken from astropy. Vectorize a method of redshift(s).
#     Parameters
#     ----------
#     func : callable or None
#         method to wrap. If `None` returns a :func:`functools.partial`
#         with ``nin`` loaded.
#     nin : int
#         Number of positional redshift arguments.
#     Returns
#     -------
#     wrapper : callable
#         :func:`functools.wraps` of ``func`` where the first ``nin``
#         arguments are converted from |Quantity| to :class:`numpy.ndarray`.
#     """
#     # allow for pie-syntax & setting nin
#     if func is None:
#         return functools.partial(vectorize_redshift_method, nin=nin)

#     @functools.wraps(func)
#     def wrapper(self, *args, **kwargs):
#         """
#         :func:`functools.wraps` of ``func`` where the first ``nin``
#         arguments are converted from |Quantity| to `numpy.ndarray` or scalar.
#         """
#         return wrapper.__vectorized__(self, *args[:nin], *args[nin:], **kwargs)

#     wrapper.__vectorized__ = np.vectorize(func)

#     return wrapper

    
# class Cosmology():
#     def __init__(self):
#         pass 

# class fwCDM(Cosmology):
#     def __init__(self, H0=70, Om0=0.3, w0=-1, wa=0, Xi0=1, n=0, hyperb=True, dist_unit='Mpc', v=False):
#         """
#         Class to compute fLCDM, fw0CDM, fw0waCDM cosmologies. Following astropy's implementation,
#         the integration for fLCDM is optimized with an hypergeometric transformation which 
#         considerably reduces the computation time. For w0!=-1, wa!=0 a more efficient evaluation
#         is obtained using the C language. 

#         Args:
#             H0 (float, optional): Hubble constant in Mpc. Defaults to 70.
#             Om0 (float, optional): Omega matter at z=0. Defaults to 0.3.
#             w0 (float, optional): Dark energy equation of state at z=0. Defaults to -1.
#             wa (float, optional): Negative derivative of the dark energy equation of state with respect to the scale factor. Defaults to 0.
#             Xi0 (float, optional): Modified GW propagation parameter. Defaults to 1 (dL_GW=dL_EM).
#             n (float, optional): Modified GW propagation parameter. Defaults to 0.
#             hyperb (float, optional): use the hypergeometric transformation for fLCDM cosmology. Defaults to True.
#             dist_unit (str): 'Mpc' or 'Gpc'.
#         """
#         self.H0        = H0 if dist_unit=='Mpc' else H0*1e3
#         self.Om0       = Om0
#         self.w0        = w0
#         self.wa        = wa
#         self.Xi0       = Xi0
#         self.n         = n
#         Tcmb           = 0 # 2.7255 # 0
#         Og0            = 4.48131e-7 * (Tcmb)**4 / (H0/100)**2
#         self.Or0       = Og0  +  Og0 * (3.04) * (7/8) * (4/11)**(4/3)
#         self.Ode0      = 1 - self.Om0 - self.Or0
#         self.dH        = C_LIGHT/self.H0
#         self.par_names = ['H0', 'Om0', 'w0', 'wa', 'Xi0', 'n']
#         self.par_vals  = [self.H0, self.Om0, self.w0, self.wa, self.Xi0, self.n]
        

#         if self.w0==-1:
#             # if v: print(" using fLCDM")
#             self.E_inv      = scalar_inv_efuncs.flcdm_inv_efunc_norel
#             self.E_inv_args = (self.Om0, self.Ode0)
#             self.integral_dC = self._int_dC_z1z2_hyperbolic if hyperb else self._int_dC_z1z2_quad

#         elif self.wa==0:
#             # if v: print(" using fwLCDM")
#             self.E_inv      = scalar_inv_efuncs.wcdm_inv_efunc_norel
#             self.E_inv_args = (self.Om0, self.Ode0, 0., self.w0)
#             self.integral_dC = self._int_dC_z1z2_quad
#         else:
#             # if v: print(" using fw0waLCDM")
#             self.E_inv      = scalar_inv_efuncs.w0wacdm_inv_efunc_norel
#             self.E_inv_args = (self.Om0, self.Ode0, 0., self.w0, self.wa)
#             self.integral_dC = self._int_dC_z1z2_quad        

#         # Backward compatibility
#         self.baseValues = {'H0': 70, 'Om' : 0.3 , 'w0': -1, 'Xi0': 1, 'n': 0}
#         self.names = {    'H0':r'$H_0$', 
#                           'Om':r'$\Omega_{\rm {m,}0 }$',
#                           'w0':r'$w_{0}$',
#                            'Xi0':r'$\Xi_0$', 
#                            'n':r'$n$', }
#         self.params = ['H0', 'Om', 'w0', 'Xi0', 'n']
#         self.n_params = len(self.params)

#     def __repr__(self):
#         return "".join(["{:s}\t{:.2f}\n".format(n, v) for n,v in zip(self.par_names, self.par_vals)])

#     ###############################################
#     ############################################### VECTORIZE
#     ###############################################

#     @vectorize_redshift_method(nin=2)
#     def _int_dC_z1z2_quad(self, z1, z2):
#         return quad(self.E_inv, z1, z2,  args=self.E_inv_args)[0]


#     @vectorize_redshift_method(nin=1)
#     def _E_inv_eval(self, z):
#         return self.E_inv(z, *self.E_inv_args)

#     ###############################################
#     ############################################### COSMOGRAPHY EQUATIONS
#     ###############################################



#     def E_inv_scalar(self, z):
#         """"Scalar 1/E(z) at redshift ``z``.."""
#         if hasattr(self, '_E_inv_scalar'): 
#             if np.array_equal(z, self._z_previous_Einv):
#                 return self._E_inv_scalar            
#         self._E_inv_scalar = self._E_inv_eval(z)
#         self._z_previous_Einv = z
#         return self._E_inv_scalar  















#     ###############################################
#     ############################################### SOLVER FOR z(dL)
#     ###############################################

#     def z_from_dL(self, dL):
#         z_grid  = np.concatenate([np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), 
#                                   np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000),
#                                   np.logspace(start=np.log10(8), stop=5, base=10, num=100)])
#         f_intp  = interp1d(self.dL(z_grid), z_grid, 
#                            kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
#         return f_intp(dL) 

#     def z_from_dLGW(self, dL):
#         z_grid  = np.concatenate([np.logspace(start=-15, stop=np.log10(9.99e-09), base=10, num=10), 
#                                   np.logspace(start=-8, stop=np.log10(7.99), base=10, num=1000),
#                                   np.logspace(start=np.log10(8), stop=5, base=10, num=100)])
#         f_intp  = interp1d(self.dL(z_grid)*self.Xi(z_grid), z_grid, 
#                            kind='cubic', bounds_error=False, fill_value=(0,np.NaN), assume_sorted=True)
#         return f_intp(dL) 


#     ###############################################
#     ############################################### UNUSED
#     ###############################################

#     def _E_inv_w0wa(self,z):
#         zp1    = z+1.0
#         exp_de = np.exp(-3.*self.wa*z/zp1) 
#         return (zp1**3 * (self.Om0 + self.Or0*zp1 + self.Ode0*(1.+self.w0+self.wa)*exp_de ) )**(-0.5)
#     def _E_inv_w(self,z):
#         zp1    = z+1.0
#         return (zp1**3 * (self.Om0 + self.Or0*zp1 + self.Ode0*(1.+self.w0) ) )**(-0.5)
#     def _E_inv_lcdm(self,z):
#         zp1    = z+1.0
#         return (zp1**3 * (self.Om0 + self.Or0*zp1) + self.Ode0)**(-0.5)

#     @property
#     def dC_optim(self, z):
#       if not hasattr(self, '_dC_optim'): self._dC_optim = self.dC()
#       return self._dC_optim

#     @property
#     def dL_optim(self, z):
#       if not hasattr(self, '_dL_optim'): self._dL_optim = self.dL()
#       return self._dL_optim
