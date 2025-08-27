
###################
# Magnitudes
###################

import logging
logs = logging.getLogger(__name__)
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc, gamma

from .config import jnp, jax
from .math import trapz
from functools import partial


SchecheterParsCOSMOS2020 = {'phi1': np.array([0.0089015,  -0.00526512,  0.00081921]),
                            'phi2': np.array([0.00261159, -0.00149962,  0.0002548 ]),
                            'alpha1': np.array([-0.3, 0, 0]),
                            'alpha2': np.array([-1.5, 0, 0]),
                            'logMstar': np.array([10.60406389, 0.19696344, -0.03471661])}
SchechterMlimCOSMOS2020 = lambda z: -3.23e7*(1+z) + 7.83e7*(1+z)**2

SchecheterParsMICE_B25 = {"alpha1":   [-0.3,                 0.,  0.        ],
                          "alpha2":   [-1.5,                 0.,  0.        ],
                          "logMstar": [10.5,                 0.,  0.        ],
                          "phi1":     [ 0.00147607,  0.00344156, -0.00168267],
                          "phi2":     [ 0.00322851, -0.00240074,  0.00057019]}

SchechterMlimMICE_B25 = lambda z: 2.89766937 + 5.32024468*(1+z) - 1.04370681*(1+z)**2




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
        M_sun = 3.27
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

    return -2.5*np.log10(L) + M_sun





######################
# Schechter functions
######################


import numpy as np
from scipy.integrate import quad, simpson

def Lstar_default(band):
    if band=="B":
        Lstar = 2.45e10   # Gehrels et al. (2016), Arcavi et al. (2017)
    elif band=="K":
        Lstar = 1.056e11    # Gehrels et al. (2016), Arcavi et al. (2017)
    return Lstar


def lambda_default(band):
    if band=="B":
        # GWCOSMO
        return {"alpha":-1.21, "M_star":-19.70, "phi_star":5.5e-3} # "Mmin":-22.96, "Mmax":-12.96
    elif band=="K":
        # GWCOSMO
        # https://iopscience.iop.org/article/10.1086/322488/pdf
        # Mmax from  Fig 3 of the above reference
        return {"alpha":-1.09, "M_star":-23.39, "phi_star":5.5e-3, "L_star":4.96e10}  # , "Mmin":-27.0, "Mmax":-19.0
    elif band=="B_Gehrels_Arcavi_17":
        # used in Fischbach 2019
        return {"phi_star":5.5e-3, "alpha":-1.07, "L_star": 2.45e10, "M_star":-20.47,}
    elif band=="K_GLADE_2.4":
        # used in Fischbach 2019
        return {"phi_star":5.5e-3, "alpha":-1.02, "L_star": 1.1e11, "M_star":-23.55, }
    else:
        raise ValueError(f"{band} band not implemented")




# phistar does not affect the analysis, (cancels in numerator and denominator of the main analysis, as for H0)


def schechter_mags(M, alpha, M_star, phi_star, h=1.):
    """Computes Schechter LF in the absolute magnitude space.

    Args:
        M (np.array, float): absolute magnitudes
        alpha (float): faint-end slope of the Schechter LF
        M_star (float): characteristic magnitude of the Schechter LF
        phi_star (float): normalization of the Schechter LF at M_star
        h (float, optional): Small h=H0/100. Defaults to 1.

    Returns:
        np.ndarray: Schechter LF in the absolute magnitude space
    """

    M_star   = M_star + 5.*np.log10(h)
    phi_star = phi_star * h**3
    x        = np.power(10., 0.4*(M-M_star))

    return 0.4 * np.log(10.0) * phi_star * x**(1.+alpha) * np.exp(-x)


def phiSchL_normL(lambda_mag):
    """
    Calculate the normalization factor for the Schechter luminosity function.

    Parameters:
        lambda_mag (dict): Dictionary containing the Schechter function magnitude parameters.
            Required keys: "phistar", "Lstar", "alpha", "cut"

    Returns:
        float: The normalization factor for the Schechter function.

    Notes:
        The function uses the gammaincc and gamma functions from scipy.special module.
        The normalization factor is calculated using the formula:
            norm = phistar * Lstar * gamma(alpha + 2) * gammaincc(alpha + 2, Lcut)
        where gammaincc is the complemented incomplete gamma function.
    """

    phi_star, L_star, alpha, cut = [lambda_mag[i] for i in ["phi_star", "L_star", "alpha", "L_over_Lstar_cut"]]

    return phi_star * L_star * gamma(alpha + 2) * gammaincc(alpha + 2, cut)


def phiSchL(L, lambda_mag):
    phi_star, L_star, alpha = [lambda_mag[i] for i in ["phi_star", "L_star", "alpha"]]

    Lx = L/L_star

    return (phi_star/L_star) * (Lx)**alpha * np.exp(-Lx)


def phiSchL_normN(lambda_mag):
    """
    Calculate the normalization factor for the Schechter function based on magnitude parameters.

    Parameters:
        lambda_mag (dict): Dictionary containing the Schechter function magnitude parameters.
            Required keys: "phistar", "Lstar", "alpha", "Lcut"

    Returns:
        float: The normalization factor for the Schechter function.

    Notes:
        The function uses the gammaincc and gamma functions from scipy.special module.
        The normalization factor is calculated using the formula:
            norm = phistar * Lstar * gamma(alpha + 2) * gammaincc(alpha + 2, Lcut)
        where gammaincc is the complemented incomplete gamma function.
    """

    phi_star, alpha, L_cut = [lambda_mag[i] for i in ["phi_star", "alpha", "L_cut"]]

    return phi_star * gamma(alpha + 1) * gammaincc(alpha + 1, L_cut)







__all__ = ['EvolvingDoubleSchechter']

class EvolvingDoubleSchechter:
    """
    Double Schechter function with polynomial, exponential, or constant redshift evolution
    in (1+z).
    """

    def __init__(self, zkind="poly", params=None, M_min=None, M_max=12.5, log=True, int_res=300, z_min=0., z_max=jnp.inf):
        """
        Args:
            zkind (str) : 'poly', 'exp', or 'const' for parameter evolution.
            params (dict): Schechter parameters; keys in {'phi1','phi2','alpha1','alpha2','logMstar'}.
            M_min (callable|float|int|None): Lower mass (logM if log=True) limit or function of z.
            M_max (float): Upper mass (logM if log=True) limit (constant).
            log (bool)   : Use log10(M/Msun). Defaults to True.
            int_res (int|None): Number of points for Simpson's rule integration, if None uses `quad`. Defaults to 300.
            z_min (float): Minimum redshift
        """
        self.zkinds_allowed = {"poly", "exp", "const"}
        if zkind not in self.zkinds_allowed:
            raise ValueError(f"zkind must be one of {self.zkinds_allowed}. Got '{zkind}'.")

        self.zkind   = zkind
        self.log     = log
        self.int_res = int_res

        self.params = self._validate_params(params or self._default_params())
        self.M_min_fcn = self._validate_M_min(M_min)
        self.M_max = M_max
        self.z_min = z_min
        self.z_max = z_max

        self._default_labels()

    # -------------------------------------------------------------------------
    # Defaults
    # -------------------------------------------------------------------------
    def _default_labels(self):
        """
        Sets the axis labels based on whether masses are in log-space or linear.
        """
        if self.log:
            self.xlab, self.ylab = r'$\log(M/M_{\odot})$', r'$\Phi(\log M|z) \, [\mathrm{Mpc^{-3}\,dex^{-1}}]$'
        else:
            self.xlab, self.ylab = r'$M/M_{\odot}$', r'$\Phi(M|z) \, [\mathrm{Mpc^{-3}}]$'

        self.ylab_n = r"$n(z)$ [Mpc$^{-3}$]"
        self.ylab_rho = r"$\rho(z)$ [M$_{\odot}$ Mpc$^{-3}$]"

    def _default_params(self):
        """Returns COSMOS+2020-like coefficients as an example."""
        p = SchecheterParsCOSMOS2020

        if self.zkind == "const":
            # take only the zeroth coefficient
            return {k: v[0] for k, v in p.items()}
        elif self.zkind == "exp":
            raise NotImplementedError("No default exponential coefficients provided.")
        return p

    def _default_M_min(self):
        """Example mass limit function from Weaver+23, simplified for demonstration."""
        f_base = SchechterMlimMICE_B25

        if self.zkind == 'const':
            val = f_base(0)  # use z=0 as a constant
            return lambda z: np.log10(val) if self.log else val

        return lambda z: np.log10(f_base(z)) if self.log else f_base(z)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_params(self, params):
        """Check each parameter array shape matches zkind requirements."""
        required = {'phi1', 'phi2', 'alpha1', 'alpha2', 'logMstar'}
        if set(params.keys()) != required:
            missing = required - set(params.keys())
            extra   = set(params.keys()) - required
            raise KeyError(f"Missing: {missing}, extra: {extra}")

        out = {}
        for k, v in params.items():
            arr = np.atleast_1d(v)
            if self.zkind == 'const' and arr.size != 1:
                raise ValueError(f"'{k}' must have 1 coeff if zkind='const'. Got: {arr}")
            if self.zkind == 'exp'   and arr.size != 2:
                raise ValueError(f"'{k}' must have 2 coeff if zkind='exp'. Got: {arr}")
            if self.zkind == 'poly'  and arr.size < 1:
                raise ValueError(f"'{k}' must have >=1 coeff if zkind='poly'. Got: {arr}")
            out[k] = arr
        return out

    def _validate_M_min(self, M_min):
        """Ensure M_min is callable or convert it to a constant function."""
        if M_min is None:
            return self._default_M_min()
        if isinstance(M_min, (float, int)):
            return lambda z: M_min
        if not callable(M_min):
            raise ValueError("M_min must be None, float, int, or callable.")
        return M_min

    def set_params(self, new_params):
        """
        Allows updating the Schechter parameters after instantiation,
        while preserving the validation logic.
        """
        self.params = self._validate_params(new_params)

    def set_M_min(self, new_M_min):
        """
        Allows updating the M_min function after instantiation,
        while preserving the validation logic.
        """
        self.M_min_fcn = self._validate_M_min(new_M_min)

    # -------------------------------------------------------------------------
    # Parameter evolution
    # -------------------------------------------------------------------------
    def param_at_z(self, z, coeffs):
        """Compute parameter value(s) at redshift z."""
        zp1 = 1. + z
        if self.zkind == "const":
            return jnp.full_like(zp1, coeffs[0])
        elif self.zkind == "exp":
            return coeffs[0] * zp1**coeffs[1]
        elif self.zkind == "poly":
            # np.polyval expects highest power first => reverse coeff array
            return jnp.polyval(coeffs[::-1], zp1)

    # -------------------------------------------------------------------------
    # Schechter evaluations
    # -------------------------------------------------------------------------

    def schechter_on_M(self, M, z=0., norm_Mstar=False):
        """Double Schechter in linear mass. JAX-compatible version."""
        M, z = jnp.atleast_1d(M), jnp.atleast_1d(z)

        # Evaluate each param at z
        phi1 = self.param_at_z(z, self.params['phi1'])
        phi2 = self.param_at_z(z, self.params['phi2'])
        alpha1 = self.param_at_z(z, self.params['alpha1'])
        alpha2 = self.param_at_z(z, self.params['alpha2'])
        Ms = jnp.power(10.0, self.param_at_z(z, self.params['logMstar']))

        x = M[:, None] / Ms
        phi = (phi1 * jnp.power(x, alpha1) + phi2 * jnp.power(x, alpha2)) * jnp.exp(-x)

        return phi / (Ms if norm_Mstar else 1.0)


    def schechter_on_logM(self, logM, z=0.0):
        """Double Schechter in log(M). JAX-compatible version."""
        logM, z = jnp.atleast_1d(logM), jnp.atleast_1d(z)

        # Evaluate each parameter at z
        phi1 = self.param_at_z(z, self.params['phi1'])
        phi2 = self.param_at_z(z, self.params['phi2'])
        alpha1 = self.param_at_z(z, self.params['alpha1'])
        alpha2 = self.param_at_z(z, self.params['alpha2'])
        logMs = self.param_at_z(z, self.params['logMstar'])

        # Compute the Schechter function
        x = jnp.power(10.0, logM[:, None] - logMs)
        schechter_values = jnp.log(10.0) *\
                           (phi1 * jnp.power(x, alpha1 + 1) + phi2 * jnp.power(x, alpha2 + 1)) * jnp.exp(-x)

        return schechter_values.reshape(logM.shape)



    # -------------------------------------------------------------------------
    # Integration wrappers
    # -------------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _density_integral_trapz(self, z, power=0.):
        """JAX-compatible integration over mass dimension."""
        z = jnp.atleast_1d(z)
        lower = jnp.maximum(self.M_min_fcn(z), 1e-99)

        m_grid = jnp.linspace(lower, self.M_max, self.int_res)

        if self.log:
            integrand = self.schechter_on_logM(m_grid, z) * jnp.power(10.0, m_grid * power)
        else:
            integrand = self.schechter_on_M(m_grid, z, norm_Mstar=True) * jnp.power(m_grid, power)

        return trapz(integrand, m_grid, axis=0)


    # -------------------------------------------------------------------------
    # Public integration methods (parallelizable)
    # -------------------------------------------------------------------------
    def weighted_density(self, z_array, power=0.):
        """
        M^power * Schechter integrated from M_min(z) to M_max, for each z.
        Returns zero for redshifts outside [zmin, zmax] range.
        """
        z_array = jnp.atleast_1d(z_array)
        # Apply z range constraint using JAX's where function
        result = jnp.where(
            (z_array >= self.z_min) & (z_array <= self.z_max),
            self._density_integral_trapz(z_array, power),
            0.
        )
        return result

    def number_density(self, z_array):
        """Equivalent to weighted_density with power=0."""
        return self.weighted_density(z_array, power=0.)

    def mass_density(self, z_array):
        """Equivalent to weighted_density with power=1 (linear mass)."""
        return self.weighted_density(z_array, power=1.)


    # -------------------------------------------------------------------------
    # Grid evaluations
    # -------------------------------------------------------------------------
    def evaluate(self, z_min=0., z_max=1.5, Nz=20, Nm=100, mask=True):
        """
        Evaluate the Schechter function on a 2D grid of redshift and mass.

        Args:
            z_min (float): minimum redshift
            z_max (float): maximum redshift
            Nz (int): number of redshift bins
            Nm (int): number of mass bins
            mask (bool): mask out-of-range values

        Returns:
            np.ndarray: 2D array of Schechter values
            np.ndarray: mass grid values
            np.ndarray: redshift grid values
        """

        zvals = np.linspace(z_min, z_max, Nz)
        Mmin  = self.M_min_fcn(z_min)
        Mvals = np.linspace(Mmin, self.M_max, Nm)
        phi_2d = self.compute(Mvals, zvals)

        if mask:
            mask_out = (Mvals[:, None] < self.M_min_fcn(zvals)[None, :]) | (Mvals[:, None] > self.M_max)
            phi_2d[mask_out] = np.nan

        return phi_2d, Mvals, zvals


    # -------------------------------------------------------------------------
    # Debugging and string representation
    # -------------------------------------------------------------------------
    def __repr__(self):
        # Try to represent the M_min_fcn nicely: show function name or repr if it's a constant
        if callable(self.M_min_fcn):
            mmin_str = getattr(self.M_min_fcn, '__name__', repr(self.M_min_fcn))
        else:
            # if it's a float/int or something else
            mmin_str = repr(self.M_min_fcn)

        # Format the parameters in a readable block
        param_lines = []
        for key in sorted(self.params.keys()):
            param_lines.append(f"    {key}: {self.params[key]}")
        param_block = "\n".join(param_lines)

        return (
            f"{self.__class__.__name__}(\n"
            f"  zkind='{self.zkind}', log={self.log},\n"
            f"  M_min_fcn={mmin_str}, M_max={self.M_max},\n"
            f"  params={{\n{param_block}\n  }},\n"
            f")"
        )
