"""Stellar mass functions and Schechter function utilities.

This module implements evolving double Schechter mass functions with redshift-dependent
parameters for modeling galaxy stellar mass distributions. Includes:
- Polynomial, exponential, or constant parameter evolution with (1+z)
- Number density and mass density integration
- JAX-compatible implementations for automatic differentiation
- Default parameters from Borghi+25 (MICE simulation)
"""

import logging
logs = logging.getLogger(__name__)
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc, gamma

from .config import jnp, jax
from .math import trapz
from functools import partial

SchecheterParsMICE_B25 = {
    "alpha1":   [-0.3,          0.,          0.        ],
    "alpha2":   [-1.5,          0.,          0.        ],
    "logMstar": [10.5,          0.,          0.        ],
    "phi1":     [ 0.00147607,   0.00344156, -0.00168267],
    "phi2":     [ 0.00322851,  -0.00240074,  0.00057019]
}
"""dict: Default double Schechter parameters from Borghi+25 MICE analysis.
Polynomial coefficients for evolution with (1+z): p(z) = c0 + c1*(1+z) + c2*(1+z)^2
"""

SchechterMlimMICE_B25 = lambda z: 2.89766937 + 5.32024468*(1+z) - 1.04370681*(1+z)**2
"""callable: Default mass completeness limit from Borghi+25 as function of redshift.
Returns log10(M_min/Msun) for polynomial evolution with (1+z).
"""


__all__ = ['EvolvingDoubleSchechter']

class EvolvingDoubleSchechter:
    """Double Schechter stellar mass function with redshift evolution.
    
    Implements the double Schechter function:
    Φ(M|z) = ln(10) * [φ₁(z) * x^(α₁+1) + φ₂(z) * x^(α₂+1)] * exp(-x)
    where x = M/M*(z) for linear mass or x = 10^(logM - logM*(z)) for log mass.
    
    All parameters (φ₁, φ₂, α₁, α₂, M*) can evolve with redshift via polynomial,
    exponential, or constant prescriptions.
    """

    def __init__(self, zkind="poly", params=None, M_min=None, M_max=12.5, log=True, int_res=300, z_min=0., z_max=jnp.inf):
        """Initializes the evolving double Schechter mass function.
        
        Args:
            zkind (str, optional): Parameter evolution type - 'poly', 'exp', or 'const'. Defaults to 'poly'.
            params (dict, optional): Schechter parameters with keys {'phi1', 'phi2', 'alpha1', 'alpha2', 'logMstar'}.
                                    For 'poly': list of coefficients [c0, c1, c2, ...] for p(z) = Σ cᵢ(1+z)^i
                                    For 'exp': [c0, c1] for p(z) = c0 * (1+z)^c1
                                    For 'const': scalar value
                                    Defaults to Borghi+25 parameters.
            M_min (callable|float|None, optional): Lower mass limit. If callable, function of z.
                                                   In log10(M/Msun) if log=True, else linear M/Msun.
                                                   Defaults to Borghi+25 completeness.
            M_max (float, optional): Upper mass limit (constant). Defaults to 12.5 (log) or 10^12.5 (linear).
            log (bool, optional): Work in log10(M/Msun) space. Defaults to True.
            int_res (int, optional): Integration grid resolution. Defaults to 300.
            z_min (float, optional): Minimum valid redshift. Defaults to 0.
            z_max (float, optional): Maximum valid redshift. Defaults to inf.
        
        Raises:
            ValueError: If zkind not in {'poly', 'exp', 'const'}
            KeyError: If params missing required keys
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
        """Sets plot axis labels based on mass representation (log vs linear)."""
        if self.log:
            self.xlab, self.ylab = r'$\log(M/M_{\odot})$', r'$\Phi(\log M|z) \, [\mathrm{Mpc^{-3}\,dex^{-1}}]$'
        else:
            self.xlab, self.ylab = r'$M/M_{\odot}$', r'$\Phi(M|z) \, [\mathrm{Mpc^{-3}}]$'

        self.ylab_n = r"$n(z)$ [Mpc$^{-3}$]"
        self.ylab_rho = r"$\rho(z)$ [M$_{\odot}$ Mpc$^{-3}$]"

    def _default_params(self):
        """Returns default Schechter parameters from Borghi+25.
        
        Returns:
            dict: Parameter dictionary compatible with zkind setting
        """
        p = SchecheterParsMICE_B25

        if self.zkind == "const":
            # take only the zeroth coefficient
            return {k: v[0] for k, v in p.items()}
        elif self.zkind == "exp":
            raise NotImplementedError("No default exponential coefficients provided.")
        return p

    def _default_M_min(self):
        """Returns default mass completeness limit from Borghi+25.
        
        Returns:
            callable: Function M_min(z) in appropriate units (log or linear)
        """
        f_base = SchechterMlimMICE_B25

        if self.zkind == 'const':
            val = f_base(0)  # use z=0 as a constant
            return lambda z: np.log10(val) if self.log else val

        return lambda z: np.log10(f_base(z)) if self.log else f_base(z)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_params(self, params):
        """Validates parameter dictionary structure.
        
        Args:
            params (dict): Parameter dictionary to validate
        
        Returns:
            dict: Validated parameters with arrays
        
        Raises:
            KeyError: If required keys missing or extra keys present
            ValueError: If coefficient array sizes don't match zkind
        """
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
        """Validates and converts M_min to callable.
        
        Args:
            M_min: None, scalar, or callable
        
        Returns:
            callable: Function M_min(z)
        
        Raises:
            ValueError: If M_min is invalid type
        """
        if M_min is None:
            return self._default_M_min()
        if isinstance(M_min, (float, int)):
            return lambda z: M_min
        if not callable(M_min):
            raise ValueError("M_min must be None, float, int, or callable.")
        return M_min

    def set_params(self, new_params):
        """Updates Schechter parameters.
        
        Args:
            new_params (dict): New parameter dictionary (validated)
        """
        self.params = self._validate_params(new_params)

    def set_M_min(self, new_M_min):
        """Updates mass completeness limit.
        
        Args:
            new_M_min: New M_min (callable, scalar, or None)
        """
        self.M_min_fcn = self._validate_M_min(new_M_min)

    # -------------------------------------------------------------------------
    # Parameter evolution
    # -------------------------------------------------------------------------
    def param_at_z(self, z, coeffs):
        """Evaluates parameter at given redshift(s).
        
        Args:
            z (array-like): Redshift(s)
            coeffs (array-like): Parameter coefficients
        
        Returns:
            jnp.ndarray: Parameter value(s) at z
        """
        zp1 = 1. + z
        if self.zkind == "const":
            return jnp.full_like(zp1, coeffs[0])
        elif self.zkind == "exp":
            return coeffs[0] * zp1**coeffs[1]
        elif self.zkind == "poly":
            # np.polyval expects highest power first => reverse coeff array
            return jnp.polyval(coeffs[::-1], zp1)

    # -------------------------------------------------------------------------
    # Schechter evaluation
    # -------------------------------------------------------------------------
    def schechter_on_M(self, M, z=0., norm_Mstar=False):
        """Evaluates double Schechter function in linear mass.
        
        Args:
            M (array-like): Stellar mass [M_sun]
            z (array-like, optional): Redshift(s). Defaults to 0.
            norm_Mstar (bool, optional): Divide by M*(z) for normalization. Defaults to False.
        
        Returns:
            jnp.ndarray: Schechter function values Φ(M|z) [Mpc^-3 M_sun^-1 or Mpc^-3]
        """
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
        """Evaluates double Schechter function in log-mass.
        
        Args:
            logM (array-like): log10(M/M_sun)
            z (array-like, optional): Redshift(s). Defaults to 0.
        
        Returns:
            jnp.ndarray: Schechter function values Φ(logM|z) [Mpc^-3 dex^-1]
        """
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
    # Integration
    # -------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _density_integral_trapz(self, z, power=0.):
        """Integrates M^power * Φ(M|z) over mass range [M_min(z), M_max].
        
        Args:
            z (array-like): Redshift(s)
            power (float, optional): Mass weighting exponent. Defaults to 0 (number density).
        
        Returns:
            jnp.ndarray: Integrated density at each redshift
        """
        z = jnp.atleast_1d(z)
        lower = jnp.maximum(self.M_min_fcn(z), 1e-99)

        m_grid = jnp.linspace(lower, self.M_max, self.int_res)

        if self.log:
            integrand = self.schechter_on_logM(m_grid, z) * jnp.power(10.0, m_grid * power)
        else:
            integrand = self.schechter_on_M(m_grid, z, norm_Mstar=True) * jnp.power(m_grid, power)

        return trapz(integrand, m_grid, axis=0)


    def weighted_density(self, z_array, power=0.):
        """Computes mass-weighted integrated density.
        
        Integrates M^power * Φ(M|z) from M_min(z) to M_max.
        Returns zero outside [z_min, z_max] range.
        
        Args:
            z_array (array-like): Redshift(s)
            power (float, optional): Mass exponent. power=0 gives number density,
                                    power=1 gives mass density. Defaults to 0.
        
        Returns:
            jnp.ndarray: Integrated density [Mpc^-3] (power=0) or [M_sun Mpc^-3] (power=1)
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
        """Computes comoving number density.
        
        Args:
            z_array (array-like): Redshift(s)
        
        Returns:
            jnp.ndarray: Number density n(z) [Mpc^-3]
        """
        return self.weighted_density(z_array, power=0.)

    def mass_density(self, z_array):
        """Computes comoving stellar mass density.
        
        Args:
            z_array (array-like): Redshift(s)
        
        Returns:
            jnp.ndarray: Mass density ρ(z) [M_sun Mpc^-3]
        """
        return self.weighted_density(z_array, power=1.)


    # -------------------------------------------------------------------------
    # Grid evaluations
    # -------------------------------------------------------------------------
    def evaluate(self, z_min=0., z_max=1.5, Nz=20, Nm=100, mask=True):
        """Evaluates Schechter function on 2D (mass, redshift) grid.
        
        Args:
            z_min (float, optional): Minimum redshift. Defaults to 0.
            z_max (float, optional): Maximum redshift. Defaults to 1.5.
            Nz (int, optional): Number of redshift points. Defaults to 20.
            Nm (int, optional): Number of mass points. Defaults to 100.
            mask (bool, optional): Mask values outside valid mass range. Defaults to True.
        
        Returns:
            tuple: (phi_2d, Mvals, zvals)
                - phi_2d (np.ndarray): Shape (Nm, Nz) Schechter values
                - Mvals (np.ndarray): Mass grid
                - zvals (np.ndarray): Redshift grid
        """

        zvals = np.linspace(z_min, z_max, Nz)
        Mmin  = self.M_min_fcn(z_min)
        Mvals = np.linspace(Mmin, self.M_max, Nm)
        if self.log:
            phi_2d = self.schechter_on_logM(Mvals, zvals)
        else:
            phi_2d = self.schechter_on_M(Mvals, zvals, norm_Mstar=True)

        if mask:
            mask_out = (Mvals[:, None] < self.M_min_fcn(zvals)[None, :]) | (Mvals[:, None] > self.M_max)
            phi_2d[mask_out] = np.nan

        return phi_2d, Mvals, zvals


    # -------------------------------------------------------------------------
    # Debugging and string representation
    # -------------------------------------------------------------------------
    def __repr__(self):
        """Returns string representation of the Schechter function configuration."""
        if callable(self.M_min_fcn):
            mmin_str = getattr(self.M_min_fcn, '__name__', repr(self.M_min_fcn))
        else:
            mmin_str = repr(self.M_min_fcn)

        param_lines = [f"    {key}: {self.params[key]}" for key in sorted(self.params.keys())]
        param_block = "\n".join(param_lines)

        return (
            f"{self.__class__.__name__}(\n"
            f"  zkind='{self.zkind}', log={self.log},\n"
            f"  M_min_fcn={mmin_str}, M_max={self.M_max},\n"
            f"  params={{\n{param_block}\n  }}\n"
            f")"
        )
