import logging
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc, gamma
import jax.numpy as jnp
import jax
from functools import partial
from CHIMERA.utils.math import trapz
logs = logging.getLogger(__name__)

###################
# Magnitudes
###################

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
    """Computes the Schechter luminosity function in absolute-magnitude space,
    including the standard 0.4*ln(10) magnitude-space Jacobian prefactor (so the
    result is phi(M), density per unit magnitude -- integrable directly over M to
    get an absolute number density, e.g. `CHIMERA.utils.mags.SchechterLF`) and the
    standard h-scaling (M_star -> M_star + 5*log10(h), phi_star -> phi_star*h^3).

    phi(M) = 0.4*ln(10) * phi_star * x^(alpha+1) * exp(-x),
             x = 10**(0.4*(M_star - M))

    NOTE ON SIGN CONVENTION (2026-10-20, FINAL): `x` is `L/L_star`, the
    standard astronomical convention (brighter/more negative M => larger x
    => more exponential suppression at the bright end). This function went
    through three states in one day. It was originally found to use the
    opposite (bright-end-favoring) sign, which was a real bug: it caused an
    8.57% normalization error in `SchechterLF` and was fixed to this standard
    form. It was then deliberately reverted to `x = 10**(0.4*(M - M_star))`
    (the inverse) on explicit user instruction, to match the convention used
    by the (otherwise unused/dormant) `schechter_mags` in the `CHIMERA_private`
    repo's `utils/mags.py`. That revert was then verified to break
    `mth_completeness`'s agreement with arXiv:2605.27227 Appendix B: at
    z=0.05/0.1/0.2 the reverted sign gave F~0.9997/0.9821/0.7905 (still highly
    "complete"), while the standard sign gives F~0.6289/0.1472/0.0002 --
    matching the paper's description of completeness dropping below 1% by
    z~0.2. (`_luminosity_weight(M) = 10**(-0.4*M)` in
    `CHIMERA.catalog.completeness._build_cumulative_schechter` depends on
    absolute M, not M-M_star, so it does NOT symmetrize away the sign choice
    -- contrary to an earlier, overly reassuring note here claiming the ratio
    was convention-independent.) The apparent "the mask-based completeness
    plot looked better" impression under the reverted sign was also diagnosed
    as a normalization artifact: `SchechterLF._n0` (the fixed theoretical
    reference density used by `mask_completeness`) is ~2.3x smaller under the
    reverted sign for GLADE+ Ks parameters (0.00734 vs 0.01715), which
    uniformly inflates every empirical completeness ratio -- not a genuine
    improvement in agreement. Given this, the user asked to switch back to
    the standard sign, which is the state implemented here. Note this also
    does NOT achieve numeric parity with `CHIMERA_private`'s real
    `MaskCompleteness`/`phiSchL_normL` pipeline, which never calls
    `schechter_mags` at all; see `completeness.py`'s `mask_completeness`
    docstring for the remaining architectural differences.

    Args:
        M (jnp.ndarray, float): absolute magnitudes
        alpha (float): faint-end slope of the Schechter LF
        M_star (float): characteristic magnitude of the Schechter LF (standard
            h=1 convention; shifted internally by +5*log10(h))
        phi_star (float): normalization of the Schechter LF at M_star, at h=1
            (standard convention; scaled internally by h**3)
        h (float, optional): Small h=H0/100. Defaults to 1.

    Returns:
        jnp.ndarray: phi(M), Schechter LF density per unit magnitude
    """

    M_star_h   = M_star + 5.*jnp.log10(h)
    phi_star_h = phi_star * h**3
    x          = jnp.power(10., 0.4*(M_star_h - M))  # L/L_star, standard convention (see NOTE above)

    return 0.4 * jnp.log(10.0) * phi_star_h * jnp.power(x, 1.+alpha) * jnp.exp(-x)


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


##################################
# Single (non-evolving) Schechter LF, as a theoretical reference density
##################################

class SchechterLF:
  r"""Single (non-evolving), fixed-magnitude-range Schechter luminosity function in
  absolute-magnitude space, meant as a drop-in ``theory_density_func``/``rho_gal_theo``
  for CHIMERA's *empirical* density-ratio completeness classes
  (:class:`CHIMERA.catalog.completeness.homogeneous_completeness`,
  :class:`CHIMERA.catalog.completeness.mask_completeness`) -- an alternative to the
  stellar-mass-based :class:`EvolvingDoubleSchechter` for a luminosity-selected
  catalog like GLADE+ (Ks-band) or DES (r-band).

  ``number_density(z)`` returns a **fixed** (redshift-independent) comoving number
  density, the Schechter integral over the full ``[M_bright, M_faint]`` range --
  deliberately *not* flux-limited at each z. This matches the gwcosmo/icarogw
  convention (and the analytic definition used by
  :class:`CHIMERA.catalog.completeness.mth_completeness`): the magnitude limit of
  the survey lives on the *observed* side only (the real, flux-limited galaxy
  counts assembled by ``compute_completeness``/``mask_completeness``), while the
  theoretical reference represents the full assumed population. Applying a
  z-dependent flux limit to *both* sides would make the empirical ratio
  double-count the survey's depth and hide genuine incompleteness (sky-coverage
  gaps, extinction, cross-matching failures) instead of revealing it.

  Unlike :class:`CHIMERA.catalog.completeness.mth_completeness` (which only ever
  needs the *ratio* of two Schechter integrals, so ``phi_star`` and the
  ``+5log10(h)`` magnitude shift cancel exactly), here ``number_density(z)`` must
  return a true physical comoving number density in galaxies/Mpc^3 -- to be
  compared against the catalog's actual observed galaxy density by
  ``compute_completeness``, which works in Mpc^3 (see ``CHIMERA.catalog.completeness
  .compute_completeness``: ``V_sky_Mpc = dz*1e9*dVdz*sky_area/(4*pi)``, where the
  ``1e9`` converts CHIMERA's internal Gpc^3 volumes to Mpc^3). ``phi_star`` and the
  ``h``-scaling therefore matter here and must be set correctly, unlike in
  ``mth_completeness``.

  Args:
    alpha (float): Schechter faint-end slope.
    M_star (float): Schechter characteristic absolute magnitude, standard
      ``h=1`` convention (shifted internally by ``+5*log10(h)``).
    phi_star (float): Schechter normalization at ``h=1``, in Mpc^-3 h^3 (standard
      convention; e.g. 0.0117 for DES r-band, 0.0102 for GLADE+ Ks-band, per the
      reference values in arXiv:2605.27227 App. B); scaled internally by ``h**3``.
    M_bright (float): bright-end integration bound (fixed, not z-dependent).
    M_faint (float): faint-end integration bound (fixed, not z-dependent).
    h (float, optional): H0/100, used for the standard Schechter h-scaling.
      Defaults to 1.
    M_int_res (int, optional): resolution of the magnitude integration grid.
      Defaults to 3000.
  """

  def __init__(self, alpha, M_star, phi_star, M_bright, M_faint, h=1., M_int_res=3000):
    self.alpha, self.M_star, self.phi_star = alpha, M_star, phi_star
    self.M_bright, self.M_faint, self.h = M_bright, M_faint, h
    self.M_int_res = M_int_res
    self._n0 = self._integrate_number_density()

  def _integrate_number_density(self):
    """∫_{M_bright}^{M_faint} phi(M) dM, in galaxies/Mpc^3 (standard h-scaling applied).

    Delegates to the module-level `schechter_mags`, which includes the full
    0.4*ln(10) magnitude-space Jacobian prefactor -- this method needs a true
    *absolute* density (not a ratio, unlike
    `CHIMERA.catalog.completeness.mth_completeness`'s self-normalizing F(M)
    ratio, which also uses `schechter_mags` but where that prefactor cancels
    exactly and so is harmless either way). An earlier version of this method
    duplicated the formula inline and dropped that prefactor by mistake
    (copied from the ratio-safe ``mth_completeness`` pattern without
    re-deriving whether that was still valid here), silently overestimating
    the theoretical density by a factor of 1/(0.4*ln(10)) =~ 1.086 (~8.6%),
    biasing every completeness value low by the same amount.
    """
    M_grid = jnp.linspace(self.M_bright, self.M_faint, self.M_int_res)
    phi = schechter_mags(M_grid, self.alpha, self.M_star, self.phi_star, self.h)

    return trapz(phi, M_grid)

  def number_density(self, z):
    """Fixed theoretical comoving number density [galaxies/Mpc^3], broadcast to the
    shape of ``z`` (see class docstring for why this does not depend on z)."""
    z = jnp.atleast_1d(z)
    return jnp.full_like(z, self._n0)







__all__ = ['EvolvingDoubleSchechter', 'SchechterLF']

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
