
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
