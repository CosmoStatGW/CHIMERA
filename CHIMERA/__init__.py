# -*- coding: utf-8 -*-

__bibtex__ = """
@article{...
}
"""
__uri__ = "https://chimera-gw.readthedocs.io"
__author__ = "Nicola Borghi"
__email__ = "nicoborghi@outlook.com"
__license__ = "MIT"
__description__ = "Python tool for gravitational-wave cosmology with galaxy catalogs"


from typing import Any
from .__version__ import __version__  # isort:skip

from . import MCMC
from . import Likelihood
from . import Bias
from . import GW 
from . import EM
from . import DataGW
from . import DataEM
from . import cosmo, astro, utils

from jax.config import config
config.update("jax_enable_x64", True)


__all__ = [
    "MCMC",
    "Likelihood",
    "Bias",
    "GW",
    "EM",
    "DataGW",
    "DataEM",
    "cosmo",
    "astro",
    "utils",
    "__version__",
]
