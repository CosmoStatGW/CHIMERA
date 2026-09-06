# -*- coding: utf-8 -*-

__bibtex__ = """
@article{Borghi:2023opd,
  author = "Borghi, Nicola and Mancarella, Michele and Moresco, Michele and Tagliazucchi, Matteo and Iacovelli, Francesco and Cimatti, Andrea and Maggiore, Michele",
  title = "{Cosmology and Astrophysics with Standard Sirens and Galaxy Catalogs in View of Future Gravitational Wave Observations}",
  eprint = "2312.05302",
  archivePrefix = "arXiv",
  primaryClass = "astro-ph.CO",
  doi = "10.3847/1538-4357/ad20eb",
  journal = "Astrophys. J.",
  volume = "964",
  number = "2",
  pages = "191",
  year = "2024"
}
"""
__url__ = "https://chimera-gw.readthedocs.io"
__author__ = "Nicola Borghi, Matteo Tagliazucchi"
__email__ = "nicola.borghi6#unibo.it"
__license__ = "MIT"
__description__ = "Python tool for gravitational-wave cosmology with galaxy catalogs"

__version__ = "2.0.0"

from typing import Any
import sys

from . import utils
from . import data
from .population import * #cosmo, mass, rate, population, compute_z_grids
from .likelihood import hyperlikelihood
from .selection_function import selection_function
from .catalog import completeness, catalog


sys.modules[__name__ +".cosmo"] = cosmo
sys.modules[__name__ +".catalog"] = catalog
sys.modules[__name__ +".rate"] = rate
sys.modules[__name__ +".completeness"] = completeness

sys.modules[__name__ + ".mass"] = mass
sys.modules[__name__ + ".mass.paired"] = mass.paired
sys.modules[__name__ + ".mass.conditioned"] = mass.conditioned

