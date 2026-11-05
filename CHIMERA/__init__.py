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
__email__ = "nicola.borghi6@unibo.it"
__license__ = "GPL-3.0-only"
__description__ = "Python tool for gravitational-wave cosmology with galaxy catalogs"

from importlib.metadata import PackageNotFoundError as _PNF, version as _version
try:
  __version__ = _version("chimera-gw")  # single source of truth: pyproject.toml
except _PNF:
  __version__ = "unknown"

from typing import Any
import sys

from . import utils
from . import data
from .population import * #cosmo, mass, rate, population, compute_z_grids
from .likelihood import hyperlikelihood
from .selection_function import selection_function
from .catalog import completeness, catalog


sys.modules[__name__ +".catalog"] = catalog


# Short import paths: `CHIMERA.mass...` -> `CHIMERA.population.mass...` etc.
# A meta-path hook (rather than plain sys.modules entries) so that aliases work
# at any depth (e.g. `from CHIMERA.mass.joint.bspline2d import ...`) and always
# resolve to the *same* module objects as the real path -- loading a second copy
# under the alias name would duplicate classes and break plum dispatch.
import importlib
import importlib.abc
import importlib.machinery
import importlib.util

_ALIASES = {
  __name__ + ".mass": __name__ + ".population.mass",
  __name__ + ".cosmo": __name__ + ".population.cosmo",
  __name__ + ".rate": __name__ + ".population.rate",
  __name__ + ".completeness": __name__ + ".catalog.completeness",
}

class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
  def __init__(self):
    self._real_specs = {}

  def _target(self, fullname):
    for alias, real in _ALIASES.items():
      if fullname == alias or fullname.startswith(alias + "."):
        return real + fullname[len(alias):]
    return None

  def find_spec(self, fullname, path=None, target=None):
    target = self._target(fullname)
    if target is None or importlib.util.find_spec(target) is None:
      return None
    return importlib.machinery.ModuleSpec(fullname, self)

  def create_module(self, spec):
    module = importlib.import_module(self._target(spec.name))
    self._real_specs[spec.name] = module.__spec__
    return module

  def exec_module(self, module):
    # module_from_spec overwrote __spec__ with the alias spec; restore it
    module.__spec__ = self._real_specs.pop(module.__spec__.name)

if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
  sys.meta_path.insert(0, _AliasFinder())

