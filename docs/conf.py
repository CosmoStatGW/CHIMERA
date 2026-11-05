import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

sys.path.insert(0, os.path.abspath(".."))

try:
    __version__ = _pkg_version("chimera-gw")
except PackageNotFoundError:
    __version__ = "unknown version"

# Tutorials: single source of truth is ../examples, copied here at build time.
import shutil
_TUTORIALS = [
    "spectral_sirens_plp",
    "spectral_sirens_spline",
    "conditioned_vs_paired_masses",
    "complete_galaxy_catalog",
    "incomplete_galaxy_catalog",
]
_here = os.path.dirname(os.path.abspath(__file__))
for _nb in _TUTORIALS:
    shutil.copy2(os.path.join(_here, "..", "examples", _nb + ".ipynb"),
                 os.path.join(_here, "user_guide", "tutorials", _nb + ".ipynb"))

# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}
master_doc = "index"

project = "CHIMERA"
author = "Nicola Borghi, Matteo Tagliazucchi"
copyright = "2023-2026, the CHIMERA developers"
version = __version__
release = __version__
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Autodoc: heavy optional backends are mocked so that a missing GPU/MPI/sampler
# stack never breaks the documentation build.
autodoc_mock_imports = [
    "cupy", "cupyx", "mpi4py", "mpi4jax", "schwimmbad",
    "zeus", "dill", "numpyro", "arviz",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

napoleon_use_ivar = True  # equinox fields: avoid duplicate attribute entries

# plum-dispatched functions: under Sphinx, plum renders the extra overloads as
# ``.. py:function::`` directives, but leaves the ``:noindex:`` option unindented when
# there is a single extra overload (docutils then reads it as text, giving a
# duplicate-object warning). Re-indent it.
def _fix_plum_docstring(app, what, name, obj, options, lines):
    if type(obj).__module__.startswith("plum"):
        lines[:] = ["   :noindex:" if l == ":noindex:" else l for l in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _fix_plum_docstring)


suppress_warnings = ["myst.header", "mystnb.unknown_mime_type"]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "CHIMERA"
html_favicon = "_static/CHIMERA_logoNB2.ico"
html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/CosmoStatGW/CHIMERA",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "logo": {
        "image_light": "_static/CHIMERA_logoNB2.svg",
        "image_dark": "_static/CHIMERA_logoNB2_dark.svg",
    },
}

# Notebooks are rendered with their stored outputs (they need GW data to run).
nb_execution_mode = "off"
nb_execution_timeout = -1
