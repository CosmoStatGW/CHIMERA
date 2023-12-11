# -*- coding: utf-8 -*-

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("CHIMERA").version
except DistributionNotFound:
    __version__ = "unknown version"

import sys
sys.path.insert(0, '../CHIMERA')

# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = ".rst"
master_doc = "index"

project = "CHIMERA"
copyright = "2023, Nicola Borghi & contributors"
version = __version__
release = __version__
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
    }
}
nb_execution_mode = "off"
nb_execution_timeout = -1
