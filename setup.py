#!/usr/bin/env python

import codecs
import os
import re
from setuptools import find_packages, setup

# PROJECT SPECIFIC

NAME = "CHIMERA"
PACKAGES = find_packages(where="CHIMERA")
META_PATH = os.path.join("CHIMERA", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
]
INSTALL_REQUIRES = install_requires=[
  "python>=3.10",
  "wheel",
  "cython",
  "numpy>=1.26",
  "scipy^1.11.4",
  "astropy^5.2",
  "h5py^3.5",
  "healpy^1.14",
  "matplotlib^3.4",
  "numba>=0.58",
  "scikit-learn>=1.0",
  "jax^0.4.31",
  "interpax",
  "equinox",
  "plum-dispatch",
  "emcee",
  "packaging",
  "mpi4py^4.0",
  "mpi4jax; extra == 'no-build-isolation'",
  "schwimmbad^0.3.2"
],
SETUP_REQUIRES = ["setuptools>=40.6.0","setuptools_scm","wheel"]

# END PROJECT SPECIFIC

HERE = os.path.dirname(os.path.realpath(__file__))

def read(*parts):
  with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
    return f.read()

def find_meta(meta, meta_file=read(META_PATH)):
  meta_match = re.search(
      r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
  )
  if meta_match:
    return meta_match.group(1)
  raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

def get_version():
  """Read version from the __init__.py file."""
  init_file = os.path.join(NAME, "__init__.py")
  with open(init_file) as f:
    for line in f:
      if line.startswith('__version__'):
        # Extract the version number from the line like: __version__ = '1.0.0'
        version = line.split('=')[1].strip().strip('"').strip("'")
        return version
  raise RuntimeError("Version string not found in __init__.py")

if __name__ == "__main__":
  setup(
    name=NAME,
    version = get_version(),
    author=find_meta("author"),
    author_email=find_meta("email"),
    maintainer=find_meta("author"),
    maintainer_email=find_meta("email"),
    url=find_meta("uri"),
    project_urls={
      "Source": "https://github.com/CosmoStatGW/CHIMERA",
    },
    license=find_meta("license"),
    description=find_meta("description"),
    long_description=read("README.md"),
    long_description_content_type="text/x-rst",
    packages=PACKAGES,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    classifiers=CLASSIFIERS,
    zip_safe=False,
    options={"bdist_wheel": {"universal": "1"}},
)
