# CHIMERA

<img src="https://raw.githubusercontent.com/CosmoStatGW/CHIMERA/main/docs/_static/CHIMERA_logoNB2.svg" alt="CHIMERA" width=300px>


**CHIMERA** (Combined Hierarchical Inference Model for Electromagnetic and gRavitational-wave Analysis) is a flexible Python code to analyze standard sirens with galaxy catalogs, allowing for a joint fitting of the cosmological and astrophysical population parameters within a Hierarchical Bayesian Inference framework.

The code is designed to be accurate for different scenarios, encompassing bright, dark, and spectral siren methods.
The code takes full advantage of JAX features such as JIT compilation, vectorization, and GPU acceleration to be computationally efficient for the next-generation GW observatories and galaxy surveys.

[![GitHub](https://img.shields.io/badge/GitHub-CHIMERA-9e8ed7)](https://github.com/CosmoStatGW/CHIMERA/)
[![arXiv](https://img.shields.io/badge/arXiv-2106.14894-28bceb)](https://arxiv.org/abs/2106.14894)
[![Read the Docs](https://readthedocs.org/projects/chimera-gw/badge/?version=latest)](https://chimera-gw.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-fb7e21)](https://github.com/CosmoStatGW/CHIMERA/blob/main/LICENSE)
[![GitLab](https://img.shields.io/github/v/tag/CosmoStatGW/CHIMERA?label=latest-release&color=da644d)](https://github.com/CosmoStatGW/CHIMERA/releases)


## Installation

The code can be quikly installed from [Pypi](https://pypi.org/project/chimera-gw/):

    pip install chimera-gw

For more flexibility, clone the source repository into your working folder and install it locally (or append the local folder using `sys`):

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .

To test the installation, run the following command:

    python -c "import CHIMERA; print(CHIMERA.__version__)"

To install and use the code on HPC facilities with GPU nodes follow, the instructions in "install_hpc.txt".


## Documentation

The full documentation is provided at [chimera-gw.readthedocs.io](https://chimera-gw.readthedocs.io)


## Citation

If you find this code useful in your research, please cite the following papers

    @ARTICLE{2024ApJ...964..191B,
      author = {{Borghi}, Nicola and {Mancarella}, Michele and {Moresco}, Michele and {Tagliazucchi}, Matteo and {Iacovelli}, Francesco and {Cimatti}, Andrea and {Maggiore}, Michele},
      title = "{Cosmology and Astrophysics with Standard Sirens and Galaxy Catalogs in View of Future Gravitational Wave Observations}",
      journal = {\apj},
      keywords = {Observational cosmology, Gravitational waves, Cosmological parameters, 1146, 678, 339, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, General Relativity and Quantum Cosmology},
      year = 2024,
      month = apr,
      volume = {964},
      number = {2},
      eid = {191},
      pages = {191},
      doi = {10.3847/1538-4357/ad20eb},
      archivePrefix = {arXiv},
      eprint = {2312.05302},
      primaryClass = {astro-ph.CO},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...964..191B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @article{Tagliazucchi:2025ofb,
      author = "Tagliazucchi, Matteo and Moresco, Michele and Borghi, Nicola and Fiebig, Manfred",
      title = "{Accelerating the Standard Siren Method: Improved Constraints on Modified Gravitational Wave Propagation with Future Data}",
      eprint = "2504.02034",
      archivePrefix = "arXiv",
      primaryClass = "astro-ph.CO",
      month = "4",
      year = "2025"
    }
