# CHIMERA

<img src="https://raw.githubusercontent.com/CosmoStatGW/CHIMERA/main/docs/_static/CHIMERA_logoNB2.svg" alt="CHIMERA" width=300px>


**CHIMERA** (Combined Hierarchical Inference Model for Electromagnetic and gRavitational-wave Analysis) is a flexible Python code to analyze standard sirens with galaxy catalogs, allowing for a joint fitting of the cosmological and astrophysical population parameters within a Hierarchical Bayesian Inference framework.

The code is designed to be accurate for different scenarios, encompassing bright, dark, and spectral sirens methods, and computationally efficient in view of next-generation GW observatories and galaxy surveys. It uses the LAX-backend implementation and Just In Time (JIT) computation capabilities of JAX.

[![GitHub](https://img.shields.io/badge/GitHub-CHIMERA-9e8ed7)](https://github.com/CosmoStatGW/CHIMERA/)
[![arXiv](https://img.shields.io/badge/arXiv-2106.14894-28bceb)](https://arxiv.org/abs/2106.14894)
[![Read the Docs](https://readthedocs.org/projects/chimera-gw/badge/?version=latest)](https://chimera-gw.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-fb7e21)](https://github.com/CosmoStatGW/CHIMERA/blob/main/LICENSE)
[![GitLab](https://img.shields.io/github/v/tag/CosmoStatGW/CHIMERA?label=latest-release&color=da644d)](https://github.com/CosmoStatGW/CHIMERA/releases)


## Installation

The code can be quikly installed from [Pypi](https://pypi.org/project/chimera-gw/):

    pip install chimera-gw

For more flexibility, clone the source repository into your working folder and install it locally:

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .

To test the installation, run the following command:

    python -c "import CHIMERA; print(CHIMERA.__version__)"


## Documentation

The full documentation is provided at [chimera-gw.readthedocs.io](https://chimera-gw.readthedocs.io)


## Citation

If you find this code useful in your research, please cite the following paper ([ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv231205302B/), [arXiv](https://arxiv.org/abs/2312.05302), [INSPIRE](https://inspirehep.net/literature/2734729)):


    @ARTICLE{2023arXiv231205302B,
        author = {{Borghi}, Nicola and {Mancarella}, Michele and {Moresco}, Michele and et al.},
            title = "{Cosmology and Astrophysics with Standard Sirens and Galaxy Catalogs in View of Future Gravitational Wave Observations}",
        journal = {arXiv e-prints},
        keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, General Relativity and Quantum Cosmology},
            year = 2023,
            month = dec,
            eid = {arXiv:2312.05302},
            pages = {arXiv:2312.05302},
            doi = {10.48550/arXiv.2312.05302},
    archivePrefix = {arXiv},
        eprint = {2312.05302},
    primaryClass = {astro-ph.CO},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231205302B},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
