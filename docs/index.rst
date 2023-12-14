.. CHIMERA documentation master file, created by
   sphinx-quickstart on Sun Jul  4 16:29:39 2021.

.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

.. |br| raw:: html

    <div style="min-height:0.1em;"></div>


*******
CHIMERA
*******

**CHIMERA** (Combined Hierarchical Inference Model for Electromagnetic and gRavitational-wave Analysis), is a flexible Python code to analyze standard sirens with galaxy catalogs, allowing for a joint fitting of the cosmological and astrophysical population parameters within a Hierarchical Bayesian Inference framework. 


.. image:: https://img.shields.io/badge/GitHub-CHIMERA-9e8ed7
    :target: https://github.com/CosmoStatGW/CHIMERA/
    :alt: GitHub
.. image:: https://img.shields.io/badge/arXiv-2106.14894-28bceb
    :target: https://arxiv.org/abs/2106.14894
    :alt: arXiv
.. image:: https://readthedocs.org/projects/chimera-gw/badge/?version=latest
    :target: https://chimera-gw.readthedocs.io/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/license-MIT-fb7e21
    :target: https://github.com/CosmoStatGW/CHIMERA/blob/main/LICENSE
    :alt: License
.. image:: https://img.shields.io/github/v/tag/CosmoStatGW/CHIMERA?label=latest-release&color=da644d
    :target: https://github.com/CosmoStatGW/CHIMERA/releases
    :alt: Release

.. raw:: html

    <br/>


Installation
------------

The code can be quikly installed from `Pypi <https://pypi.org/project/chimera-gw>`_:

.. code-block:: bash

   pip install chimera-gw

For more flexibility, clone the source repository into your working folder and install it locally:

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .

To test the installation, run the following command:

.. code-block:: bash

    python -c "import CHIMERA; print(CHIMERA.__version__)"



License & Attribution
---------------------

**CHIMERA** is free software made available under the MIT License. For details see the ``LICENSE``.

If you find this code useful in your research, please cite the following paper (`ADS <https://ui.adsabs.harvard.edu/abs/2022ApJ...927..164B/abstract>`_, `arXiv <https://arxiv.org/abs/2106.14894>`_, `INSPIRE <https://inspirehep.net/literature/1871797>`_):

.. code-block:: tex

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


The team
--------

**Core team**

- `Nicola Borghi <https://orcid.org/0000-0002-2889-8997>`_ (**main developer**, nicola.borghi6@unibo.it)
- `Michele Mancarella <https://orcid.org/0000-0002-0675-508X>`_ (michele.mancarella@unimib.it) 
- `Michele Moresco <https://orcid.org/0000-0002-7616-7136>`_ (michele.moresco@unibo.it) 

**Contributors**

- `Matteo Tagliazucchi <https://orcid.org/0000-0002-2889-8997>`_
- Niccolò Passaleva
- `Francesco Iacovelli <https://orcid.org/0000-0002-4875-5862>`_

The code was developed starting from https://github.com/CosmoStatGW/DarkSirensStat (`Finke et al. 2019 <https://orcid.org/0000-0002-2889-8997>`_) and https://github.com/CosmoStatGW/MGCosmoPop (`Mancarella et al. 2021 <https://orcid.org/0000-0002-2889-8997>`_).


Documentation
-------------

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    user_guide/installation
    user_guide/introduction
    user_guide/getting_started
    user_guide/framework
    user_guide/changelog
    user_guide/citing

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/GW170817

.. toctree::
    :maxdepth: 1
    :caption: Python APIs

    api/MCMC
    api/Likelihood
    api/GW
    api/EM 
    api/Bias
    api/Models





.. Changelog
.. ---------

.. .. include:: changelog.rst


.. TO BULD THE DOCS
   python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
   python -m sphinx -b latex -D language=en -d _build/doctrees . _build/latex


.. TO BUILD CHIMERA for pypi
    poetry env use python
    poetry build
    poetry lock
    poetry update
    poetry check
    poetry build
