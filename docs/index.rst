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

**CHIMERA** is a hierarchical Bayesian pipeline for standard siren cosmology with gravitational wave data alone or in combination with galaxy catalogs. 

The latest version delivers 10-1000× speedup through JAX and GPU acceleration, enabling the analysis of thousands of events for next-generation gravitational wave observatories.

.. image:: https://img.shields.io/badge/GitHub-CHIMERA-9e8ed7
    :target: https://github.com/CosmoStatGW/CHIMERA/
    :alt: GitHub
.. image:: https://img.shields.io/badge/arXiv-2106.14894-5185C4
    :target: https://arxiv.org/abs/2106.14894
    :alt: arXiv
.. image:: https://img.shields.io/badge/arXiv-2504.02034-45bbd5
    :target: https://arxiv.org/abs/2504.02034
    :alt: arXiv
.. image:: https://readthedocs.org/projects/chimera-gw/badge/?version=latest
    :target: https://chimera-gw.readthedocs.io/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/license-GPLv3-fb7e21
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

You can also run **CHIMERA** on GPU, but you have to install JAX with GPU support as explained in the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.



Citation
--------

If you find this code useful in your research, please cite the following papers: 

- Borghi et al. 2024 (`ADS <https://ui.adsabs.harvard.edu/abs/2024ApJ...964..191B/abstract>`_, `arXiv <https://arxiv.org/abs/2312.05302>`_, `INSPIRE <https://inspirehep.net/literature/2734729>`_)
- Tagliazucchi et al. 2025 (`ADS <https://ui.adsabs.harvard.edu/abs/2023arXiv231205302B/>`_, `arXiv <https://arxiv.org/abs/2312.05302>`_, `INSPIRE <https://inspirehep.net/literature/2907634>`_):

BibTeX from INSPIRE:

.. code-block:: tex

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

    @article{Tagliazucchi:2025ofb,
        author = "Tagliazucchi, Matteo and Moresco, Michele and Borghi, Nicola and Fiebig, Manfred",
        title = "{Accelerating the Standard Siren Method: Improved Constraints on Modified Gravitational Wave Propagation with Future Data}",
        eprint = "2504.02034",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        month = "4",
        year = "2025"
    }


Contributions
-------------

CHIMERA is actively maintained at the **University of Bologna** by: `Nicola Borghi (nicola.borghi6@unibo.it) <https://orcid.org/0000-0002-2889-8997>`_, `Matteo Tagliazucchi (matteo.tagliazucchi2@unibo.it) <https://orcid.org/0009-0003-8886-3184>`_, and `Michele Moresco (michele.moresco@unibo.it) <https://orcid.org/0000-0002-7616-7136>`_.

Michele Mancarella, Francesco Iacovelli and Michele Maggiore contributed to the development of the first version of the code.

The development of CHIMERA has also been supported from the work of Master's thesis students at the University of Bologna (in reverse chronological order):

- *Giulia Cuomo* (2025, `thesis <https://amslaurea.unibo.it/id/eprint/35185/>`_): incompleteness function and application to GWTC-3 data
- *Manfred Fiebig* (2025, `thesis <https://amslaurea.unibo.it/id/eprint/34082/>`_): modified GW propagation function and forecasts for LVK-O5
- *Niccolò Passaleva* (2024, `thesis <https://amslaurea.unibo.it/id/eprint/30896/>`_): mass function models and inference with nested sampling
- *Matteo Schulz* (2024, `thesis <https://amslaurea.unibo.it/id/eprint/30896/>`_): mass function models and cosmological analysis


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
