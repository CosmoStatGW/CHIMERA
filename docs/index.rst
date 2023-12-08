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

.. .. raw:: html

..    <img src="_static/CHIMERA_logoNB2.svg";" style="width:350px; margin:5px;"/>
   
..    <br/>


**CHIMERA** (Combined Hierarchical Inference Model for Electromagnetic and gRavitational-wave Analysis), is a flexible Python code to analyze standard sirens with galaxy catalogs, allowing for a joint fitting of the cosmological and astrophysical population parameters within a Hierarchical Bayesian Inference framework. 

.. The code is designed to be accurate for different scenarios, encompassing bright, dark, and spectral sirens methods, and computationally efficient in view of next-generation GW observatories and galaxy surveys. It uses the LAX-backend implementation and Just In Time (JIT) computation capabilities of JAX. 


.. image:: https://img.shields.io/badge/GitHub-CHIMERA-9e8ed7
    :target: https://github.com/CosmoStatGW/CHIMERA/
.. image:: https://img.shields.io/badge/arXiv-2106.14894-28bceb
    :target: https://arxiv.org/abs/2106.14894
.. image:: https://readthedocs.org/projects/chimera-gw/badge/?version=latest
    :target: https://chimera-gw.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/badge/license-MIT-fb7e21
    :target: https://github.com/CosmoStatGW/CHIMERA/blob/main/LICENSE
.. image:: https://img.shields.io/gitlab/v/release/14528131
    :target: https://gitlab.com/mmoresco/CHIMERA/-/tags

.. raw:: html

    <br/>


Installation
------------

**CHIMERA** can be installed using `Pypi <https://pypi.org/project/chimera-gw>`_:

.. code-block:: bash

   pip install chimera-gw



License & Attribution
---------------------

**CHIMERA** is free software made available under the MIT License. For details see the ``LICENSE``.

If you find this code useful in your research, please cite the following paper (`ADS <https://ui.adsabs.harvard.edu/abs/2022ApJ...927..164B/abstract>`_, `arXiv <https://arxiv.org/abs/2106.14894>`_, `INSPIRE <https://inspirehep.net/literature/1871797>`_):

.. code-block:: tex

    @ARTICLE{Borghi2022a,
        author = {{Borghi}, Nicola and {Moresco}, Michele and {Cimatti}, Andrea and et al.},
         title = "{Toward a Better Understanding of Cosmic Chronometers: Stellar Population Properties of Passive Galaxies at Intermediate Redshift}",
       journal = {ApJ},
          year = 2022,
         month = mar,
        volume = {927},
         pages = {164},
           doi = {10.3847/1538-4357/ac3240},
        eprint = {2106.14894},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJ...927..164B},
    }



The team
--------

Main developers:

- `Nicola Borghi (nicola.borghi6@unibo.it) <https://orcid.org/0000-0002-2889-8997>`_
- `Michele Mancarella (michele.mancarella@unimib.it) <https://orcid.org/0000-0002-0675-508X>`_


Contributors:

- `Michele Moresco (michele.moresco@unibo.it) <https://orcid.org/0000-0002-7616-7136>`_
- `Matteo Tagliazucchi <https://orcid.org/0000-0002-2889-8997>`_
- `Francesco Iacovelli <https://orcid.org/0000-0002-4875-5862>`_
- Niccolò Passaleva


The code was developed starting from https://github.com/CosmoStatGW/DarkSirensStat (`Finke et al. 2019 <https://orcid.org/0000-0002-2889-8997>`_) and https://github.com/CosmoStatGW/MGCosmoPop (`Mancarella et al. 2021 <https://orcid.org/0000-0002-2889-8997>`_).



Documentation
-------------

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    user_guide/installation
    user_guide/introduction
    user_guide/getting_started
    user_guide/framework
    user_guide/citing


.. toctree::
    :maxdepth: 2
    :caption: Python APIs

    api/MCMC
    api/Likelihood
    api/GW
    api/EM 
    api/Bias
    api/Models


.. .. toctree::
..    :maxdepth: 1
..    :caption: Tutorials

..    tutorials/quickstart


.. Changelog
.. ---------

.. .. include:: changelog.rst


.. TO BULD THE DOCS
   python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
   python -m sphinx -b latex -D language=en -d _build/doctrees . _build/latex
