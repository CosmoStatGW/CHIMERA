.. default-role:: math

.. _framework:

Introduction
============

CHIMERA is a Python package for hierarchical Bayesian inference of gravitational wave population parameters using galaxy catalog data. It extends the framework from `Mandel et al. 2019 <http://doi.org/10.1093/mnras/stz896/>`_ and `Vitale et al. 2022 <http://doi.org/10.1007/978-981-15-4702-7_45-1/>`_.

.. seealso::
    More details of the statistical framework are presented in `Borghi et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024ApJ...964..191B/>`_, for most recent implementations (performance and GPU support) see `Tagliazucchi et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250402034T/abstract>`_.

Framework and Code Structure
----------------------------

CHIMERA evaluates the hyper-likelihood for population parameters `\boldsymbol{\lambda}=\{\boldsymbol{\lambda}_\mathrm{c},\boldsymbol{\lambda}_\mathrm{m},\boldsymbol{\lambda}_\mathrm{z}\}` (cosmology, source mass distribution, rate evolution):

.. math::

    p(\boldsymbol{d}^{\rm GW} | \boldsymbol{\lambda}) \propto \frac{1}{\xi(\boldsymbol{\lambda})^{N_{\rm ev}}} \prod_{i=1}^{N_{\rm ev}} \int \mathcal{K}_{\mathrm{gw},i}(z, \hat{\Omega} | \boldsymbol{\lambda}_\mathrm{c}, \boldsymbol{\lambda}_\mathrm{m}) \,
    p_{\rm gal}(z, \hat{\Omega} | \boldsymbol{\lambda}_{\rm c})\, \frac{\psi(z ; \boldsymbol{\lambda}_{\rm z})}{1+z}\, \mathrm{d}z\, \mathrm{d}\hat{\Omega}

The GW kernel `\mathcal{K}_{\mathrm{gw},i}` is computed via KDE while the selection bias `\xi(\boldsymbol{\lambda})` uses Monte Carlo integration.

**Core modules:**
    * ``likelihood.py`` - Main likelihood computation
    * ``selection_function.py`` - Selection bias calculations  
    * ``data.py`` - GW and electromagnetic data handling
    * ``population/`` - Population models (mass, rate, cosmology)
    * ``catalog/`` - Galaxy catalog processing for redshift priors

**Structure and dependencies:**

.. image:: ../_static/CHIMERA_diagram.drawio.svg
  :width: 600
  :align: center
  :alt: Flowchart of CHIMERA
