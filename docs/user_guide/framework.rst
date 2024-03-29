.. default-role:: math

.. _framework:

Statistical framework
=====================

.. seealso::
    A full description of the statistical framework of CHIMERA is presented in `Borghi et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023arXiv231205302B%2F/>`_. 
    
CHIMERA is based on an extension of the Hierarchical Bayesian inference framework presented in `Mandel et al. 2019 <http://doi.org/10.1093/mnras/stz896/>`_ and `Vitale et al. 2022 <http://doi.org/10.1007/978-981-15-4702-7_45-1/>`_. Consider a population of GW sources, individually described by source-frame parameters `\boldsymbol{\theta}` which globally follow a probability distribution described by hyper-parameters `\boldsymbol{\lambda}`, namely:

- `\boldsymbol{\lambda}=\{\boldsymbol{\lambda}_\mathrm{c},\boldsymbol{\lambda}_\mathrm{m},\boldsymbol{\lambda}_\mathrm{z}\}` population hyper-parameters describing cosmology `(\boldsymbol{\lambda}_\mathrm{c})`, binary mass distribution `(\boldsymbol{\lambda}_\mathrm{m})` and rate evolution `(\boldsymbol{\lambda}_\mathrm{z})`;
- `\boldsymbol{\theta}=\{z,m_1,m_2,\hat{\Omega},\dots\}` GW event source-frame parameters, including redshift `(z)`, primary mass `(m_1)`, secondary mass `(m_2)`, and sky location `(\hat{\Omega})`;
- `\boldsymbol{\theta}^\mathrm{det}=\{d_L,m_1^\mathrm{det},m_2^\mathrm{det},\dots\}` GW event detector-frame parameters, including luminosity distance `(d_L)`, detected primary mass `(m_1)`, detected secondary mass `(m_2)`.


Given a set `\boldsymbol{d}^{\rm GW}=\{\boldsymbol{d}^{\rm GW}_i\}~~\left(i=1,\dots,N_{\rm ev}\right)` of data from independent GW events, the goal of CHIMERA is to evaluate the hyper-likelihood

.. math::

    p(\boldsymbol{d}^{\rm GW} | \boldsymbol{\lambda}) \propto \frac{1}{\xi(\boldsymbol{\lambda})^{N_{\rm ev}}} \prod_{i=1}^{N_{\rm ev}} \int \mathrm{d}z\, \mathrm{d}\hat{\Omega} \, \mathcal{K}_{\mathrm{gw},i}(z, \hat{\Omega} | \boldsymbol{\lambda}_\mathrm{c}, \boldsymbol{\lambda}_\mathrm{m}) \,
    p_{\rm gal}(z, \hat{\Omega} | \boldsymbol{\lambda}_{\rm c})\, \frac{\psi(z ; \boldsymbol{\lambda}_{\rm z})}{1+z}\,, \label{eq:like_full}

where the GW kernel `\mathcal{K}_{\mathrm{gw},i} (z, \hat{\Omega} | \boldsymbol{\lambda}_\mathrm{c}, \boldsymbol{\lambda}_\mathrm{m})` and the selection bias term `\xi(\boldsymbol{\lambda})` are defined as:

.. math::
    \begin{align}
    \mathcal{K}_{\mathrm{gw},i} (z, \hat{\Omega} | \boldsymbol{\lambda}_\mathrm{c}, \boldsymbol{\lambda}_\mathrm{m}) &\equiv 
    \int \mathrm{d}m_1 \mathrm{d}m_2 \,  \frac{ p(z, m_1, m_2, \hat{\Omega} | \boldsymbol{d}^{\rm GW}_i)}{ \pi( d_L ) \pi( m_1^\mathrm{det} ) \pi( m_2^\mathrm{det} ) } \, \frac{p(m_1, m_2 | \boldsymbol{\lambda}_{\rm m})}{\frac{\mathrm{d} d_L}{\mathrm{d} z}(z, \boldsymbol{\lambda}_\mathrm{c}) (1+z)^2}\,,\label{eq:Kgw} \\
    \xi(\boldsymbol{\lambda}) &\equiv \int \mathrm{d} \boldsymbol{\theta}^\mathrm{det} \,  P_{\rm det}(\boldsymbol{\theta}^\mathrm{det})\, \, \frac{p(m_1, m_2 | \boldsymbol{\lambda}_\mathrm{m})}{\frac{\mathrm{d} d_L}{\mathrm{d} z}(z, \boldsymbol{\lambda}_\mathrm{c}) (1+z)^2}  \, p_{\rm gal}(z, \hat{\Omega} | \boldsymbol{\lambda}_{\rm c})\, \frac{\psi(z ; \boldsymbol{\lambda}_{\rm z})}{1+z} \,, \label{eq:selection_effects_xi}
    \end{align}

[TBD]

In CHIMERA, the GW kernel is computed via kernel density estimation (KDE) method, while the selection bias term is computed via Monte Carlo integration. 
