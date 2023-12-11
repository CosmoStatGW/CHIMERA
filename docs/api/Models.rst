.. _Models:

Population Models
=================

Currently, CHIMERA includes the following population models:

- **Cosmology** (module): fLCDM, fLCDM modified gravity
- **Mass** (functions): logpdf\_TPL (Truncated Power Law), logpdf\_BPL (Broken Power Law), logpdf\_PLP (Power Law + Peak), logpdf\_PL2P (Power Law + 2 Peaks)
- **Rate** (functions) logphi\_PL (Power Law), logphi\_MD (Madau-like)
- **Spin** (functions) logpdf\_G (Gaussian), logpdf\_U (Uniform)

All the model functions in CHIMERA accept parameters in the form of dictionaries:

.. code-block:: python

    params = {'param1':value1, 'param2':value2, ...}


Cosmology
---------

fLCDM
^^^^^

.. autofunction:: CHIMERA.cosmo.fLCDM.dC
.. autofunction:: CHIMERA.cosmo.fLCDM.dL
.. autofunction:: CHIMERA.cosmo.fLCDM.ddL_dz
.. autofunction:: CHIMERA.cosmo.fLCDM.V
.. autofunction:: CHIMERA.cosmo.fLCDM.dV_dz
.. autofunction:: CHIMERA.cosmo.fLCDM.z_from_dL


Mass
----

.. autofunction:: CHIMERA.astro.mass.logpdf_BPL
.. autofunction:: CHIMERA.astro.mass.logpdf_PLP
.. autofunction:: CHIMERA.astro.mass.logpdf_PL2P


Rate
----

.. autofunction:: CHIMERA.astro.rate.logphi_PL
.. autofunction:: CHIMERA.astro.rate.logphi_MD


Spin
----

.. autofunction:: CHIMERA.astro.spin.logpdf_dummy
.. autofunction:: CHIMERA.astro.spin.logpdf_truncGauss