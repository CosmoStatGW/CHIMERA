Population API
==============

Population Wrapper
------------------

.. automodule:: CHIMERA.population.pop_wrapper

Cosmology
---------

.. automodule:: CHIMERA.population.cosmo.core

.. automodule:: CHIMERA.population.cosmo.distances

.. automodule:: CHIMERA.population.cosmo.volumes

Mass Models
-----------

.. automodule:: CHIMERA.population.mass

.. automodule:: CHIMERA.population.mass.core

.. automodule:: CHIMERA.population.mass.bspline_utils

Conditioned models, :math:`p(m_1)\,p(m_2|m_1)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: CHIMERA.population.mass.conditioned.base
.. automodule:: CHIMERA.population.mass.conditioned.bpl
.. automodule:: CHIMERA.population.mass.conditioned.pl2p
.. automodule:: CHIMERA.population.mass.conditioned.plp
.. automodule:: CHIMERA.population.mass.conditioned.pls

Paired models, :math:`p(m_1)\,p(m_2)\,f(q)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: CHIMERA.population.mass.paired.base
.. automodule:: CHIMERA.population.mass.paired.plp
.. automodule:: CHIMERA.population.mass.paired.pls_free_knots
.. automodule:: CHIMERA.population.mass.paired.bpl_dip_two_peaks
.. automodule:: CHIMERA.population.mass.paired.bpl_dip_three_peaks
.. automodule:: CHIMERA.population.mass.paired.random_features
.. automodule:: CHIMERA.population.mass.paired.nn

Joint models, :math:`p(m_1, m_2)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: CHIMERA.population.mass.joint.base
.. automodule:: CHIMERA.population.mass.joint.base_m1m2
.. automodule:: CHIMERA.population.mass.joint.bspline2d
.. automodule:: CHIMERA.population.mass.joint.random_features

Merger Rate Models
------------------

.. automodule:: CHIMERA.population.rate.core
.. automodule:: CHIMERA.population.rate.madau_dickinson
.. automodule:: CHIMERA.population.rate.power_law
