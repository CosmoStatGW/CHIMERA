.. _Bias:

Selection Bias
==============

.. autoclass:: CHIMERA.Bias.Bias
    :inherited-members:
    :members:



Injections requirements
-----------------------

The injection file located at ``file_inj`` should be in HDF5 format and contain the following fields:

- draw probability:
    - ``log_p_draw_nospin``: log of the reference prior probability used to draw the injections
- detector-frame parameters:
    - ``dL``: luminosity distance
    - ``m1_det``: detector-frame primary mass
    - ``m2_det``: detector-frame secondary mass
- *or* source-frame masses and redshift (then, the detector-frame parameters are computed internally):
    - ``dL``: luminosity distance
    - ``m1src``: source-frame primary mass
    - ``m2src``: source-frame secondary mass
    - ``z``: redshift
