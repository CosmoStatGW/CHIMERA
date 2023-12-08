.. _installation:

Installation
============
CHIMERA is a pure Python (>3) code with the following dependencies: `numpy <http://www.numpy.org/>`_ (math operations), `scipy <https://www.scipy.org/>`_ (interpolation and integration functions), `JAX <https://jax.readthedocs.io/>`_ (LAX implementation of scipy functions and just-in-time compilation), `healpy <https://healpy.readthedocs.io//>`_ (HEALPix pixelization), `emcee <https://emcee.readthedocs.io/>`_ (MCMC affine invariant sampler), `h5py <https://www.h5py.org/>`_ (I/O operations), `matplotlib <https://matplotlib.org/>`_ (plotting routines), which are automatically installed.



Using pip (available soon)
--------------------------
CHIMERA can be easily and quicly installed just by using `Pypi <https://pypi.org/project/pylick>`_:

.. code-block:: bash

   pip install chimera


From source
-----------
If you want to modify the code you can clone the source repository hosted on on GitLab

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA.git
    cd CHIMERA
    python -m pip install -e .


Issues
------

If you find issues in the installation process please contact nicola.borghi6@unibo.it. The code has been tested on Linux (Ubuntu 20.04 and Debian GNU/Linux 11).