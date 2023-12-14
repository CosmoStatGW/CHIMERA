.. _installation:

Installation
============
CHIMERA is a pure Python (>3) code with the following dependencies: `numpy <http://www.numpy.org/>`_ (math operations), `scipy <https://www.scipy.org/>`_ (interpolation and integration functions), `JAX <https://jax.readthedocs.io/>`_ (LAX implementation of scipy functions and just-in-time compilation), `healpy <https://healpy.readthedocs.io//>`_ (HEALPix pixelization), `h5py <https://www.h5py.org/>`_ (I/O operations), `matplotlib <https://matplotlib.org/>`_ (plotting routines), which are automatically installed.

Using pip
---------
The code can be quikly installed from `Pypi <https://pypi.org/project/chimera-gw>`_:

.. code-block:: bash

   pip install chimera-gw

From source
-----------
For more flexibility, clone the source repository into your working folder and install it locally:

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .

Test the installation
---------------------
To test the installation, run the following command:

.. code-block:: bash

    python -c "import CHIMERA; print(CHIMERA.__version__)"

Issues
------

If you find issues in the installation process please contact nicola.borghi6@unibo.it. The code has been tested on Linux (Ubuntu 20.04 and Debian GNU/Linux 11).
