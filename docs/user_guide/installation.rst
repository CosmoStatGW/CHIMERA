.. _installation:

Installation
============

Using pip
---------
The code can be quikly installed from `Pypi <https://pypi.org/project/chimera-gw>`_:

.. code-block:: bash

  pip install chimera-gw

From source
-----------
For more flexibility, clone the source repository into your working folder and install it locally (or append the local folder using `sys`):

.. code-block:: bash

  git clone https://github.com/CosmoStatGW/CHIMERA
  cd CHIMERA/
  pip install -e .

Test the installation
---------------------
To test the installation, run the following command:

.. code-block:: bash

    python -c "import CHIMERA; print(CHIMERA.__version__)"

To install and use the code on HPC facilities with GPU nodes follow, the instructions in "install_hpc.txt".

Issues
------

If you find issues in the installation process please contact nicola.borghi6@unibo.it. The code has been tested on Linux (Ubuntu 20.04 and Debian GNU/Linux 11).
