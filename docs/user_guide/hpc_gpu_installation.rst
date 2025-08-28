.. _hpc_gpu_installation:

HPC and GPU Installation
========================

This guide provides instructions for installing CHIMERA on High Performance Computing (HPC) systems also enabling GPU support.

General Installation Steps
---------------------------

The typical installation process on HPC systems involves:

1. Loading appropriate Python and compiler modules
2. Creating and activating a virtual environment
3. Installing MPI4PY with proper MPI support
4. Installing JAX with GPU support (if needed)
5. Installing remaining Python dependencies
6. Installing CHIMERA

Below are detailed instructions for two specific systems where this process has been tested:

- **DIFA OPH**: University of Bologna's HPC cluster (`website <https://fisica-astronomia.unibo.it/it/dipartimento/servizi-tecnici-e-amministrativi/servizi-tecnici-informatici/servizi-informatici/cluster-oph>`_)
- **LEONARDO**: One of CINECA's supercomputing facilities (`website <https://www.hpc.cineca.it/user-support/documentation/>`_)


DIFA OPH
--------

1. **Load Python 3.10**

.. code-block:: bash

    module purge
    module load astro/python/3.10.0

2. **Create and activate virtual environment**

.. code-block:: bash

    mkdir virtual_env
    python -m venv "virtual_env/chimera-gw"
    source virtual_env/chimera-gw/bin/activate

3. **Install MPI4PY with OpenMPI support**

.. code-block:: bash

    module load astro/gcc/10.3.0
    module load mpi/openmpi/4.1.4
    MPICC=$(which mpicc) pip install --no-cache-dir mpi4py>=4.0

Verify the installation:

.. code-block:: bash

    python -m mpi4py --mpi-lib-version
    ldd $(python -m mpi4py --prefix)/MPI.*.so

4. **Install required packages**

.. code-block:: bash

    pip install wheel cython numpy packaging tqdm jax
    pip install mpi4jax --no-build-isolation
    pip install equinox plum-dispatch emcee h5py healpy
    pip install schwimmbad numba dill interpax matplotlib

5. **Install CHIMERA**

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .


CINECA/LEONARDO
---------------

1. **Load Python 3.10**

.. code-block:: bash

    module load python/3.10.8--gcc--8.5.0

2. **Create and activate virtual environment**

.. code-block:: bash

    mkdir virtual_env
    python -m venv "virtual_env/chimera-gw"
    source virtual_env/chimera-gw/bin/activate

3. **Load required modules and install MPI4PY**

.. code-block:: bash

    module load openmpi/4.1.6--gcc--12.2.0
    module load cuda/12.2
    MPICC=$(which mpicc) pip install --no-cache-dir mpi4py>=4.0

4. **Install required packages with CUDA support**

.. code-block:: bash

    pip install wheel cython numpy packaging tqdm
    pip install "jax[cuda12]"
    pip install mpi4jax --no-build-isolation
    pip install equinox plum-dispatch emcee healpy h5py
    pip install schwimmbad numba dill interpax matplotlib

Alternative using micromamba:

.. code-block:: bash

    micromamba install -c conda-forge gcc_linux-64 gxx_linux-64
    # Then install remaining packages with pip as above

5. **Install CHIMERA**

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/
    pip install -e .

Notes
-----

- For systems requiring SLURM job submission, you may need to install MPI4PY within a SLURM job
- Ensure GPU drivers and CUDA libraries are properly configured on your system
- The installation order is important: install MPI4PY before MPI4JAX to avoid conflicts
- Test your installation by importing CHIMERA and running a simple example