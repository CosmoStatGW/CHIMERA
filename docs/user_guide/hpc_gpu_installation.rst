.. _hpc_gpu_installation:

HPC and GPU Installation
========================

This guide installs CHIMERA on HPC systems with `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_,
with optional MPI and GPU support. It has been tested on:

- **DIFA OPH**: University of Bologna's HPC cluster (`website <https://fisica-astronomia.unibo.it/it/dipartimento/servizi-tecnici-e-amministrativi/servizi-tecnici-informatici/servizi-informatici/cluster-oph>`_), CPU only.
- **CINECA LEONARDO**: Booster (GPU, 4x NVIDIA A100 64 GB per node) and DCGP (CPU) partitions
  (`documentation <https://docs.hpc.cineca.it/hpc/leonardo.html>`_).

Overview
--------

The same recipe works on both machines:

1. micromamba provides Python and the compiled scientific stack (``environment.yml``).
2. ``mpi4py`` is built with ``pip`` against the cluster's **own** MPI module, so that it
   uses the cluster interconnect and plays well with SLURM. Do not install ``mpi4py`` or
   ``openmpi`` from conda-forge on a cluster.
3. On GPU nodes, JAX is installed as ``jax[cuda12]``, which ships its own CUDA/cuDNN
   libraries as pip wheels: only the NVIDIA driver is taken from the system.
4. CHIMERA is installed in editable mode from a clone of the repository.

.. note::

    Install everything from a **login node** (compute nodes usually have no internet access),
    then test from a **compute node** (login nodes have no GPUs and must not be used for computations).


Step 0: micromamba (once per cluster)
-------------------------------------

.. code-block:: bash

    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

When prompted for the root prefix, choose a filesystem with enough quota (on LEONARDO, a folder
in your project ``$WORK`` area rather than ``$HOME``). Then reload your shell (``source ~/.bashrc``)
and check ``micromamba --version``.

Step 1: base environment
------------------------

.. code-block:: bash

    git clone https://github.com/CosmoStatGW/CHIMERA
    cd CHIMERA/

    module purge
    micromamba create -y -n chimera -f environment.yml
    micromamba activate chimera


DIFA OPH (CPU)
--------------

2. **Build mpi4py against the cluster OpenMPI**

.. code-block:: bash

    module load astro/gcc/10.3.0
    module load mpi/openmpi/4.1.4
    MPICC=$(which mpicc) pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4.0"

Module names change over time: check them with ``module avail`` and keep the same modules
loaded in your job scripts.

3. **Install JAX (CPU), the MPI extras and CHIMERA**

.. code-block:: bash

    pip install jax
    pip install mpi4jax --no-build-isolation
    pip install -e ".[plots,samplers,mpi]"

4. **Check the installation**

.. code-block:: bash

    python -m mpi4py --mpi-lib-version          # should report the cluster OpenMPI
    python -c "import CHIMERA; print(CHIMERA.__version__)"

Example SLURM script (fill in partition and resources for your allocation):

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=chimera
    #SBATCH --partition=<partition>
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=32
    #SBATCH --cpus-per-task=1
    #SBATCH --time=12:00:00
    #SBATCH --output=chimera-%j.out
    #SBATCH --error=chimera-%j.err

    module purge
    module load astro/gcc/10.3.0 mpi/openmpi/4.1.4
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate chimera

    export OMP_NUM_THREADS=1
    mpirun -np $SLURM_NTASKS python run_chimera.py


CINECA LEONARDO (GPU and CPU)
-----------------------------

2. **Build mpi4py against the cluster OpenMPI**

.. code-block:: bash

    module load openmpi/4.1.6--gcc--12.2.0
    MPICC=$(which mpicc) pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4.0"

3. **Install JAX with CUDA support, the MPI extras and CHIMERA**

.. code-block:: bash

    pip install "jax[cuda12]"
    pip install mpi4jax --no-build-isolation
    pip install -e ".[plots,samplers,mpi]"

Do **not** ``module load cuda`` together with ``jax[cuda12]``: the CUDA libraries are provided by the
pip wheels, and a different CUDA version on ``LD_LIBRARY_PATH`` can shadow them. For CPU-only runs on the
DCGP partition, install ``jax`` instead of ``jax[cuda12]``.

4. **Check the installation on a GPU node**

.. code-block:: bash

    srun -A <account> -p boost_usr_prod --qos=boost_qos_dbg -N 1 --gres=gpu:1 -t 00:10:00 --pty bash
    micromamba activate chimera
    python -c "import jax; print(jax.devices())"          # expect [CudaDevice(id=0)]
    CHIMERA_ENABLE_GPU=true python -c "import CHIMERA; print(CHIMERA.__version__)"

Example SLURM script for one Booster node (4 GPUs):

.. code-block:: bash

    #!/bin/bash
    #SBATCH -A <account>
    #SBATCH -p boost_usr_prod
    #SBATCH --qos=normal
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:4
    #SBATCH --time=04:00:00
    #SBATCH --job-name=chimera-gpu
    #SBATCH --output=chimera-%j.out
    #SBATCH --error=chimera-%j.err

    module purge
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate chimera

    export CHIMERA_ENABLE_GPU=true
    export CHIMERA_USE_x64=false      # set to true for float64 (slower on GPU)
    python run_chimera.py

For CPU runs use ``-p dcgp_usr_prod`` (112 cores per node), drop ``--gres``, ``module load
openmpi/4.1.6--gcc--12.2.0`` and launch with ``srun`` or ``mpirun`` as in the OPH example.
Use the debug QOS (``boost_qos_dbg`` / ``dcgp_qos_dbg``, 30 min) for tests.


Runtime configuration
---------------------

CHIMERA reads these environment variables **once, at import time**:

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Variable
     - Default
     - Effect
   * - ``CHIMERA_ENABLE_GPU``
     - ``false``
     - Run JAX on GPU.
   * - ``CHIMERA_USE_x64``
     - ``false``
     - Enable 64-bit floats in JAX (float32 otherwise).
   * - ``CHIMERA_SAFE_WHERE``
     - ``true``
     - NaN/Inf-gradient guards in ``jnp.where`` branches (needed for gradient-based samplers).
   * - ``CHIMERA_XLA_FTZ``
     - ``true``
     - GPU flush-to-zero for denormals (avoids large slowdowns on A100).

CHIMERA also sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and a few ``XLA_FLAGS`` defaults; values you export
yourself take precedence. To limit GPU memory per process, export ``XLA_PYTHON_CLIENT_MEM_FRACTION``.


Troubleshooting
---------------

- ``python -m mpi4py --mpi-lib-version`` reports Open MPI from conda (not the module): reinstall with
  ``pip uninstall -y mpi4py`` and repeat the ``MPICC=... pip install --no-binary=mpi4py`` step with the
  module loaded, and make sure ``openmpi`` is not installed in the environment (``micromamba list openmpi``).
- ``jax.devices()`` shows only CPU on a GPU node: check that ``jax[cuda12]`` was installed (``pip list | grep jax``),
  that no ``cuda`` module is loaded, and that the job requested GPUs (``--gres=gpu:N``).
- ``mpi4jax`` build fails: it needs ``cython`` and ``wheel`` in the environment (provided by ``environment.yml``)
  and the MPI module loaded, since it is built with ``--no-build-isolation``.
- Keep the same modules loaded at install time and in every job script.
