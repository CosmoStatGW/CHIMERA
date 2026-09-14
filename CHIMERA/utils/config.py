import os
USE_GPU = os.getenv('CHIMERA_ENABLE_GPU', 'False').lower() == 'true'
USE_x64 = os.getenv('CHIMERA_USE_x64', 'False').lower() == 'true'
# Global switch for the "double jnp.where" NaN/Inf-gradient guard used throughout
# CHIMERA (see CHIMERA.utils.math.safe_where): substitutes a safe dummy value on the
# branch of a jnp.where that would otherwise feed an out-of-domain input into a risky
# op (division, log, pow, ...), which is needed because jax.grad's reverse-mode AD
# through jnp.where still differentiates the untaken branch and can propagate NaN/Inf.
# Default True (safe). Set CHIMERA_SAFE_WHERE=False to disable it everywhere in CHIMERA
# and probe how much it actually matters for gradient-based inference (e.g. NUTS).
SAFE_WHERE = os.getenv('CHIMERA_SAFE_WHERE', 'True').lower() == 'true'

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    #"--xla_gpu_ftz=true "
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true"
)
  
import jax
jax.config.update("jax_enable_x64", USE_x64)
if USE_GPU:
  jax.config.update('jax_platform_name', 'gpu')
  #import cupy as xp
else:
  jax.config.update('jax_platform_name', 'cpu')
  #import numpy as xp

import logging
logger = logging.getLogger('CHIMERA')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info(F"Loading `CHIMERA`. GPU acceleration: {USE_GPU}. USE x64: {USE_x64}. Safe-where guards: {SAFE_WHERE}.")
