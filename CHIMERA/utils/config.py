import os
USE_GPU = os.getenv('CHIMERA_ENABLE_GPU', 'False').lower() == 'true'
USE_x64 = os.getenv('CHIMERA_USE_x64', 'False').lower() == 'true'

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_ftz=true "
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
logger.info(F"Loading `CHIMERA`. GPU acceleration: {USE_GPU}. USE x64: {USE_x64}.")
