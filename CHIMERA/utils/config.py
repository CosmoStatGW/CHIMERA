import os
USE_GPU = os.getenv('CHIMERA_ENABLE_GPU', 'False').lower() == 'true'

import jax
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp

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
logger.info(F"Loading `CHIMERA`. GPU acceleration: {USE_GPU}")
