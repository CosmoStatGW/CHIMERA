USE_GPU = False

########################################
#Do not touch what's follow

import jax
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp

if USE_GPU:
  jax.config.update('jax_platform_name', 'gpu')
  import cupy as xp
else:
  jax.config.update('jax_platform_name', 'cpu')
  import numpy as xp

from packaging import version
if version.parse(jax.__version__) < version.parse("0.4.16"):
	trapz = jnp.trapz
elif version.parse(jax.__version__) > version.parse("0.4.16") and version.parse(jax.__version__) < version.parse("0.4.26"):
	trapz = jax.scipy.integrate.trapezoid
else:
	trapz = jnp.trapezoid

import logging
logger = logging.getLogger('CHIMERA')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info(F"Loading `CHIMERA`. GPU acceleration: {USE_GPU}")
