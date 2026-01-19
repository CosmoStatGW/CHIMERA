"""CHIMERA configuration module.

This module handles the global configuration for the CHIMERA code, including:
- GPU/CPU backend selection via environment variables
- JAX configuration for numerical precision and platform
- Package-wide logging setup

Environment Variables:
    CHIMERA_ENABLE_GPU: Set to 'true' to enable GPU acceleration via JAX.
                       Defaults to 'False' (CPU-only mode).

Global Variables:
    USE_GPU (bool): Flag indicating whether GPU acceleration is enabled.
    jax: JAX library configured for the appropriate platform.
    jnp: JAX NumPy array operations.
    logger: Configured logger instance for CHIMERA package.

Example:
    The GPU mode can be enabled by setting the environment variable:
    
    .. code-block:: bash
    
        export CHIMERA_ENABLE_GPU=true
        python -c "import CHIMERA"
"""

import os

USE_GPU = os.getenv('CHIMERA_ENABLE_GPU', 'False').lower() == 'true'
"""bool: Global flag for GPU acceleration, controlled by CHIMERA_ENABLE_GPU environment variable."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

if USE_GPU:
  jax.config.update('jax_platform_name', 'gpu')
else:
  jax.config.update('jax_platform_name', 'cpu')

import logging

logger = logging.getLogger('CHIMERA')
"""logging.Logger: Package-wide logger instance for CHIMERA."""

logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info(F"Loading `CHIMERA`. GPU acceleration: {USE_GPU}")
