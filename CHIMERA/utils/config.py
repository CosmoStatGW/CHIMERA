"""CHIMERA configuration module.

This module handles the global configuration for the CHIMERA code, including:

- GPU/CPU backend selection via environment variables
- JAX configuration for numerical precision, platform and XLA flags
- Package-wide logging setup

All variables are read **once, at import time**: set them before ``import CHIMERA``.

Environment variables:

- ``CHIMERA_ENABLE_GPU``: ``true`` to run JAX on GPU. Default ``false`` (CPU).
- ``CHIMERA_USE_x64``: ``true`` to enable 64-bit floats in JAX. Default ``false`` (float32).
- ``CHIMERA_SAFE_WHERE``: ``false`` to disable the "double ``jnp.where``" NaN/Inf-gradient
  guards (see :func:`CHIMERA.utils.math.safe_where`). Default ``true``.
- ``CHIMERA_XLA_FTZ``: ``false`` to disable GPU flush-to-zero for denormals. Default ``true``.

Module globals: ``USE_GPU``, ``USE_x64``, ``SAFE_WHERE``, ``XLA_FTZ`` (bools), and the configured
``jax``, ``jnp`` and ``logger`` objects.

Example:
    .. code-block:: bash

        export CHIMERA_ENABLE_GPU=true
        export CHIMERA_USE_x64=true
        python -c "import CHIMERA"
"""

import os


def _env_flag(name, default):
  return os.getenv(name, str(default)).lower() == 'true'


USE_GPU = _env_flag('CHIMERA_ENABLE_GPU', False)
"""bool: GPU acceleration, controlled by ``CHIMERA_ENABLE_GPU``."""

USE_x64 = _env_flag('CHIMERA_USE_x64', False)
"""bool: 64-bit precision in JAX, controlled by ``CHIMERA_USE_x64``."""

# Global switch for the "double jnp.where" NaN/Inf-gradient guard used throughout
# CHIMERA (see CHIMERA.utils.math.safe_where): substitutes a safe dummy value on the
# branch of a jnp.where that would otherwise feed an out-of-domain input into a risky
# op (division, log, pow, ...), which is needed because jax.grad's reverse-mode AD
# through jnp.where still differentiates the untaken branch and can propagate NaN/Inf.
# Set CHIMERA_SAFE_WHERE=False to disable it everywhere in CHIMERA and probe how much
# it actually matters for gradient-based inference (e.g. NUTS).
SAFE_WHERE = _env_flag('CHIMERA_SAFE_WHERE', True)
"""bool: NaN/Inf-gradient guards, controlled by ``CHIMERA_SAFE_WHERE``."""

# Flush-to-zero for denormals on GPU. Some hyperparameters (e.g. cosmological
# params like H0, which shift the FFT-KDE's own frequency grid as they vary)
# can drive float32 computations into the subnormal range across large chunks
# of a parameter sweep; on some GPU/driver/cuFFT combinations (observed: A100)
# arithmetic on denormals falls back to a drastically slower hardware path --
# a >100x slowdown has been observed for H0's gradient specifically. FTZ avoids
# that by rounding those (already numerically negligible) values to zero.
# Set CHIMERA_XLA_FTZ=False if you need strict IEEE-754 denormal handling.
XLA_FTZ = _env_flag('CHIMERA_XLA_FTZ', True)
"""bool: GPU flush-to-zero, controlled by ``CHIMERA_XLA_FTZ``."""

# XLA settings must be in the environment before JAX is imported. User-provided
# values (e.g. exported in a SLURM script) take precedence over CHIMERA defaults.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_xla_defaults = {
  "--xla_gpu_ftz": "true" if XLA_FTZ else "false",
  "--xla_gpu_enable_triton_gemm": "true",
  "--xla_gpu_enable_latency_hiding_scheduler": "true",
}
_xla_user = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = " ".join(
  [_xla_user] + [f"{k}={v}" for k, v in _xla_defaults.items() if k not in _xla_user]
).strip()
XLA_FTZ = "--xla_gpu_ftz=true" in os.environ["XLA_FLAGS"]  # effective value

import jax
jax.config.update("jax_enable_x64", USE_x64)
import jax.numpy as jnp

if USE_GPU:
  jax.config.update('jax_platform_name', 'gpu')
else:
  jax.config.update('jax_platform_name', 'cpu')

import logging

logger = logging.getLogger('CHIMERA')
"""logging.Logger: Package-wide logger instance for CHIMERA."""

logger.setLevel(logging.INFO)
if not logger.handlers:
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
logger.info(f"Loading `CHIMERA`. GPU acceleration: {USE_GPU}. USE x64: {USE_x64}. "
            f"Safe-where guards: {SAFE_WHERE}. XLA FTZ: {XLA_FTZ}.")
