.. _changelog:

Changelog
=========

1.0 (2023-12-12)
++++++++++++++++++

- Initial public release used in Borghi et al. (2024).


2.0 (2025-08-31)
++++++++++++++++++

- Major restructure of the code, improving modularity and performance.
- Implementation of JAX and GPU support for accelerated computation.
- Extension to modified GW propagation models.
- Release presented and used in Tagliazucchi et al. (2025).


2.2 (2026-06-08)
++++++++++++++++++

- Population module restructured into ``population/cosmo``, ``population/mass`` and ``population/rate`` subpackages.
- New likelihood API.
- PLS (power law + spline) mass model, Tagliazucchi et al. (2026).
- Galaxy catalog incompleteness correction and galaxy weighting, Borghi et al. (2026).
- Paired mass model structure alongside the conditioned one.
- New example notebooks.


2.3 (unreleased)
++++++++++++++++++

- New mass models: paired BPL + dip with two/three peaks, PLS with free knots, random-features and
  neural-network densities, joint :math:`(m_1, m_2)` B-spline and random-features models.
- ``mth_completeness``: completeness from an apparent-magnitude-threshold HEALPix map.
- Gradient-safe and float32-friendly mass functions; new ``CHIMERA_USE_x64`` (default ``false``),
  ``CHIMERA_SAFE_WHERE`` and ``CHIMERA_XLA_FTZ`` switches. User-set ``XLA_FLAGS`` are now respected.
- Epanechnikov kernel for the FFT KDE (via KDExpress).
- Fixed a broken merge of ``catalog/completeness.py`` on ``main``.
- Documentation: fixed the Read the Docs build, full API reference for the population subpackages,
  micromamba-based HPC guide for DIFA OPH and CINECA LEONARDO.
- Dependencies: build tools removed from runtime requirements, new ``samplers`` extra, security floors for
  transitive dependencies.
- Python >= 3.12 is now required. License metadata corrected to GPL-3.0-only.
- Releases: wheels and sdists are built, tested and published to PyPI automatically when a ``v*`` tag is pushed.
