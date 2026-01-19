"""Plotting utilities for CHIMERA gravitational wave analysis.

This module provides visualization functions for GW events, including:
- HEALPix pixelization plots with event localization regions
- 3D GW probability distributions (approximated, marginalized, full)
- Galaxy catalog probabilities per pixel
- Catalog completeness distributions

All plotting functions support custom colormaps, axes, and LaTeX labels.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True

def plot_pixelization(pix_cat, ev, cmap=None, ax=None, figsize=(6, 4), label=None, plot_grid=False):
  """Plots HEALPix pixelization of a GW event's sky localization.
  
  Visualizes the pixelated localization region showing pixel boundaries,
  pixel centers, and posterior samples color-coded by pixel.
  
  Args:
    pix_cat: Pixelated GW catalog object with attributes:
             ra_pix, dec_pix, pixels_opt_nsides, opt_nsides, pixels_pe_opt_nside
    ev (int): Event index to plot
    cmap (list, optional): Color list. Defaults to 'tab20' colormap.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    figsize (tuple, optional): Figure size if creating new figure. Defaults to (6, 4).
    label (bool, optional): Show axis labels. Defaults to None.
    plot_grid (bool, optional): Show grid lines. Defaults to False.
  
  Returns:
    matplotlib.figure.Figure or None: Figure if ax=None, otherwise None
  """
  if cmap is None:
    cmap = mpl.colormaps['tab20'].colors
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  ra_pix = pix_cat.ra_pix[ev][pix_cat.ra_pix[ev] != -100.]
  dec_pix = pix_cat.dec_pix[ev][pix_cat.dec_pix[ev] != -100.]
  pix_idx = pix_cat.pixels_opt_nsides[ev][pix_cat.pixels_opt_nsides[ev] != -100.]

  grid = np.array([ra_pix, dec_pix])
  samples = np.array([pix_cat.ra[ev], pix_cat.dec[ev]])
  nside = int(pix_cat.opt_nsides[ev])
  pe_pix_idx = pix_cat.pixels_pe_opt_nside[ev]

  for i, jpix in enumerate(np.array(pix_idx)):
    if np.isnan(jpix):
      continue
    
    jpix_int = int(jpix)
    
    ax.scatter(samples[0][pe_pix_idx == jpix_int], samples[1][pe_pix_idx == jpix_int], 
               color=cmap[i], alpha=0.25, s=50, marker='x')
    ax.scatter(grid[0][i], grid[1][i], s=100, marker='o', 
               color=cmap[i], edgecolor='black', linewidth=1.)
    
    boundaries = hp.boundaries(nside, jpix_int, step=10)
    b_theta, b_phi = hp.vec2ang(boundaries.T)
    b_theta, b_phi = np.append(b_theta, b_theta[0]), np.append(b_phi, b_phi[0])
    ax.plot(b_phi, np.pi/2 - b_theta, lw=0.85, zorder=0, color=cmap[i])

  if plot_grid:
    ax.grid(True, alpha=0.5)
  if label:
    ax.set_xlabel(r"RA [rad]", fontsize=14)
    ax.set_ylabel(r"Dec [rad]", fontsize=14)
  
  return fig if ax is None else None

def plot_p_gw3d_pixelated(hyperlike_obj, ev, pixel=None, kind='approximated', cmap=None, ax=None, label=True, figsize=(6, 4), **hyper_params):
  """Plots 3D GW probability distribution per pixel.
  
  Visualizes p_gw(z, RA, Dec | Λ) as a function of redshift for each pixel
  in the event's localization region. Supports different computation methods.
  
  Args:
    hyperlike_obj: CHIMERA hyperlikelihood object with population and p_gw3d methods
    ev (int): Event index to plot
    pixel (int, optional): Single pixel index to plot. If None, plots all pixels.
    kind (str, optional): Probability type - 'approximated', 'marginalized', or 'full'.
                         Defaults to 'approximated'.
    cmap (list, optional): Color list. Defaults to 'tab20' colormap.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    label (bool, optional): Show axis labels. Defaults to True.
    figsize (tuple, optional): Figure size if creating new figure. Defaults to (6, 4).
    **hyper_params: Population hyperparameters to pass to population model
  
  Returns:
    matplotlib.figure.Figure or None: Figure if ax=None, otherwise None
  
  Raises:
    ValueError: If kind not in {'approximated', 'marginalized', 'full'}
  """
  if cmap is None:
    cmap = mpl.colormaps['tab20'].colors
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  pop_lambdas = hyperlike_obj.population.update(**hyper_params)
  if kind == 'approximated':
    pgw3d = hyperlike_obj.p_gw3dapprox(pop_lambdas)
  elif kind == 'marginalized':
    pgw3d = hyperlike_obj.p_gw3dmarg(pop_lambdas)
  elif kind == 'full':
    pgw3d = hyperlike_obj.p_gw3dfull(pop_lambdas)
  else:
    raise ValueError("kind must be 'approximated', 'marginalized', or 'full'")

  if pixel is not None:
    ax.plot(hyperlike_obj.z_grids[ev], pgw3d[ev][pixel], color=cmap[0])
  else:
    [ax.plot(hyperlike_obj.z_grids[ev], pgw3d[ev][i], color=cmap[i]) for i in range(hyperlike_obj.neff_pixels[ev])]
  if label:
    ax.set_xlabel(r"$z$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{K}_{\mathrm{gw}}(z, \mathrm{RA}, \mathrm{Dec} | \mathbf{\Lambda})$", fontsize=14)
  return fig if ax is None else None

def plot_p_gal_pixelated(hyperlike_obj, ev, pixel=None, cmap=None, ax=None, figsize=(6, 4), label=True, **hyper_params):
  """Plots galaxy catalog probability distribution per pixel.
  
  Visualizes p_gal(z, RA, Dec | λ) as a function of redshift for each pixel,
  computed from the galaxy catalog and cosmological model.
  
  Args:
    hyperlike_obj: CHIMERA hyperlikelihood object with galcat_obj and z_grids
    ev (int): Event index to plot
    pixel (int, optional): Single pixel index to plot. If None, plots all pixels.
    cmap (list, optional): Color list. Defaults to 'tab20' colormap.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    figsize (tuple, optional): Figure size if creating new figure. Defaults to (6, 4).
    label (bool, optional): Show axis labels. Defaults to True.
    **hyper_params: Cosmological parameters to pass to galaxy catalog model
  
  Returns:
    matplotlib.figure.Figure or None: Figure if ax=None, otherwise None
  """
  if cmap is None:
    cmap = mpl.colormaps['tab20'].colors
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  cosmo_lambdas = hyperlike_obj.population.cosmo.update(**hyper_params)
  pgals = hyperlike_obj.galcat_obj.compute_pgal(cosmo_lambdas, hyperlike_obj.z_grids)[ev]
  pgals = pgals[pgals != -100.].reshape(hyperlike_obj.neff_pixels[ev], hyperlike_obj.z_int_res)

  if pixel is not None:
    ax.plot(hyperlike_obj.z_grids[ev], pgals[pixel], color=cmap[0])
  else:
    [ax.plot(hyperlike_obj.z_grids[ev], pgals[i], color=cmap[i]) for i in range(hyperlike_obj.neff_pixels[ev])]
  if label:
    ax.set_xlabel(r"$z$", fontsize=14)
    ax.set_ylabel(r"$p_{\mathrm{gal}}(z, \mathrm{RA}, \mathrm{Dec} | \lambda)$", fontsize=14)
  return fig if ax is None else None

def plot_p_cat_pixelated(hyperlike_obj, ev, cmap=None, ax=None, label=True, figsize=(6, 4)):
  """Plots catalog completeness probability distribution per pixel.
  
  Visualizes p_cat(z, RA, Dec) as a function of redshift for each pixel,
  representing the catalog completeness as a function of sky position and redshift.
  
  Args:
    hyperlike_obj: CHIMERA hyperlikelihood object with galcat_obj.pcat
    ev (int): Event index to plot
    cmap (list, optional): Color list. Defaults to 'tab20' colormap.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
    label (bool, optional): Show axis labels. Defaults to True.
    figsize (tuple, optional): Figure size if creating new figure. Defaults to (6, 4).
  
  Returns:
    matplotlib.figure.Figure or None: Figure if ax=None, otherwise None
  """
  if cmap is None:
    cmap = mpl.colormaps['tab20'].colors
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  pcats = hyperlike_obj.galcat_obj.pcat[ev]
  pcats = pcats[pcats != -100.].reshape(hyperlike_obj.neff_pixels[ev], hyperlike_obj.z_int_res)
  [ax.plot(hyperlike_obj.z_grids[ev], pcats[i], color=cmap[i]) for i in range(hyperlike_obj.neff_pixels[ev])]
  if label:
    ax.set_xlabel(r"$z$", fontsize=14)
    ax.set_ylabel(r"$p_{\mathrm{cat}}(z, \mathrm{RA}, \mathrm{Dec})$", fontsize=14)
  return fig if ax is None else None
