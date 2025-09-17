import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex']=True

def plot_pixelization(pix_cat, ev, cmap=None, ax=None, figsize=(6,4), label=None, plot_grid=False):
  """
  Plots the pixelated localization region of a GW event.
  Args:
      pix_cat (dict): Pixelated gw catalog
      ev (int): The event index to plot the data for.
      cmap (Optional[List]): A list of colors to use for the plot. Defaults to the 'tab20' colormap.
      ax (Optional[matplotlib.axes.Axes]): A matplotlib axes object to plot on. If not provided, a new figure and axes are created.
      figsize (Optional[Tuple[int]]): The size of the figure to be created if `ax` is not provided. Defaults to (6, 4).
  Returns:
      If `ax` is provided, returns None. Otherwise, returns the figure created.
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
    # Skip NaN values to avoid the ValueError
    if np.isnan(jpix):
      continue
        
    # Convert jpix to integer to avoid the ValueError
    jpix_int = int(jpix)
    
    ax.scatter(samples[0][pe_pix_idx==jpix_int], samples[1][pe_pix_idx==jpix_int], 
               color=cmap[i], alpha=0.25, s=50, marker='x')
    ax.scatter(grid[0][i], grid[1][i], s=100, marker='o', 
               color=cmap[i], edgecolor='black', linewidth=1.)
    
    # Use the integer pixel index for boundaries
    boundaries = hp.boundaries(nside, jpix_int, step=10)
    b_theta, b_phi = hp.vec2ang(boundaries.T)
    b_theta, b_phi = np.append(b_theta, b_theta[0]), np.append(b_phi, b_phi[0])  # Append the starting point to close the plot
    ax.plot(b_phi, np.pi/2 - b_theta, lw=0.85, zorder=0, color=cmap[i])

  if plot_grid:
    ax.grid(True, alpha=0.5)
  if label:
    ax.set_xlabel(r"RA [rad]", fontsize=14)
    ax.set_ylabel(r"Dec [rad]", fontsize=14)
  
  if ax is None:
    return fig
  else:
    return None

def plot_p_gw3d_pixelated(hyperlike_obj, ev, pixel=None, kind='approximated', cmap=None, ax = None, label=True, figsize=(6,4), **hyper_params):
  """
  Plots the value of `p_gal` in each pixel of a GW event.
  Args:
    hyperlike_obj (object): An instance of a CHIMERA.likelihood.hyperlike
    ev (int): The event index to plot the data for.
    kind (Optional[str]): The type of probability distribution to plot. Can be 'approximated', 'marginalized', or 'full'. Defaults to 'approximate'.
    cmap (Optional[List]): A list of colors to use for the plot. Defaults to the 'tab20' colormap.
    ax (Optional[matplotlib.axes.Axes]): A matplotlib axes object to plot on. If not provided, a new figure and axes are created.
    figsize (Optional[Tuple[int]]): The size of the figure to be created if `ax` is not provided. Defaults to (6, 4).
    **hyper_params: population paramters
  Returns:
    If `ax` is provided, returns None. Otherwise, returns the figure created.
  """
  if cmap is None:
    cmap = mpl.colormaps['tab20'].colors
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  pop_lambdas = hyperlike_obj.population.update(**hyper_params)
  if kind=='approximated':
    pgw3d = hyperlike_obj.p_gw3dapprox(pop_lambdas)
  elif kind=='marginalized':
    pgw3d = hyperlike_obj.p_gw3dmarg(pop_lambdas)
  elif kind=='full':
    pgw3d = hyperlike_obj.p_gw3dfull(pop_lambdas)
  else:
    raise ValueError("`kind` can be 'approximated', 'marginalized', or 'full'")

  if pixel is not None:
    ax.plot(hyperlike_obj.z_grids[ev], pgw3d[ev][pixel], color=cmap[0])
  else:
    [ax.plot(hyperlike_obj.z_grids[ev], pgw3d[ev][i], color=cmap[i]) for i in range(hyperlike_obj.neff_pixels[ev])]
  if label:
    ax.set_xlabel(r"$z$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{K}_{\mathrm{gw}}(z, \mathrm{RA}, \mathrm{Dec} | \mathbf{\Lambda})$", fontsize=14)
  if ax is None:
    return fig
  else:
    return None

def plot_p_gal_pixelated(hyperlike_obj, ev, pixel=None, cmap=None, ax = None, figsize=(6,4), label=True, **hyper_params):
  """
  Plots the value of `p_gal` in each pixel of a GW event.
  Args:
    hyperlike_obj (object): An instance of a CHIMERA.likelihood.hyperlike
    ev (int): The event index to plot the data for.
    kind (Optional[str]): The type of probability distribution to plot. Can be 'approximate', 'marginalized', or 'full'. Defaults to 'approximate'.
    cmap (Optional[List]): A list of colors to use for the plot. Defaults to the 'tab20' colormap.
    ax (Optional[matplotlib.axes.Axes]): A matplotlib axes object to plot on. If not provided, a new figure and axes are created.
    figsize (Optional[Tuple[int]]): The size of the figure to be created if `ax` is not provided. Defaults to (6, 4).
    **hyper_params: population paramters
  Returns:
    If `ax` is provided, returns None. Otherwise, returns the figure created.
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
  if ax is None:
    return fig
  else:
    return None

def plot_p_cat_pixelated(hyperlike_obj, ev, cmap=None, ax = None, label=True, figsize=(6,4)):
  """
  Plots the value of `p_cat` in each pixel of a GW event.
  Args:
    hyperlike_obj (object): An instance of a CHIMERA.likelihood.hyperlike
    ev (int): The event index to plot the data for.
    kind (Optional[str]): The type of probability distribution to plot. Can be 'approximate', 'marginalized', or 'full'. Defaults to 'approximate'.
    cmap (Optional[List]): A list of colors to use for the plot. Defaults to the 'tab20' colormap.
    ax (Optional[matplotlib.axes.Axes]): A matplotlib axes object to plot on. If not provided, a new figure and axes are created.
    figsize (Optional[Tuple[int]]): The size of the figure to be created if `ax` is not provided. Defaults to (6, 4).
  Returns:
    If `ax` is provided, returns None. Otherwise, returns the figure created.
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
  if ax is None:
    return fig
  else:
    return None
