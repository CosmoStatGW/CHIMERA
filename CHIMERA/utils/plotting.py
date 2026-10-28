import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex']=True

def plot_pixelization(pix_cat, ev, cmap=None, ax = None, figsize=(6,4), label=None, plot_grid=False):
    """
    Plots the pixelated localization region of a GW event.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Filter both -100 AND NaN values
    valid_mask = (pix_cat.ra_pix[ev] != -100.) & (~np.isnan(pix_cat.ra_pix[ev]))
    ra_pix = pix_cat.ra_pix[ev][valid_mask]
    dec_pix = pix_cat.dec_pix[ev][valid_mask]
    
    # Also filter NaN from pixel indices
    pix_idx_mask = (pix_cat.pixels_opt_nsides[ev] != -100.) & (~np.isnan(pix_cat.pixels_opt_nsides[ev]))
    pix_idx = pix_cat.pixels_opt_nsides[ev][pix_idx_mask]

    grid = np.array([ra_pix, dec_pix])
    samples = np.array([pix_cat.ra[ev], pix_cat.dec[ev]])
    nside = int(pix_cat.opt_nsides[ev])
    pe_pix_idx = pix_cat.pixels_pe_opt_nside[ev]

    n_pix = len(pix_idx)
    if cmap is None:
        # Leaving color=None per draw call (as this function used to do below)
        # does NOT give each pixel's three artists (sample cloud, center dot,
        # boundary) one shared color: ax.scatter and ax.plot pull from
        # separate internal matplotlib color cyclers that each advance once
        # per call, and the two ax.scatter calls below (cloud + dot) share
        # one cycler while ax.plot's boundary uses another, so the three
        # advance at different rates and drift out of sync as i grows.
        # Pick one explicit color per pixel up front instead.
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = [cycle_colors[i % len(cycle_colors)] for i in range(n_pix)]

    for i in range(n_pix):
        # Safely convert to int, skipping any remaining invalid values
        try:
            jpix = int(pix_idx[i])
        except (ValueError, TypeError):
            continue  # Skip invalid pixel indices

        # modulo guards against a user-supplied cmap shorter than n_pix
        color_i = cmap[i % len(cmap)]

        ax.scatter(samples[0][pe_pix_idx==jpix], samples[1][pe_pix_idx==jpix],
                   color=color_i, alpha=0.25, s=50, marker='x')
        ax.scatter(grid[0][i], grid[1][i], s=100, marker='o',
                   color=color_i, edgecolor='black', linewidth=1.)
        boundaries = hp.boundaries(nside, jpix, step=10)
        b_theta, b_phi = hp.vec2ang(boundaries.T)
        b_theta, b_phi = np.append(b_theta, b_theta[0]), np.append(b_phi, b_phi[0])
        b_dec = np.pi/2 - b_theta

        # Break the polyline at the RA=0/2pi branch cut instead of letting
        # matplotlib draw a spurious line straight across the plot whenever
        # a pixel straddles RA=0 (its boundary vertices jump e.g. from
        # phi~0.05 to phi~6.2, which are adjacent on the sky but far apart
        # in raw phi).
        wrap = np.abs(np.diff(b_phi)) > np.pi
        if np.any(wrap):
            idx = np.where(wrap)[0] + 1
            b_phi = np.insert(b_phi, idx, np.nan)
            b_dec = np.insert(b_dec, idx, np.nan)

        ax.plot(b_phi, b_dec, lw=0.85, zorder=0, color=color_i,)

    if plot_grid:
        ax.grid(True, alpha=0.5)
    if label:
        ax.set_xlabel(r"RA [rad]", fontsize=14)
        ax.set_ylabel(r"Dec [rad]", fontsize=14)
    if ax is None:
        return fig
    else:
        return None

def plot_pixelization_mollview(pix_cat, ev, cmap=None, title=None, badcolor='0.92',
                                show_samples=False, sample_color='black', sample_alpha=0.15,
                                sample_size=0.3, fig=None, sub=None, nest=False):
    """Plots the pixelated localization region of a GW event on a full-sky
    Mollweide projection (healpy mollview), as an alternative to the flat
    RA/Dec projection used by `plot_pixelization`.

    The flat projection distorts area/shape away from the equator and has a
    coordinate discontinuity at RA=0/2pi (see `plot_pixelization`'s RA-wrap
    handling), which makes events whose credible region sits near the poles,
    straddles RA=0, or simply covers a large fraction of the sky (as can
    happen for a poorly-localized event -- `pixelize_gw_catalog`'s
    `mean_npixels_event` nside selection is a soft target, not a cap, so an
    event stuck at the coarsest candidate nside can still need far more
    pixels, and area, than the target) hard to read. Mollweide is equal-area
    and has no coordinate discontinuity, so both issues go away. The
    trade-off is the opposite: a *well*-localized event's tiny patch can be
    hard to see on the full sky, in which case `plot_pixelization`'s zoomed
    flat view is usually the more readable one.

    Each selected pixel is colored individually, using the same `cmap`/
    color-cycle convention as `plot_pixelization` (pass the same `cmap` list
    to get matching colors between the two plots for the same event); sky
    outside the event's selected pixels is left in `badcolor`.

    Args:
      pix_cat: struct with `pixels_opt_nsides`/`opt_nsides`/`ra`/`dec` fields
        (as produced by `pixelize_gw_catalog` / `pixelize_or_load_gw_catalog`).
      ev (int): Event index to plot.
      cmap (list, optional): List of colors, one per selected pixel (cycled
        if shorter than the number of pixels). Defaults to the current
        `axes.prop_cycle`, matching `plot_pixelization`'s default.
      title (str, optional): Plot title. Defaults to a generated one with the
        event index, nside and pixel count.
      badcolor (str, optional): Color for sky outside the event's selected
        pixels. Defaults to a light gray.
      show_samples (bool, optional): Overlay the event's raw PE (ra, dec)
        samples as points on top of the pixel map. Defaults to False.
      sample_color, sample_alpha, sample_size: styling for the PE sample
        overlay, used only if `show_samples=True`.
      fig (int or matplotlib.figure.Figure, optional): Figure (or figure
        number) to draw into, per `healpy.visufunc.mollview`'s own `fig`
        argument. If None, healpy creates a new figure.
      sub (int or tuple, optional): healpy `sub` spec (e.g. `(2, 3, 1)`) to
        place this mollview as one panel of a multi-event grid -- call this
        function once per event with the same `fig` and an incrementing
        `sub` index. See `healpy.visufunc.mollview`.
      nest (bool, optional): Healpix ordering scheme of `pix_cat`'s pixel
        indices. Defaults to False (RING), matching `plot_pixelization` and
        CHIMERA's default pipeline configuration.

    Returns:
      matplotlib.figure.Figure: the figure the mollview was drawn into.
    """
    pix_idx_all = np.asarray(pix_cat.pixels_opt_nsides[ev])
    pix_idx_mask = (pix_idx_all != -100.) & (~np.isnan(pix_idx_all))
    pix_idx = pix_idx_all[pix_idx_mask].astype(int)
    nside = int(pix_cat.opt_nsides[ev])
    n_pix = len(pix_idx)

    if title is None:
        title = rf"Event {ev}: {n_pix} pixels @ nside={nside}"

    # healpy's `fig` wants a figure number (or None), not a Figure instance
    fig_num = fig.number if hasattr(fig, 'number') else fig

    m = np.full(hp.nside2npix(nside), hp.UNSEEN)
    for i, jpix in enumerate(pix_idx):
        m[jpix] = i

    if n_pix == 0:
        # nothing to color individually -- just show the (empty) footprint
        hp.mollview(m, fig=fig_num, sub=sub, cmap=mpl.colors.ListedColormap([badcolor]),
                    cbar=False, title=title + " (no valid pixels)",
                    badcolor=badcolor, bgcolor='white', nest=nest)
        hp.graticule()
        return plt.gcf()

    if cmap is None:
        # Same reasoning as plot_pixelization: pick one explicit color per
        # pixel up front rather than relying on per-call auto-cycling.
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = [cycle_colors[i % len(cycle_colors)] for i in range(n_pix)]
    listed_cmap = mpl.colors.ListedColormap([cmap[i % len(cmap)] for i in range(n_pix)])

    # min=0, max=n_pix (not n_pix-1) makes each integer pixel label land
    # exactly in the center of its own color bin, rather than blending
    # between two colors at the normalization boundaries.
    hp.mollview(m, fig=fig_num, sub=sub, cmap=listed_cmap, min=0, max=n_pix,
                cbar=False, title=title, badcolor=badcolor, bgcolor='white', nest=nest)
    hp.graticule()

    if show_samples:
        theta_s = np.pi / 2 - np.asarray(pix_cat.dec[ev])
        phi_s = np.asarray(pix_cat.ra[ev])
        hp.projscatter(theta_s, phi_s, color=sample_color, alpha=sample_alpha, s=sample_size)

    return plt.gcf()

def plot_p_gw3d_pixelated(hyperlike_obj, ev, pixel=None, cmap=None, ax=None, label=True, figsize=(6, 4), **hyper_params):
    """
    Plot p_gw(z, RA, Dec | Λ) for a pixelated GW event.

    Parameters
    ----------
    hyperlike_obj : hyperlikelihood
        Hyperlikelihood object.
    ev : int
        Event index.
    pixel : int, optional
        Pixel index to plot. If None, plot all valid pixels.
    cmap : sequence, optional
        Matplotlib colors.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.
    label : bool
        Add axis labels.
    figsize : tuple
        Figure size if ax is None.
    **hyper_params
        Population hyperparameters passed to population.update().
    """
    created_fig = False
    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
      created_fig = True

    # Build population parameters
    pop_lambdas = hyperlike_obj.population.update(**hyper_params)
  
    # Compute p_gw3d using the configured KDE method
    p_gw3d = hyperlike_obj.p_gw3d(pop_lambdas)
    z_grid = hyperlike_obj.z_grids[ev]
  
  # Determine valid pixels (exclude padded NaN pixels)
    valid_pixels = np.where(np.isfinite(np.asarray(hyperlike_obj.theta_gw_det.gw_loc2d_pdf[ev])))[0]

    if pixel is not None:
      ax.plot(z_grid, p_gw3d[ev, pixel], color=cmap[0] if cmap is not None else None)
    else:
      for i, pix in enumerate(valid_pixels):
        ax.plot(
          z_grid,
          p_gw3d[ev, pix],
          color=cmap[i % len(cmap)]  if cmap is not None else None,
        )

    if label:
      ax.set_xlabel(r"$z$", fontsize=14)
      ax.set_ylabel(
        r"$p_{\rm gw}(z,\mathrm{RA},\mathrm{Dec}\mid\Lambda)$",
        fontsize=14,
      )

    if created_fig:
      return fig

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
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  cosmo_lambdas = hyperlike_obj.population.cosmo.update(**hyper_params)
  pgals = hyperlike_obj.population.gal_cat.p_gal(cosmo_lambdas, hyperlike_obj.z_grids)[ev, :hyperlike_obj.theta_gw_det.neff_pixels[ev], :]

  if pixel is not None:
    ax.plot(hyperlike_obj.z_grids[ev], pgals[pixel], color=cmap[0] if cmap is not None else None)
  else:
    [ax.plot(hyperlike_obj.z_grids[ev], pgals[i], color=cmap[i] if cmap is not None else None) for i in range(hyperlike_obj.theta_gw_det.neff_pixels[ev])]
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
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  pcats = hyperlike_obj.population.gal_cat.p_cat[ev, :hyperlike_obj.theta_gw_det.neff_pixels[ev], :]
  [ax.plot(hyperlike_obj.z_grids[ev], pcats[i], color=cmap[i]  if cmap is not None else None,) for i in range(hyperlike_obj.theta_gw_det.neff_pixels[ev])]
  if label:
    ax.set_xlabel(r"$z$", fontsize=14)
    ax.set_ylabel(r"$p_{\mathrm{cat}}(z, \mathrm{RA}, \mathrm{Dec})$", fontsize=14)
  if ax is None:
    return fig
  else:
    return None
