#
#   Plotting functions
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#


import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from CHIMERA.utils import (angles, misc, presets)
from CHIMERA.EM import sum_Gaussians

def legend_unique(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def ax_sky_datashader(ax, ra, dec, size=[100, 60]):
    # Datashader
    Nra, Ndec = (np.array(size)).astype(np.int64)
    ra_lims   = np.min(ra), np.max(ra)
    dec_lims  = np.min(dec), np.max(dec)
    sky_bins  = [np.linspace(*ra_lims, Nra),
                np.linspace(*dec_lims, Ndec)]

    H, xedges,yedges = np.histogram2d(ra,dec,bins=sky_bins)
    cax = (ax.imshow(H.T, extent=[*ra_lims,*dec_lims],
        interpolation='none', origin='lower', aspect=Ndec/Nra, cmap="Blues", alpha=.9))#, norm=colors.LogNorm()))
    
    return cax


def plot_2Dcoring(like,
                  iteration     = 0,
                  event         = 0,
                  do_boundaries = True,
                  do_KDE        = False,
                  do_gal        = True,
                  do_p_gal      = True,
                  do_p_gw       = True,
                  do_p_z        = False,
                  
                  renorm_all    = False,
                  N_z_bins      = 30,
                  KDE_levels    = [0.1,0.8,0.9, 0.99],
                  lw            = 1,
                  lab_p_gal     = "$p_{\mathrm{gal},k}(z)$",
                  lab_p_gw      = "$p_{\mathrm{gw},k}(z)$",
                  lab_p_z       = "$p_{z,k}(z)$",
                  ax            = None, 
                  ):
    
    if ax is None: fig, ax = plt.subplots(1,2,figsize=(13,5))
    
    pixels = like.gw.pix_conf[event]
    nside  = like.gw.nside[event]
    z_grid = like.z_grids[event]
    z_lims = z_grid[0], z_grid[-1]
    z_bins = np.linspace(*z_lims, N_z_bins)

    def pdf(y):
        if renorm_all:
            return y/np.trapz(y, z_grid)
        return y
    
    dgal = like.gal.select_event_region(*z_lims, pixels, nside)
    ragal, decgal, zgal = dgal["ra"], dgal["dec"], dgal["z"] 

    # Individual galaxies
    alpha = misc.remapMinMax(-zgal, a=.2, b=.7)
    ax[0].scatter(ragal, decgal, marker='x', c='darkred', s=30, label="Potential hosts", alpha=alpha)
    # ax[0].scatter(ragal, decgal, marker='x', c='darkred', s=30, label="All galaxies", alpha=0.5)

    if do_boundaries: # Highlight Healpixels
        for jpix in pixels:
            boundaries = hp.boundaries(nside, jpix, step=10)
            b_theta, b_phi = hp.vec2ang(boundaries.T)
            b_theta, b_phi = np.append(b_theta, b_theta[0]), np.append(b_phi, b_phi[0]) # Append the starting point to close the plot
            ax[0].plot(b_phi, np.pi/2 - b_theta, c='silver', lw=0.5, zorder=0, label="Healpix pixels" if jpix==0 else None)

    if do_KDE: # Plot contours (KDE)
            nbins=100
            x1,x2     = ax[0].get_xlim()
            y1,y2     = ax[0].get_ylim()
            xx, yy    = np.mgrid[x1:x2:nbins*1j, y1:y2:nbins*1j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel    = gaussian_kde(np.vstack([like.gw.data["ra"][event], like.gw.data["dec"][event]]), bw_method=0.2)
            f         = np.reshape(kernel(positions).T, xx.shape)
            ax[0].contour(xx, yy, f/np.max(f), colors='k', alpha=0.5, linewidths=.8, levels=KDE_levels)#, levels=[0.00001,0.8,0.9, 0.99999])

    if do_gal: # Overplot histogram of galaxies
        ax[1].hist(zgal, bins=z_bins, color='silver', density=True, label="all galaxies") 

    if do_p_gal: # Plot p(z)_gal for each pixel
        [ax[1].plot(z_grid, pdf(like.p_gal[event][:,pix]), c="darkred", lw=lw, alpha=0.7, label=lab_p_gal if pix==0 else None) for pix in range(len(pixels))]

    if do_p_gw: # Plot p(z)_gw for each pixel
        [ax[1].plot(z_grid, pdf(like.p_gw[iteration][event][:,pix]), c="darkblue", lw=lw, alpha=0.7, label=lab_p_gw if pix==0 else None) for pix in range(len(pixels))]

    if do_p_z: # Plot p(z) for each pixel
        [ax[1].plot(z_grid, pdf(like.p_z[iteration][event][:,pix]), c="teal", lw=lw, alpha=0.7, label=lab_p_z if pix==0 else None) for pix in range(len(pixels))]

    ax[0].set_xlabel("RA")
    ax[0].set_ylabel("DEC")
    ax[1].set_xlabel(r"$z$")
    ax[1].set_ylabel(r"$p(z)~\textit{renormalized}$" if renorm_all else r"$p(z)$")
    ax[1].set_xlim(*z_lims)
    ax[0].legend()
    ax[1].legend()






def plot_completeness(filename_base, GWobj, catalogue, z_lims = [0,1], mask = None, verbose=True):

    print('Plotting completeness...')
    
    if mask is not None:
        if verbose:
            print("Plotting mask mollview...")
        plt.figure(figsize=(20,10))
        hp.mollview(mask)
        plt.savefig(filename_base+"_mask.pdf")
       
    zslices = np.linspace(0,1,20)
    th, ph  = hp.pix2ang(128, np.arange(hp.nside2npix(128)))

    for z in zslices:
        c = catalogue.completeness(th, ph, z)
        plt.figure(figsize=(20,10))
        hp.mollview(c)
        plt.savefig(filename_base+"_complz={:05.2f}.pdf".format(z))

    zmin, zmax = z_lims
    z = np.linspace(zmin, zmax, 1000)

    theta, phi = angles.th_phi_from_ra_dec(np.array(GWobj.ra_conf_finite), 
                                            np.array(GWobj.dec_conf_finite))

    for i in range(GWobj.Nevents):

        plt.figure(figsize=(20,10))
        c = catalogue.completeness(theta[i], phi[i], z) 
        plt.plot(z, c.T, linewidth=4)
        plt.savefig(filename_base+"_compl_central.txt", np.array([z, np.squeeze(c)]))
        plt.close()
        
    plt.close('all')
    print('Done.')






def plot_conf(ax, x, y, perc_val=[0.1,0.5,0.9], key="$H_0$"):
    f    = interp1d(x,y)
    cy   = np.cumsum(y)
    cy  /= cy[-1]
    xl,xm,xu = x[[np.abs(cy - t).argmin() for t in perc_val]]
    ul, ll   = xu-xm, xm-xl
    lab = r"<{:s}>$ = {:.1f}^{{+{:.1f}}}_{{-{:.1f}}}$".format(key, xm,ul,ll)
    ax.plot([xm,xm],[0,f(xm)], c='k', ls='--', label=lab)
    xs = np.mean([xu-xm,xm-xl])
    ax.fill_between(x, f(x), where=((x>=xm-xs) & (x<=xm+xs)), color='k', alpha=0.2)





# def plot_2Dcoring_like(like, event=0, 
#                        lambda_cosmo = presets.lambda_cosmo_mock_v1, 
#                        lambda_mass  = presets.lambda_mass_PLP_mock_v1):
    
#     z_min, z_max = like.z_grids[event][0], like.z_grids[event][-1]
#     z_grid       = np.linspace(z_min, z_max, 2000)
#     pix_conf     = like.gw.pix_conf[event]
#     nside        = like.gw.nside[event]
#     Npix         = len(pix_conf)
    
#     # Cut catalog around the event
#     gal_selected = like.gal.select_event_region(z_min, z_max, pix_conf, nside)
#     N_gal_pix    = np.array([sum(gal_selected["pix"+str(nside)] == p) for p in pix_conf])

#     ########## PLOTTING
#     fig, ax = plt.subplots(1,2,figsize=(20,7))
#     ax[0].set_xlabel("RA"); ax[0].set_ylabel("DEC")
#     ax[1].set_xlabel("z"); ax[1].set_ylabel("N")
#     fig.suptitle("Event {:d}:   Npix = {:d},  Ngal_tot = {:d},  mean(gal/pix) ~ {:.1f} ".format(event,Npix,len(gal_selected),np.mean(N_gal_pix)))

#     # Plot pixels boundaries
#     for jpix in pix_conf:
#         boundaries = hp.boundaries(nside, jpix, step=10)
#         b_theta, b_phi = hp.vec2ang(boundaries.T)
#         ax[0].plot(b_phi, np.pi/2 - b_theta, c='silver', lw=0.5, zorder=0, label="Healpixels" if jpix==pix_conf[0] else None)

#     # Plot galaxies
#     ax[0].scatter(gal_selected["ra"], gal_selected["dec"], marker='x', c='red', s=30, label="all galaxies", 
#                   alpha=misc.remapMinMax(-gal_selected["z"], a=.2, b=.7))
#     ax[1].hist(gal_selected["z"], bins=np.linspace(z_min, z_max, 70), color='k', alpha=0.2, label="all galaxies", density=True)

#     # Plot p_GW
#     like.gw.like(lambda_cosmo, lambda_mass, bw_method=[0.5]*Nevents)

#     args    =  np.array(  [np.tile(z_grid, Npix),
#                            np.hstack([ np.full_like(z_grid, gw.ra_conf_finite[event][pix])  for pix in range(Npix)]),
#                            np.hstack([ np.full_like(z_grid, gw.dec_conf_finite[event][pix]) for pix in range(Npix)])] )
#     p_gw    = gw.eventKDEs[event](args).reshape(Npix,len(z_grid)).T
#     p_gw   /= np.trapz(p_gw,z_grid, axis=0)

#     ax[1].plot(z_grid, p_gw, color="steelblue", alpha=1, label=r"$p_{{GW}}^{{pix}}~(H_0={:.0f})$".format(H0))


#     # Plot p_GAL
#     p_gal = np.vstack([sum_Gaussians(z_grid,
#                                      gal.selectedData["z"][gal.selectedData["pix"+str(NSIDE)] == pix],
#                                      gal.selectedData["z_err"][gal.selectedData["pix"+str(NSIDE)] == pix]) for pix in pix_conf]).T
#     p_gal /= np.trapz(p_gal,z_grid, axis=0)
#     p_gal[~np.isfinite(p_gal)] = 0.  # pb. if falls in an empty pixel

#     ax[1].plot(z_grid, p_gal, label=r"$p_{CAT}^{pix}$", color='red', alpha=0.2)
    
#     legend_unique(ax[0]); legend_unique(ax[1])

#     return fig







def quick_corner(data, parnames=None, smooth=False, **kwargs):
    if data.shape[1]>30:
        print("Transpose data")
        return
    from chainconsumer import ChainConsumer
    c        = ChainConsumer()
    c.add_chain(data, parameters=parnames)
    c.configure(usetex=False, smooth=smooth, kde=False)
    fig      = c.plotter.plot(legend=True, **kwargs)
    return fig.axes
