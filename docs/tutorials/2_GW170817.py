# %%
# DIRECTORIES AND IMPORTS

dir_LVK       = "../../data/LVK"
dir_GLADE     = "../../data/GLADE/glade+_GW170817_cutout.hdf5"
dir_GLADE_int = "../../data/GLADE/p_bkg_gauss_smooth_zres5000_smooth30.pkl"
dir_plot       = "out/"

import numpy as np
import healpy as hp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.style.use("plotbelli.style")

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from CHIMERA.DataGW import DataLVK
from CHIMERA.Likelihood import LikeLVK
from CHIMERA.astro.mass import pdf_FLAT, dummy_mass
from CHIMERA.astro.rate import phi_PL, dummy_rate
from CHIMERA.cosmo import fLCDM
from CHIMERA.utils import misc

data_GW_n = "GW170817"
ra4993    = 3.44613079
dec4993   = -0.40812585
z4993     = 0.0091909380476333
errz4993  = 0.0002665372033813657

# %%
# Load GW data


data_GW = DataLVK(dir_LVK).load(data_GW_n)
data_GW["dL"] /= 1000. # Mpc -> Gpc

# %%
# Recipe 

recipe  = {"data_GW_names":     data_GW_n,
           "data_GW":           data_GW,
           "data_GW_smooth":    0.3,
           "data_GAL_dir":      dir_GLADE,
           "data_GAL_int_dir":  dir_GLADE_int,
           "data_GAL_weights":  "L_K",
           "npix_event":        20,
           "sky_conf":          0.90,
           "nside_list":        [512,256,128,64,32,16,8],
           "z_int_sigma":       5,
           "z_int_res":         900,
           "z_det_range":       [0.01, 0.5],
           "z_int_H0_prior":    [30.,220.],
           "Lcut":              0.02,
           "band":              "K",
           "completeness":      None,
}


lambda_mass       = {"ml": 1., "mh": 3.}
lambda_rate       = {"gamma": 2.7}
lambda_cosmo      = {"H0":70, "Om0":0.3}


# %%
# Likelihood initialization

like_dark   = LikeLVK(model_cosmo = fLCDM, model_mass  = dummy_mass, model_rate  = dummy_rate, **recipe)
like_bright = LikeLVK(model_cosmo = fLCDM, model_mass  = dummy_mass, model_rate  = dummy_rate, **recipe)

# Set Abbott+2017 redshift measurement
NGC4993_index = np.argmin(np.sqrt((like_dark.gal.data["ra"] - ra4993)**2 + (like_dark.gal.data["dec"] - dec4993)**2))

like_dark.gal.data["z"][NGC4993_index] = 3017/fLCDM.c_light
like_dark.gal.data["z_err"][NGC4993_index] = 166/fLCDM.c_light
like_dark.p_cat_all, like_dark.ngal_pix = like_dark.gal.precompute(like_dark.nside, like_dark.pix_conf, like_dark.z_grids, 
                                                                   like_dark.data_GW_names, like_dark.data_GAL_weights)


# Set all galaxy weights to 0 except for NGC 4993
like_bright.gal.data['L_K'] = np.zeros(len(like_bright.gal.data["z"]))
like_bright.gal.data['L_K'][NGC4993_index] = 1.
like_bright.data_GAL_weights = "L_K"
like_bright.p_cat_all, like_dark.ngal_pix = like_bright.gal.precompute(like_bright.nside, like_bright.pix_conf, like_bright.z_grids, 
                                                                     like_bright.data_GW_names, like_bright.data_GAL_weights)

likes1D = [like_dark, like_bright]



# %%
# 1D ANALYSIS - K band

par_1D = {"H0":np.linspace(20.,200.,199)}
key    = "H0"
grid   = par_1D[key]

posteriorK = np.zeros((len(likes1D), len(grid)))
# for i, like in enumerate([like1K,like2K,like3K]):
for i, like in enumerate(likes1D):

    for g in tqdm(range(len(grid))):
        lambda_cosmo[key] = grid[g]
        posteriorK[i, g]  = like.compute(lambda_cosmo, lambda_mass, lambda_rate, inspect=True)



# %% 
# Plot 3x1

def pdf(y):
    return y/np.trapz(y, z_grid)


fig, ax   = plt.subplots(1,3, figsize=(10,2.1), dpi=150)
iteration = np.abs(grid - 70).argmin()
event     = 0
N_z_bins  = 10
lw        = 0.7
like      = like_dark

pixels = like.gw.pix_conf[event]
nside  = like.gw.nside[event]
z_grid = like.z_grids[event]
z_lims = z_grid[0], z_grid[-1]
z_bins = np.linspace(*z_lims, N_z_bins)

dgal = like_dark.gal.select_event_region(*z_lims, pixels, nside)
ragal, decgal, zgal = dgal["ra"], dgal["dec"], dgal["z"] 

# Individual galaxies
alpha = misc.remapMinMax(-zgal, a=.2, b=.7)
ax[0].scatter(ragal, decgal, marker='x', c='darkred', s=30, label="Potential hosts", alpha=alpha)

# NGC 4993
ax[0].scatter(ra4993, dec4993, edgecolors='k', facecolors='none', s=55, lw=1,  label="NGC 4993")


# Highlight Healpixels
for jpix in pixels:
    boundaries = hp.boundaries(nside, jpix, step=10)
    b_theta, b_phi = hp.vec2ang(boundaries.T)
    b_theta, b_phi = np.append(b_theta, b_theta[0]), np.append(b_phi, b_phi[0]) # Append the starting point to close the plot
    ax[0].plot(b_phi, np.pi/2 - b_theta, c='silver', lw=0.5, zorder=0, label="Healpix pixels" if jpix==0 else None)

# Plot p(z)_gal for each pixel
[ax[1].plot(z_grid, pdf(like.p_gal_all[event][:,pix]), c="darkred", lw=lw, alpha=0.7, label=r"$p_{\mathrm{gal},k}(z)$" if pix==0 else None) for pix in range(len(pixels))]

# Plot p(z)_gal for the NGC 4993 pixel
pix_NGC = 34316
ax[1].plot(z_grid, pdf(like.p_gal_all[event][:,like.gw.pix_conf[event]==pix_NGC].flatten()), c="k", lw=1.3)

# Plot p(z)_gw for each pixel
[ax[1].plot(z_grid, 2.9*pdf(like.p_gw_all[iteration][:,pix]), c="#2a79f9", lw=lw, alpha=0.7, label=r"$\mathcal{K}_{\mathrm{gw},k}(z|H_0)$" if pix==0 else None) for pix in range(len(pixels))]


# Plot posteriors
post_dark   = posteriorK[0,:]/grid**3
post_bright = posteriorK[1,:]/grid**3 
post_bright = np.load("/home/debian/software/CHIMERA_scripts/scripts/2_OneEvent/data/post_GW170817bright_Abbott17.npy")

xmax, xlow, xup, post_dark_int = misc.get_confidence_HDI(post_dark, grid, kde=0.15, color="#400456", median=True)
lab = r"dark: $H_0={:.0f}^{{+{:.0f}}}_{{-{:.0f}}}$".format(xmax, xup-xmax, xmax-xlow)
print(lab)

xmax, xlow, xup, post_bright_int = misc.get_confidence_HDI(post_bright, grid, kde=0.25,  color="#400456", median=True) 
lab = r"bright: $H_0={:.0f}^{{+{:.0f}}}_{{-{:.0f}}}$".format(xmax, xup-xmax, xmax-xlow)
print(lab)


ax[2].plot(grid, post_dark_int(grid), ls='-', color="#400456", label="Dark siren")
ax[2].plot(grid, post_bright_int(grid), ls='--', color="#400456", label="Bright siren")


ax[0].set_xlim((3.346263556208937, 3.5004548346094804))
ax[0].set_ylim((-0.450730778969222, -0.27028802893635696))
ax[1].set_ylim(0,None)

def rad_to_deg(x, pos):
    return f'{np.degrees(x):.0f}°'
ax[0].xaxis.set_major_formatter(FuncFormatter(rad_to_deg))
ax[0].yaxis.set_major_formatter(FuncFormatter(rad_to_deg))

ax[0].set_xlabel("RA")
ax[0].set_ylabel("Dec")
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$p(z)$")
ax[1].set_xlim(*z_lims)

ax[2].set_xlabel(r"$H_0~~\mathrm{[km\;s^{-1}\;Mpc^{-1}]}$")
ax[2].set_ylabel(r"$p(H_0)~~\mathrm{[km^{-1}\;s\;Mpc]}$")

ax[0].legend(fontsize=9, handlelength=0.5)
ax[1].legend(fontsize=9)
ax[2].legend(fontsize=9)
ax[2].set_ylim(0,0.04)
ax[2].set_xlim(20,200)



x, y = np.loadtxt("/home/debian/software/CHIMERA_scripts/scripts/2_OneEvent/data/post_GW170817_LVKCosmoGWTC3.txt", delimiter=",", unpack=1)
y /= np.trapz(y, x)
if 0: plt.plot(x,y, label="(bright) LVK GWTC-3")


plt.suptitle(r"GW170817 - $L_K>{{{:.2f}}}\,L^\ast_K$".format(recipe["Lcut"]), fontsize=11, y=1.06)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(dir_plot+"fig_GW170817.pdf")




# %%
