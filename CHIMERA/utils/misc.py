# 
#   Base functions and parameters
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>, Michele Mancarella <michele.mancarella@unimib.it>               
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import os, h5py
import numpy as np
from scipy.stats import norm, gaussian_kde






###################
# Other functions
###################

def get_Neff_log(log_weights, log_norm):
    """Compute Neff as in Farr+2019 arXiv:1904.10879 with log_weigths

    Args:
        log_weights (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_s2      = np.logaddexp.reduce(2.*log_weights) - 2.*np.log(len(log_weights)) 
    log_sig2    = logdiffexp(log_s2, 2.*log_norm-np.log(len(log_weights)))
   
    return np.exp(2.*log_norm - log_sig2)

def get_Neff(weights, mu, Ndraw=None):
    # arXiv:1904.10879
    if Ndraw is None:
        Ndraw = len(weights)
    s2    = np.sum(weights**2) / Ndraw**2
    sig2  = s2 - mu**2 / Ndraw
    return mu**2 / sig2

def logdiffexp(x, y):
    return x + np.log1p(-np.exp(y-x))


def log1m_exp(x):

    arr_x = 1.0 * np.array(x)
    oob = arr_x < np.log(np.finfo(arr_x.dtype).smallest_normal)
    mask = arr_x > -0.6931472  # appox -log(2)
    more_val = np.log(-np.expm1(arr_x))
    less_val = np.log1p(-np.exp(arr_x))

    return np.where(oob,0.,np.where(mask,more_val,less_val))

def log_diff_exp(a, b):
    mask = a > b
    masktwo = (a == b) & (a < np.inf)
    return np.where(mask, 1.0 * a + log1m_exp(1.0 * b - 1.0 * a), np.where(masktwo,-np.inf,np.nan))


def nanaverage(A,weights,axis):
    return np.nansum(A*weights,axis=axis) /((~np.isnan(A))*weights).sum(axis=axis)
    
def remapMinMax(value, a=0, b=1):
    return (value - value.min()) / (value.max() - value.min()) * (b - a) + a

import time

class Stopwatch:
    """
    Simple stopwatch class
    """
    def __init__(self):
        import time

        self.start_time = time.time()

    def __call__(self, msg=None):
        elapsed_time = time.time() - self.start_time

        if msg is None:
            print("Elapsed time: {:.6f} s".format(elapsed_time))
        else:
            print("{:s}: {:.6f} s".format(msg, elapsed_time))
        self.start_time = time.time()


# Temporary 
def load_data_LVK(events, run, nSamplesUse=None, verbose=False, BBH_only=True, SNR_th = 12, FAR_th = 1):
    import MGCosmoPop
    from MGCosmoPop.dataStructures.O1O2data import O1O2Data
    from MGCosmoPop.dataStructures.O3adata import O3aData
    from MGCosmoPop.dataStructures.O3bdata import O3bData
    dir_data = os.path.join(MGCosmoPop.Globals.dataPath, run)
    events   = {'use':events, 'not_use':None}

    if run == "O1O2":
        data = O1O2Data(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose, BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)
    elif run == "O3a":
        data = O3aData(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose, BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)
    elif run == "O3b":
        data = O3bData(dir_data, events_use=events, nSamplesUse=nSamplesUse, verbose=verbose,BBH_only=BBH_only, SNR_th=SNR_th, FAR_th=FAR_th)

    # dt          = []
    new_data    = {"m1z" : data.m1z,
                   "m2z" : data.m2z,
                   "dL"  : data.dL, # Gpc
                   "ra"  : data.ra,
                   "dec" : data.dec}

    return new_data



def load_data_h5(fname):
    """Generic function to load data from h5 files

    Args:
        fname (str): path to the h5 file

    Returns:
        h5py.File: h5py file
    """
    events={}
    with h5py.File(fname, 'r') as f:
        for key in f.keys(): 
            events[key] = np.array(f[key])
    return events


from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import fmin


def get_confidence_HDI(post, grid, kde=None, interval=0.683, ax=None, color="k", lab=None, median=False, ls='--'):

    if kde is not None:
        N_samples = 10**6
        post /= np.sum(post)
        samples   = np.random.choice(grid, size=N_samples, replace=True, p=post)
        med    = np.median(samples)
        lo, hi = np.percentile(samples, 16), np.percentile(samples, 84)
        post      = gaussian_kde(samples,bw_method=kde).evaluate(grid)
    
    if kde is None and median:
        raise ValueError("kde must be provided to compute the median. Too lazy to write other code.")
    
    # Higher resolution grid
    xx       = np.linspace(grid[0], grid[-1], 10**6)
    post_int = interp1d(grid, post, kind="cubic", bounds_error=False)
    yy       = post_int(xx)

    # Renormalize
    norm     = np.trapz(yy, xx)
    yy       = yy/norm

    # Compute CDF and inverse CDF
    cdf = cumtrapz(yy, xx)
    ppf = interp1d(cdf[cdf > 0.], xx[1:][cdf > 0.], fill_value=0., bounds_error=False)

    def width(low):
        w = ppf(interval + low) - ppf(low)
        return w if w > 0. else np.inf

    # Find the interval
    HDI_low = fmin(width, 1. - interval, disp=False)[0]

    if not median:
        xmax = xx[np.argmax(yy)]
        xlow = ppf(HDI_low)
        xup  = ppf(HDI_low + interval)
    else:
        xmax = med
        xlow = lo
        xup  = hi 
        

    if ax is not None:
        # lab = "$H_0={:.0f}^{{+{:.0f}}}_{{-{:.0f}}}$".format(xmax, xup-xmax, xmax-xlow)
        ax.fill_between(xx, 0, yy, where=(xx >= xlow) & (xx <= xup), color=color, alpha=0.2, linewidth=0.0)
        ax.plot([xmax,xmax], [0, float(post_int(xmax))/norm], color=color, linestyle=ls, label=lab)
        ax.plot(xx, yy, lw=1, color=color, ls="-")
    
    return xmax, xlow, xup, post_int





##################################################################



# def find_maximum_and_interval(xs, ys, desired_area = 0.673):
#     threshold = 0.003
#     startIndex = ys.argmax()
#     maxVal = ys[startIndex]
#     minVal = 0
#     x1 = None
#     x2 = None
#     count = 0
    
#     while x1 is None:
#         mid = (maxVal + minVal) / 2.0
#         count += 1
        
#         try:
#             if count > 50:
#                 raise ValueError("Failed to converge")
                
#             i1 = startIndex - np.where(ys[:startIndex][::-1] < mid)[0][0]
#             i2 = startIndex + np.where(ys[startIndex:] < mid)[0][0]
#             area = np.trapz(ys[i1:i2+1], x=xs[i1:i2+1])
#             deviation = np.abs(area - desired_area)
            
#             if deviation < threshold:
#                 x1 = xs[i1]
#                 x2 = xs[i2]
#             elif area < desired_area:
#                 maxVal = mid
#             elif area > desired_area:
#                 minVal = mid
                
#         except ValueError:
#             return [None, xs[startIndex], None]
    
#     return [x1, xs[startIndex], x2]


