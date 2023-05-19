import os
#import sys

###########################
# PATHS
###########################

dirName         = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')




###########################
# CONSTANTS
###########################

clight = 2.99792458* 10**5


# I don't remember what this is...
zRglob = 0.5

# CMB dipole
l_CMB, b_CMB = (263.99, 48.26)
v_CMB = 369

# Solar magnitude in B and K band
MBSun = 5.498
MKSun = 3.27

# Cosmological parameters used in GLADE for z-dL conversion
H0GLADE = 70
Om0GLADE = 0.27

# Parameters of Schechter function in B band in units of 10^10 solar B band
# for h0=0.7
LBstar07 =2.45
phiBstar07   = 5.5 * 1e-3
alphaB07 = -1.07

# Parameters of Schechter function in K band in units of 10^10 solar K band
# for h0=0.7
LKstar07 = 10.56
phiKstar07 = 3.70 * 1e-3
alphaK07 =-1.02



        
###########################
# Parallelization
###########################

import multiprocessing

nCores = max(1,int(multiprocessing.cpu_count()/2)-1)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nCores)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nCores)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


###########################
# Base parameters
###########################

# MASS: lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass 
lambda_mass_PLP_mock_v1   = [0.039, 3.4, 1.1, 4.8, 5.1, 87., 34., 3.6]

# RATE: R0, alphaRedshift , betaRedshift, zp 
lambda_rate_Madau_mock_v1 = [17, 2.7, 3., 2.]

# COSMOLOGY: Planck18 H0, Om0
lambda_cosmo_mock_v1      = [67.66, 0.30966]
