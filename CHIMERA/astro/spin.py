import numpy as np

######################################################
###################################################### Dummy
######################################################


def logpdf_dummy(chi1, chi2, lambda_spin={}):
    """Dummy spin distribution

    Args:
        chi1 (np.ndarray): spin1
        chi2 (np.ndarray): spin2
        lambda_spin (dict): parameters of the spin function with keys: ["muEff", "sigmaEff", "muP", "sigmaP", "rho"]

    Returns:
        np.ndarray: log(pdf)
    """
    return np.zeros(chi1.shape[0])
