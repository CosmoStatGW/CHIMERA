
def phi_MD(z, args):
    """Madau-Dickinson star formation rate density

    Args:
        z (np.ndarray): redshift
        args (dict): model parameters

    Returns:
        np.ndarray: SFRD
    """

    return 1.e-9 * args["R0"] * (z+1.)**args["gamma"] / ( 1.+ ((z+1.)/(args["zp"]+1.))**(args["gamma"]+args["kappa"]) )





def phi_MD_norm(z, args):
    """Madau-Dickinson star formation rate density normalized

    Args:
        z (np.ndarray): redshift
        args (dict): model parameters

    Returns:
        np.ndarray: SFRD
    """
    # Ensiure R(z=0) = R0

    norm = 1 + (z+1.)**(-args["gamma"]-args["kappa"])

    return 1.e-9 * norm * args["R0"] * (z+1.)**args["gamma"] / ( 1.+ ((z+1.)/(args["zp"]+1.))**(args["gamma"]+args["kappa"]) )