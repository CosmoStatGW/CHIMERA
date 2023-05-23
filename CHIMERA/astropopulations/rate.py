
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
    """Madau-Dickinson star formation rate density normalized so that R(z=0) = R0

    Args:
        z (np.ndarray): redshift
        args (dict): model parameters

    Returns:
        np.ndarray: SFRD
    """
    zp1    = z + 1.
    norm0  = 1. + (args["zp"] + 1.)**(-args["gamma"]-args["kappa"])
    
    num    = norm0 * args["R0"] * zp1 ** args["gamma"]
    denom  = 1. + (zp1/(args["zp"] + 1.))**(args["gamma"]+args["kappa"])

    return num/denom





# def phi_MD_norm(z, args):
#     """Madau-Dickinson star formation rate density normalized so that R(z=0) = R0

#     Args:
#         z (np.ndarray): redshift
#         args (dict): model parameters

#     Returns:
#         np.ndarray: SFRD
#     """
#     # Ensure R(z=0) = R0

#     norm0 = 1 + (z+1.)**(-args["gamma"]-args["kappa"])

#     return 1.e-9 * norm0 * args["R0"] * (z+1.)**args["gamma"] / ( 1.+ ((z+1.)/(args["zp"]+1.))**(args["gamma"]+args["kappa"]) )

