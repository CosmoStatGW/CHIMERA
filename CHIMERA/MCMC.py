#
#   This module handles the full MCMC analysis.
#
#   Copyright (c) 2023 Nicola Borghi <nicola.borghi6@unibo.it>
#
#   All rights reserved. Use of this source code is governed by the license that can be found in the LICENSE file.
#

import numpy as np

__all__ = ['MCMC']



class MCMC(object):

    """Class to handle the full MCMC analysis

    >>> from CHIMERA.MCMC import MCMC
    >>> mcmc    = MCMC(...)
    >>> loglike = mcmc.compute(lambda_cosmo, lambda_mass, lambda_rate)

    or, equivalently:

    >>> loglike = hyperlike.compute_array(lambda_values)


    Args:
        like (CHIMERA.Likelihood): :class:`CHIMERA.Likelihood.Likelihood` object
        bias (CHIMERA.Bias): :class:`CHIMERA.Bias.Bias` object
        keys_cosmo (list): List of cosmological parameter names
        keys_mass  (list): List of mass parameter names
        keys_rate  (list): List of rate parameter names
        priors_cosmo (dict): Dictionary of cosmological priors
        priors_mass  (dict): Dictionary of mass priors
        priors_rate  (dict): Dictionary of rate priors
        prior_kind (str): Kind of prior to use
    """

    def __init__(self, 
                 like, 
                 bias,
                 keys_cosmo = None,
                 keys_mass  = None,
                 keys_rate  = None,
                 priors_cosmo = None,
                 priors_rate  = None,
                 priors_mass  = None,
                 prior_kind = "flat",
                 ):

        self.like = like
        self.bias = bias

        self.keys_cosmo = keys_cosmo
        self.keys_mass  = keys_mass
        self.keys_rate  = keys_rate

        self.priors_cosmo = priors_cosmo
        self.priors_mass  = priors_mass
        self.priors_rate  = priors_rate
        
        if keys_cosmo is not None:
            self.keys = [*keys_cosmo, *keys_mass, *keys_rate]

        if priors_cosmo is not None:
            self.priors = {**priors_cosmo, **priors_mass, **priors_rate}
            self.hyperspace = np.array([self.priors[x] for x in self.keys])

        if prior_kind == "skip":
            self.log_prior = lambda x : 0
        elif prior_kind == "flat":
            self.log_prior = self.log_prior_flat
        else:
            raise NotImplementedError("Prior kind not implemented") 



    def compute(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Compute log-likelihood for a set of parameter values checking the prior

        Args:
            lambda_cosmo (dict): Dictionary of cosmological parameters
            lambda_mass  (dict): Dictionary of mass parameters
            lambda_rate  (dict): Dictionary of rate parameters

        Returns:    
            loglike (float): Log-likelihood
        """

        lp = self.log_prior(self.get_vals(lambda_cosmo, lambda_mass, lambda_rate))

        if not np.isfinite(lp):
            return -np.inf

        llike = np.log(self.like.compute(lambda_cosmo, lambda_mass, lambda_rate))
        lbias = np.log(self.bias.compute(lambda_mass, lambda_cosmo, lambda_rate))

        finite  = np.isfinite(llike)
        llike[~finite] = -1000
        
        return np.sum(llike) - np.sum(finite)*lbias
        # return np.sum(llike) - self.like.Nevents*lbias



    def compute_array(self, lambda_values):
        """Compute log-likelihood for an array of parameter values

        Args:
            lambda_values (array): Array of parameter values

        Returns:
            loglike (float): Log-likelihood
        """

        lambdas = self.get_lambdas(lambda_values)
    
        return self.compute(**lambdas)



    def get_lambdas(self, lambda_values):
        """Retrieve parameter dictionaries from parameter values

        Args:
            lambda_values (array): Array of parameter values

        Returns:
            lambda_cosmo (dict): Dictionary of cosmological parameters
            lambda_mass  (dict): Dictionary of mass parameters
            lambda_rate  (dict): Dictionary of rate parameters
        """

        param_dict   = dict(zip(self.keys, lambda_values))
        lambda_cosmo = {k : param_dict[k] for k in self.keys_cosmo}
        lambda_mass  = {k : param_dict[k] for k in self.keys_mass}
        lambda_rate  = {k : param_dict[k] for k in self.keys_rate}

        return lambda_cosmo, lambda_mass, lambda_rate
    


    def get_vals(self, lambda_cosmo, lambda_mass, lambda_rate):
        """Retrieve parameter values from parameter dictionaries

        Args:
            lambda_cosmo (dict): Dictionary of cosmological parameters
            lambda_mass  (dict): Dictionary of mass parameters
            lambda_rate  (dict): Dictionary of rate parameters

        Returns:
            lambda_values (array): Array of parameter values
        """ 

        param_dict = {**lambda_cosmo, **lambda_mass, **lambda_rate}

        return np.array(param_dict.vals())



    def log_prior_flat(self, lambda_values):
        """Flat prior for all parameters

        Args:
            lambda_values (array): Array of parameter values

        Returns:    
            log_prior (float): Log-prior. 0 if all parameters are within the prior range, -inf otherwise.
        """

        if all(self.hyperspace[:, 0] < lambda_values) and all(lambda_values < self.hyperspace[:, 1]):
            return 0
        return -np.inf