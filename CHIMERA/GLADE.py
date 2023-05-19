#
#    Copyright (c) 2023 Michele Mancarella <michele.mancarella@unige.ch>,
#                       Nicola Borghi <nicola.borghi6@unibo.it>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.

import pandas as pd
import healpy as hp
import numpy as np
import h5py

import os, os.path
from os.path import isfile

import logging
log = logging.getLogger(__name__)

####
# This module contains a class to handle the GLADE catalogue
####

import DSglobals as glob 
from Galaxies import GalCat

from cosmologies import fLCDM
lambda_cosmo_GLADE = {"H0":glob.H0GLADE, "Om0":glob.Om0GLADE}

def TotLum(x, MSun): 
    return 10**(-0.4* (x+25-MSun))


from scipy.stats import gaussian_kde, norm

def sum_of_gaussians(z, mu, sigma, weights=None):
    # Since does not depend on H0 can be generated in advance

    z = np.array(z)[:, np.newaxis]
    mu = np.array(mu)
    sigma = np.array(sigma)

    if weights is None:
        weights = np.ones(len(mu))

    dVdz     = fLCDM.dV_dz(z, {"H0":70,"Om0":0.3})
    gauss    = norm.pdf(z, mu, sigma)
    integral = np.trapz(dVdz*gauss, z, axis=0)

    return np.sum(weights * dVdz * gauss/integral, axis=1)


class GLADE(GalCat):

    def __init__(self, foldername, compl, useDirac, #finalData = None,
                 subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS'], 
                 subsurveysExcl = [], 
                 catalogname    = "GLADE_2.4.txt",
                 groupname      = "Galaxy_Group_Catalogue.csv",
                 **kwargs):

        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        #self._finalData = finalData
        self.catalogname     = catalogname
        self.groupname      = groupname

        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)

        GalCat.__init__(self, foldername, compl, useDirac, **kwargs)


    def load(self, band             = None,
                   band_weight      = None,
                   Lcut             = 0,
                   zMax             = 100,
                   z_flag           = None,
                   drop_z_uncorr    = False,
                   get_cosmo_z      = True, #cosmo=None, 
                   pos_z_cosmo      = True,
                   drop_no_dist     = False,
                   group_correct    = True,
                   which_z_correct  = 'z_cosmo',
                   CMB_correct      = True,
                   which_z          = 'z_cosmo_CMB',
                   galPosterior     = True,
                   err_vals         = 'GLADE',
                   drop_HyperLeda2  = True, 
                   colnames_final   = ['theta','phi','z','z_err','z_lower','z_lowerbound',
                                       'z_upper','z_upperbound','w','completenessGoal']):

        if band_weight is not None:
            assert band_weight==band

        loaded = False
        computePosterior = False
        posteriorglade = os.path.join(self._path, 'posteriorglade.csv')

        if galPosterior:
            if isfile(posteriorglade):
                log.info("Directly loading final data ")
                df     = pd.read_csv(os.path.join(self._path, 'posteriorglade.csv'))
                loaded = True
            else:
                computePosterior = True
                loaded           = False

        if not loaded:
            filepath_GLADE  = os.path.join(self._path, self.catalogname)
            filepath_groups = os.path.join(self._path, self.groupname)


            # ------ LOAD CATALOGUE

            log.info("\nLoading GLADE from {:s}".format(filepath_GLADE))

            df = pd.read_csv(filepath_GLADE, sep=" ", header=None, low_memory=False)
            colnames = ['PGC', 'GWGC_name', 'HYPERLEDA_name', 'TWOMASS_name', 'SDSS_name', 'flag1', 'RA', 'dec',
                        'dL', 'dL_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
                        'flag2', 'flag3']
            df.columns = colnames
            
             
            # ------  SELECT SUBSURVEYS
            
            # if object is named in a survey (not NaN), it is present in a survey
            for survey in ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS']:
                # new column named suvey that will be True or False
                # if object is in or not
                
                #copy the name column
                df.loc[:,survey] = df[survey + '_name']
                
                # NaN to False
                df[survey] = df[survey].fillna(False)
                # not False (i.e. a name) to True
                df.loc[df[survey] != False, survey] = True

            # include only objects that are contained in at least one of the surveys listed in _subsurveysIncl (is non-empty list)
            mask = df[self._subsurveysIncl[0]] == True
            for incl in self._subsurveysIncl[1:]:
                mask = mask | (df[incl] == True)
            df = df.loc[mask]
            
            # explicitely exclude objects if they are contained in some survey(s) in _subsurveysExcl
            # (may be necessary if an object is in multiple surveys)
            for excl in self._subsurveysExcl:
                df = df.loc[df[excl] == False]   


            # ------ Add theta, phi for healpix in radians

            df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
            df.loc[:,"phi"]   = df.RA*np.pi/180
      
            or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
            log.info('N. of objects: %s' %or_dim)


            # ------ Select parts of the catalogue
                    
            if z_flag is not None:
                df = df[df.flag2 != z_flag ]
                log.info("Dropping galaxies with flag2={:s}".format(z_flag))
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
                
            if drop_z_uncorr:
                df = df[df['flag3']==1]
                log.info("Keeping only galaxies with redshift corrected for peculiar velocities")
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
              
            if drop_no_dist:
                df = df[df.dL.notna()==True]
                log.info("Keeping only galaxies with known value of luminosity distance")
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
            
            if drop_HyperLeda2:
                df = df.drop(df[(df['HYPERLEDA_name'].isna()) & (df['flag2']==2)].index)
                log.info("Dropping galaxies with HyperLeda name=null and flag2=2")
                log.info(" > kept {:d} points ({:.00%} of total)".format(df.shape[0], df.shape[0]/or_dim) )


            # ------ Add z corrections

            if get_cosmo_z:

                z_max   = df[df.dL.notna()]['z'].max() + 0.01
                z_min   = max(0, df[df.dL.notna()]['z'].min() - 1e-05)
                log.info("Computing z(dL; H0={:.1f} km/Mpc/s, Om0={:.2f}) ".format(glob.H0GLADE, glob.Om0GLADE))
                log.info(" > interpolating grid between z_min={:f}, z_max={:f}".format(z_min, z_max))
                z_grid  = np.linspace(z_min, z_max, 200000)
                dL_grid = fLCDM.dL(z_grid, lambda_cosmo_GLADE)                

                if not drop_no_dist:
                    dLvals       = df[df.dL.notna()]['dL']
                    zvals        = df[df.dL.isna()]['z']    # not used
                    z_cosmo_vals = np.where(df.dL.notna(), np.interp(df.dL, dL_grid, z_grid), df.z)

                    log.info(" > {:d} points have valid entry for dist".format(dLvals.shape[0]))
                    log.info(" > {:d} points have null entry for dist, correcting original z".format(zvals.shape[0]))

                else:
                    z_cosmo_vals = np.interp(df.dL , dL_grid, z_grid)

                df.loc[:,'z_cosmo'] = z_cosmo_vals

                if not CMB_correct and not group_correct and pos_z_cosmo:
                    log.info("Keeping only galaxies with positive cosmological redshift")
                    df = df[df.z_cosmo >= 0]
                    log.info(" > kept {:s} points ({0:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )

            if group_correct:

                if not get_cosmo_z:
                    raise ValueError("To apply group corrections, compute cosmological redshift first")
                
                log.info("Loading galaxy group catalogue from {:s}".format(filepath_groups))
                df_groups = pd.read_csv(filepath_groups)
                self.group_correction(df, df_groups, which_z=which_z_correct)

            if CMB_correct:

                if not get_cosmo_z:
                    raise ValueError("To apply CMB corrections, compute cosmological redshift first")

                self.CMB_correction(df, which_z=which_z_correct)
                if pos_z_cosmo:
                    log.info("Keeping only galaxies with positive redshift in the colums {:s}".format(which_z))
                    df = df[df[which_z]>= 0]
                    log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
            
            if which_z!='z':

                log.info("Renaming column '{:s}' to 'z'. This will be used in the analysis".format(which_z))
                df = df.drop(columns='z')
                df.rename(columns={which_z:'z'}, inplace=True)

            # From now on, the relevant column for redshift, including all corrections, will be 'z'

            # ------ Potentially drop large z

            df = df[df.z < zMax]

            # ------ Add z errors

            if err_vals is not None:
                log.info('Adding errors on z with %s values' %err_vals)
                if err_vals=='GLADE':
                    scales = np.where(df['flag2'].values==3, 1.5*1e-04, 1.5*1e-02)
                elif err_vals=='const_perc':
                    scales = np.where(df['flag2'].values==3, df.z/100, df.z/10)
                elif err_vals=='const':
                    scales = np.full(df.shape[0], 200/glob.clight)
                else:
                    raise ValueError('Enter valid choice for err_vals (GLADE, const, const_perc). Got %s' %err_vals)

                # restrict error to <=z itself, otherwise for z very close to 0 input is
                # infeasible for keelin distributions, which would break things silently
                df.loc[:, 'z_err']                          = np.minimum(scales, df.z.to_numpy())
                df.loc[:, 'z_lowerbound']                   = df.z - 3*df.z_err
                df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
                df.loc[:, 'z_lower']                        = df.z - df.z_err
                df.loc[df.z_lower < 0.5*df.z, 'z_lower']    = 0.5*df.z
                df.loc[:, 'z_upper']                        = df.z + df.z_err
                df.loc[:, 'z_upperbound']                   = df.z + 3*df.z_err

                # ------ Estimate galaxy posteriors with contant-in-comoving prior

                if computePosterior:
                    self.include_vol_prior(df)

        # ------ End if not use precomputed table
        #        Always be able to still chose the weighting and cut.

        if band=='B' or band_weight=='B':
            add_B_lum = True
            add_K_lum = False
        elif band=='K' or band_weight=='B':
            add_B_lum = False
            add_K_lum = True
        else:
            add_B_lum = False
            add_K_lum = False

        # ------ Add B luminosity

        if add_B_lum:
            log.info('Computing total luminosity in B band')
            
            my_dist                 = fLCDM.dL(df.z.values, lambda_cosmo_GLADE)     
            df.loc[:,"B_Abs_corr"]  = df.B_Abs - 5*np.log10(my_dist) + 5*np.log10(df.dL.values)
            BLum                    = df.B_Abs_corr.apply(lambda x: TotLum(x, glob.MBSun))
            df.loc[:,"B_Lum"]       = BLum
            df                      = df.drop(columns='B_Abs')
            # df = df.drop(columns='B') don't assume it's here
        
        # ------ Add K luminosity
        
        if add_K_lum:
            log.info('Computing total luminosity in K band')

            my_dist                 = fLCDM.dL(df.z.values, lambda_cosmo_GLADE)     
            df.loc[:,"K_Abs"]       = df.K-5 * np.log10(my_dist) - 25
            KLum                    = df.K_Abs.apply(lambda x: TotLum(x, glob.MKSun))
            df.loc[:,"K_Lum"]       = KLum
            df                      = df.drop(columns='K')
            # df = df.drop(columns='K_Abs') don't assume it's here
        
        
        # ------ Apply cut in luminosity
        
        if band is not None:
            col_name  = band+'_Lum'
            if band=='B':
                Lstar = glob.LBstar07
            elif band=='K':
                Lstar = glob.LKstar07
            
            L_th = Lcut*Lstar
            log.info('Applying cut in luminosity in %s-band ' %band)
            log.info(' > L_* in %s band is L_*=%s' %(band, np.round(Lstar,5)))
            log.info(' > selecting galaxies with %s>%sL_* = %s' %(col_name, Lcut, np.round(L_th,2)))
            or_dim    = df.shape[0]
            df        = df[df[col_name]>L_th]
            log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
            log.info(' > using %s-band to compute completeness.' %band)
            band_vals = df.loc[:, col_name].values   
        else:
            log.info('No cut in luminosity applied ' )
            log.info(' > using number counts to compute completeness.')
            band_vals = np.ones(df.shape[0])
        
        df.loc[:, 'completenessGoal'] = band_vals
        
        # ------ Add 'w' column for weights

        if band_weight is not None:
            w_name = band_weight+'_Lum'
            w      = df.loc[:, w_name].values
            log.info('Using %s for weighting' %col_name)
        else:
            w      = np.ones(df.shape[0])
            log.info('Using weights = 1')
            
        df.loc[:, 'w'] = w
        
        # ------ Keep only some columns
        
        if colnames_final is not None:
            log.info('Keeping only columns: %s' %colnames_final)
            df = df[colnames_final]
       
        # ------ Add pixel column. Note that not providing nest parameter to ang2pix defaults to nest=True, which has to be set in GW too!
        # df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta.to_numpy(), df.phi.to_numpy())
        
        # ------
        log.info('Loaded GLADE\n')
         
        self.data = self.data.append(df, ignore_index=True)
            

## ------------------------------------------------------------------------------------


class GLADEPlus(GalCat):
    
    def __init__(self, foldername, compl, useDirac, #finalData = None,
                 subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS', 'WISE'],
                 subsurveysExcl = [],
                 catalogname    = "GLADE+.txt",
                 groupname      = "Galaxy_Group_Catalogue.csv",
                 **kwargs):
        
        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        self.catalogname     = catalogname
        self.groupname       = groupname
        #self._finalData = finalData
        
        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)

        GalCat.__init__(self, foldername, compl, useDirac, **kwargs)
        
    
    def load(self, band             = None, 
                   band_weight      = None,
                   Lcut             = 0,
                   zMax             = 100,
                   z_flag           = None,
                   drop_z_uncorr    = False, # Not used
                   get_cosmo_z      = False, # Not used
                   pos_z_cosmo      = True,
                   drop_no_dist     = True,  # drops points with no redshift z_cmb entry
                   group_correct    = True,
                   which_z_correct  = 'z_cosmo',
                   CMB_correct      = True, # Now not used, because GLADE+ has already z_cmb
                   which_z          = 'z_cosmo_CMB',
                   galPosterior     = True,
                   err_vals         = True, #'GLADE',
                   drop_HyperLeda2  = True,
                   colnames_final   = ['theta','phi','z','z_err','z_lower','z_lowerbound',
                                       'z_upper','z_upperbound','w','completenessGoal']): 

        if band_weight is not None:
            assert band_weight==band

        loaded = False
        computePosterior = False
        posteriorglade = os.path.join(self._path, 'posteriorglade.csv')

        if galPosterior:
            if isfile(posteriorglade):
                log.info("Directly loading final data ")
                df     = pd.read_csv(os.path.join(self._path, 'posteriorglade.csv'))
                loaded = True
            else:
                computePosterior = True
                loaded           = False

        if not loaded:
            filepath_GLADE  = os.path.join(self._path, self.catalogname)
            filepath_groups = os.path.join(self._path, self.groupname)
            

            # ------ LOAD CATALOGUE

            log.info("\nLoading GLADE+ from %s" %filepath_GLADE)
            df = pd.read_csv(filepath_GLADE, sep=" ", header=None, low_memory=False)

            colnames = ['GLADE_n', 'PGC', 'GWGC_name', 'HYPERLEDA_name', 'TWOMASS_name', 'WISE_name', 'SDSS_name',
            'Obj_type_flag', 'RA', 'dec', 'B', 'B_err', 'B_flag', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
            'W1', 'W1_err', 'W2', 'W2_err', 'W1_flag', 'BJ', 'BJ_err', 'z_helio', 'z_cmb', 'z_flag', 'v_err', 'z_err',
            'dL', 'dL_err', 'dist_flag', 'Mst', 'Mst_err', 'BNS_MR', 'BNS_MR_err']
            df.columns = colnames
             

            # ------  SELECT SUBSURVEYS
            
            # if object is named in a survey (not NaN), it is present in a survey
            for survey in ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS', 'WISE']:
                # new column named suvey that will be True or False
                # if object is in or not
                
                #copy the name column
                df.loc[:,survey] = df[survey + '_name']
                
                # NaN to False
                df[survey] = df[survey].fillna(False)
                # not False (i.e. a name) to True
                df.loc[df[survey] != False, survey] = True

          
            # include only objects that are contained in at least one of the surveys listed in _subsurveysIncl (is non-empty list)
            mask = df[self._subsurveysIncl[0]] == True
            for incl in self._subsurveysIncl[1:]:
                mask = mask | (df[incl] == True)
            df = df.loc[mask]
            
            # explicitely exclude objects if they are contained in some survey(s) in _subsurveysExcl
            # (may be necessary if an object is in multiple surveys)
            for excl in self._subsurveysExcl:
                df = df.loc[df[excl] == False]
            
            
            # ------ Add theta, phi for healpix in radians
            
            df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
            df.loc[:,"phi"]   = df.RA*np.pi/180

            or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
            log.info('N. of objects: %s' %or_dim)

            
            # ------ Select parts of the catalogue
                
            if drop_no_dist:
                df=df[df.z_cmb.notna()==True]
                log.info('Keeping only galaxies with known value of redshift')
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim))
            
            if drop_HyperLeda2:
                df=df.drop(df[(df['HYPERLEDA_name'].isna()) & (df['dist_flag']==2)].index)
                log.info("Dropping galaxies with HyperLeda name=null and flag2=2") 
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim))
            
            
            # ------ Add z corrections

            df.loc[:,'z_cosmo'] = df.z_helio    # z_helio is the cosmological redshift in the heliocentric frame. 
                                                # Before converting to cmb, we apply group velocity corrections 
                                                # Use z_cmb if wanting to skip this step
                    
            if not CMB_correct and not group_correct and pos_z_cosmo:
                log.info('Keeping only galaxies with positive cosmological redshift')
                df = df[df.z_cosmo >= 0]
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
            
            if group_correct:
                log.info('Loading galaxy group catalogue from %s' %filepath_groups)
                df_groups = pd.read_csv(filepath_groups)
                self.group_correction(df, df_groups, which_z=which_z_correct)
                
            if which_z!='z':
                log.info("Renaming column '{:s}' to 'z'. This will be used in the analysis.".format(which_z))
                df.rename(columns={which_z:'z'}, inplace=True)
            
            # From now on, the relevant column for redshift, including all corrections, will be 'z'
            
            # ------ Potentially drop large z

            df = df[df.z < zMax]
            
            
            # ------ Add z errors
            
            if err_vals: #is not None:
                df.loc[:, 'z_lowerbound']                   = df.z - 3*df.z_err
                df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
                df.loc[:, 'z_lower']                        = df.z - df.z_err
                df.loc[df.z_lower < 0.5*df.z, 'z_lower']    = 0.5*df.z
                df.loc[:, 'z_upper']                        = df.z + df.z_err
                df.loc[:, 'z_upperbound']                   = df.z + 3*df.z_err
            
                # ------ Estimate galaxy posteriors with contant-in-comoving prior
                
                if computePosterior:
                    self.include_vol_prior(df)

        # ------ End if not use precomputed table
        #        Always be able to still chose the weighting and cut.
        
        if band=='B' or band_weight=='B':
            add_B_lum = True
            add_K_lum = False
        elif band=='K' or band_weight=='B':
            add_B_lum = False
            add_K_lum = True
        else:
            add_B_lum = False
            add_K_lum = False
            
        # ------ Add B luminosity

        if add_B_lum:
            log.info('Computing total luminosity in B band')
            
            my_dist                 = fLCDM.dL(df.z.values, lambda_cosmo_GLADE)     
            df.loc[:,"B_Abs_corr"]  = df.B_Abs - 5*np.log10(my_dist) + 5*np.log10(df.dL.values)
            BLum                    = df.B_Abs_corr.apply(lambda x: TotLum(x, glob.MBSun))
            df.loc[:,"B_Lum"]       = BLum
            df                      = df.drop(columns='B_Abs')
            # df = df.drop(columns='B') don't assume it's here      
        
        # ------ Add K luminosity

        if add_K_lum:
            log.info('Computing total luminosity in K band')

            my_dist                 = fLCDM.dL(df.z.values, lambda_cosmo_GLADE)     
            df.loc[:,"K_Abs"]       = df.K-5 * np.log10(my_dist) - 25
            KLum                    = df.K_Abs.apply(lambda x: TotLum(x, glob.MKSun))
            df.loc[:,"K_Lum"]       = KLum
            df                      = df.drop(columns='K')
            # df = df.drop(columns='K_Abs') don't assume it's here        
        
        # ------ Apply cut in luminosity

        if band is not None:
            col_name  = band+'_Lum'
            if band=='B':
                Lstar = glob.LBstar07
            elif band=='K':
                Lstar = glob.LKstar07
            
            L_th = Lcut*Lstar
            log.info('Applying cut in luminosity in %s-band ' %band)
            log.info(' > L_* in %s band is L_*=%s' %(band, np.round(Lstar,5)))
            log.info(' > selecting galaxies with %s>%sL_* = %s' %(col_name, Lcut, np.round(L_th,2)))
            or_dim    = df.shape[0]
            df        = df[df[col_name]>L_th]
            log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
            log.info(' > using %s-band to compute completeness.' %band)
            band_vals = df.loc[:, col_name].values   
        else:
            log.info('No cut in luminosity applied ' )
            log.info(' > using number counts to compute completeness.')
            band_vals = np.ones(df.shape[0])
        
        df.loc[:, 'completenessGoal'] = band_vals

        # ------ Add 'w' column for weights

        if band_weight is not None:
            w_name = band_weight+'_Lum'
            w      = df.loc[:, w_name].values
            log.info('Using %s for weighting' %col_name)
        else:
            w      = np.ones(df.shape[0])
            log.info('Using weights = 1')
            
        df.loc[:, 'w'] = w
        
        # ------ Keep only some columns
        
        if colnames_final is not None:
            log.info('Keeping only columns: %s' %colnames_final)
            df = df[colnames_final]
       
        # ------ Add pixel column. Note that not providing nest parameter to ang2pix defaults to nest=True, which has to be set in GW too!
        # df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta.to_numpy(), df.phi.to_numpy())
        
        # ------
        log.info('Loaded GLADE+\n')
        
        self.data = self.data.append(df, ignore_index=True)



## ------------------------------------------------------------------------------------


class GLADEPlus_v2(GalCat):
    
    def __init__(self, 
                foldername, 
                compl, 
                useDirac, #finalData = None,
                catalogname    = "glade+.hdf5",
                groupname      = "Galaxy_Group_Catalogue.csv",
                **kwargs):
        

        self.catalogname     = catalogname
        self.groupname       = groupname

        GalCat.__init__(self, foldername, compl, useDirac, **kwargs)
        
    
    def load(self,
   
             band             = None,  #  ["B", "bJ", "K", "W1"],
             band_weight      = None,  #  ["B", "bJ", "K", "W1"],
             Lcut             = None,
            #  galPosterior     = True,
             err_vals         = True, #'GLADE',
             pixels           = None, 
            #  nside            = None,
             nest             = False,
             z_range          = None,
             ): 

        if band_weight is not None:
            assert band_weight==band

        loaded                = False
        computePosterior      = False
        # posteriorglade        = os.path.join(self._path, 'posteriorglade.csv')

        # if galPosterior:
        #     if isfile(posteriorglade):
        #         log.info("Directly loading final data ")
        #         df     = pd.read_csv(os.path.join(self._path, 'posteriorglade.csv'))
        #         loaded = True
        #     else:
        #         computePosterior = True
        #         loaded           = False

        if not loaded:
            filepath_GLADE  = os.path.join(self._path, self.catalogname)
            filepath_groups = os.path.join(self._path, self.groupname)
            

            # ------ LOAD CATALOGUE

            log.info("\nLoading GLADE+ from %s" %filepath_GLADE)
            columns = ['ra', 'dec', 'z', 'sigmaz', 'm_B', 'm_K', 'm_W1', 'm_bJ']

            if pixels is None:
                log.warning("Bye bye memory! Ask for specific pixels!")
                with h5py.File(filepath_GLADE, 'r') as f:
                    df = pd.DataFrame(np.array([np.array(f['table'][i]) for i in columns]).T)
                    df.columns=columns
            else:
                log.info("Slicing the catalog")
                log.info(" > keeping galaxies inside {:d} pixels".format(len(pixels)))
                
                if nside is None:
                    log.error("You should provide a value for `nside`")
                elif not isinstance(nside, int):
                    log.error("`nside` must be an int")

                pixn = "indices_"+str(nside)

                with h5py.File(filepath_GLADE, 'r') as f:
                    all_pixels = np.array(f[pixn]["pixel_indices"])
                    mask       = np.isin(all_pixels, pixels)
                    log.info(" > {:d} galaxies inside given pixels ({:.3%} of total)".format(mask.sum(), mask.sum()/mask.shape[0]))

                    if z_range is not None:
                        z      = np.array(f["table"]["z"])
                        mask_z = (z>z_range[0]) & (z<z_range[1])
                        mask   = mask & mask_z
                        log.info(" > {:d} inside given z range ({:.3%} of total)".format(mask.sum(), mask.sum()/mask.shape[0]))

                    df         = pd.DataFrame(np.array([np.array(f['table'][i][mask]) for i in columns]).T)
                    df.columns = columns
                    all_pixels = all_pixels[mask]

            # ------ Add theta, phi for healpix in radians
            
            df.loc[:,"theta"] = np.pi/2 - df.dec
            df.loc[:,"phi"]   = df.ra
            df.loc[:,"pix"]   = all_pixels

            or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
            log.info('N. of objects: %s' %or_dim)
            
            
            # ------ Add z errors
            
            # if err_vals: #is not None:
            #     df.loc[:, 'z_lowerbound']                   = df.z - 3*df.sigmaz
            #     df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
            #     df.loc[:, 'z_lower']                        = df.z - df.sigmaz
            #     df.loc[df.z_lower < 0.5*df.z, 'z_lower']    = 0.5*df.z
            #     df.loc[:, 'z_upper']                        = df.z + df.sigmaz
            #     df.loc[:, 'z_upperbound']                   = df.z + 3*df.sigmaz
            
            # ------ Estimate galaxy posteriors with contant-in-comoving prior
            
            if computePosterior:
                self.include_vol_prior(df)

        # ------ End if not use precomputed table
        #        Always be able to still chose the weighting and cut.
        
        if band=='B' or band_weight=='B':
            add_B_lum = True
            add_K_lum = False
        elif band=='K' or band_weight=='B':
            add_B_lum = False
            add_K_lum = True
        else:
            add_B_lum = False
            add_K_lum = False
            
        # # ------ Add luminosity

        if band is not None:
            if ~np.isin(band,["B", "bJ", "K", "W1"]):
                ValueError("band not available!")
            
            if band == "B":
                Msun = glob.MBSun
            elif band == "K":
                Msun = glob.MKSun
            else:
                ValueError("Msun not implemented for this band")

            bandn = "m_" + band
            lumn  = "L_" + band

            my_dist                = fLCDM.dL(df.z.values, lambda_cosmo_GLADE)  
            df.loc[:,bandn+"_abs"] = df[bandn]-5 * np.log10(my_dist) - 25
            df.loc[:,lumn]         = TotLum(df[bandn+"_abs"], Msun)


        
            # ------ Apply cut in luminosity

            if Lcut is not None:

                if band=='B':
                    Lstar = glob.LBstar07
                elif band=='K':
                    Lstar = glob.LKstar07
                
                L_th      = Lcut*Lstar
                log.info('Applying cut in luminosity in %s-band ' %band)
                log.info(' > L_* in %s band is L_*=%s' %(band, np.round(Lstar,5)))
                log.info(' > selecting galaxies with %s>%sL_* = %s' %(lumn, Lcut, np.round(L_th,2)))
                or_dim    = df.shape[0]
                df        = df[df[lumn]>L_th]
                log.info(" > kept {:d} points ({:.0%} of total)".format(df.shape[0], df.shape[0]/or_dim) )
                log.info(' > using %s-band to compute completeness.' %band)
                band_vals = df.loc[:, lumn].values   
            else:
                log.info('No cut in luminosity applied ' )
                log.info(' > using number counts to compute completeness.')
                band_vals = np.ones(df.shape[0])
            
            df.loc[:, 'completenessGoal'] = band_vals

            # ------ Add 'w' column for weights

            if band_weight is not None:
                w      = df.loc[:, lumn].values
                log.info('Using %s for weighting' %lumn)
            else:
                w      = np.ones(df.shape[0])
                log.info('Using weights = 1')
                
            df.loc[:, 'w'] = w
               
        # # ------ Add pixel column. Note that not providing nest parameter to ang2pix defaults to nest=True, which has to be set in GW too!
        # df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta.to_numpy(), df.phi.to_numpy())
        
        # ------
        log.info('Loaded GLADE+')
        
        self.data = self.data.append(df, ignore_index=True)


    def pixelize_region(self, pixels):
        mu_list     = []
        sigma_list  = []
        N_list      = []

        for ipix, pix in enumerate(pixels):
            isin = (self.data.pix == pix)

            N_list.append(isin.sum())
            mu_list.append(np.array(self.data.z[isin]))
            sigma_list.append(np.array(self.data.sigmaz[isin]))
        
        return {"mu": mu_list, "sigma": sigma_list, "Ngal": N_list, "NgalTot": sum(N_list), "Npix": len(pixels)}



    def compute_pCAT_pixelized(self, z_grid, dict_galaxies, do_post_norm=False):
        Npix  = dict_galaxies["Npix"]
        p_cat = np.zeros((Npix, len(z_grid)))

        for i in range(Npix):
            p_cat[i,:] = sum_of_gaussians(z_grid, dict_galaxies["mu"][i], dict_galaxies["sigma"][i])

            if do_post_norm:
                p_cat[i,:] /= np.trapz(p_cat[i,:], z_grid)

        p_catw = p_cat * np.array(dict_galaxies["Ngal"])[:, np.newaxis] / dict_galaxies["NgalTot"]

        return p_cat, p_catw


