.. default-role:: math

.. _getting_started:



Mock data analysis
------------------
This workflow reproduces the analysis bases on the mock O4- and O5-like catalogs in `Borghi et al. 2023 <http://www.numpy.org/>`_.


1. Initialize the Likelihood and Bias terms

.. code-block:: python

    from CHIMERA.Likelihood import MockLike
    from CHIMERA.Bias import Bias

    like = MockLike(model_cosmo, model_mass,  model_rate,
                    data_GW, data_GW_names, data_GW_smooth, data_GAL_dir, data_GAL_zerr,
                    nside_list, npix_event,sky_conf,
                    z_int_H0_prior, z_int_sigma, z_int_res, z_det_range)

    bias = Bias(model_cosmo, model_mass, model_rate, file_inj, snr_th)


2. Define function to compute the full likelihood (likelihood and bias)

.. code-block:: python
    
    def combine_events():

    def lnlike(lambda_cosmo, lambda_mass, lambda_rate):
        return like.compute_ln(lambda_cosmo, lambda_mass, lambda_rate) -\
               like.Nevents * bias.compute_ln(lambda_mass, lambda_cosmo, lambda_rate)






..    import pylick.io as io
..    from pylick.indices import IndexLibrary
..    from pylick.analysis import Galaxy

..    spectrum     = io.load_spec_fits(dir_spec, filename, colnames=['lambda', 'flux', 'flux_err'])
..    ind_library  = IndexLibrary(index_keys)

..    ind_measured = Galaxy(ID, index_list, spec_wave=spectrum[0], spec_flux=spectrum[1], spec_err=spectrum[2], z=z)
..    vals, errs   = ind_measured.vals, ind_measured.errs
..    print(ind_measured)


LVK data analysis
-----------------
Define a function to load the spectra from a catalog folder, (2) load the table of spectral indices to measure, (3) call the *Catalog* class,

.. code-block:: python

   import pylick.io as io
   from pylick.indices import IndexLibrary
   from pylick.analysis import Catalog

   def load_spec(ID):
      ...
      return [wave, flux, ferr, mask]

   IDs = [...]
   ind_library  = IndexLibrary(index_keys)

   ind_measured = Catalog(IDs, load_spec, index_keys, z=zs, do_plot=True, verbose=True)

