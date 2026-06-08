import os
import pickle
import arviz as az
import jax
from numpyro.infer import MCMC

def run_mcmc_with_checkpoints(mcmc, rng_key, checkpoint_every = 500, output_file = './test', restart=False, additional_samples = 0, model_args={}):
  # File names
  arviz_file = f"{output_file}.nc"
  state_file = f"{output_file}_state.pkl"

  # Random keys
  warmup_key, sampling_key = jax.random.split(rng_key)

  # Initialize or load existing data
  if restart:
    files_exist = os.path.exists(arviz_file) and os.path.exists(state_file)
    if files_exist:
      print(f"Loading checkpoint from {arviz_file}")
      idata = az.from_netcdf(arviz_file)
      print(f"Loading last mcmc state from {state_file}")
      with open(state_file, 'rb') as f:
        saved_state = pickle.load(f)
      warmup_done = saved_state['warmup_done']
      num_samples_drawn = saved_state['num_samples_drawn']

      if warmup_done:
        num_samples_to_do = mcmc.num_samples - num_samples_drawn + additional_samples
        mcmc_state = saved_state['mcmc_state']
        print(f"Resuming from {num_samples_drawn} samples, need {num_samples_to_do} more samples")
      else:
        # Warmup was not completed, but files exist - this is an inconsistent state
        print("Warmup was not completed in previous run. Restarting warmup...")
        files_exist = False  # Force fresh start for warmup
    else:
      print("Checkpoint files not found or incomplete. Starting from scratch.")
      files_exist = False

  if not restart or not files_exist:
    print("Running warmup...")
    mcmc_warmup = MCMC(mcmc.sampler,
                    num_warmup=mcmc.num_warmup,
                    num_samples=1,
                    num_chains=mcmc.num_chains,
                    chain_method=mcmc.chain_method,
                    jit_model_args=True,
                    progress_bar=mcmc.progress_bar
                  )
    mcmc_warmup.warmup(warmup_key, **model_args)
    mcmc_state = mcmc_warmup.last_state
    print("Warmup completed.")

    # Initialize state
    warmup_done = True
    num_samples_drawn = 0
    num_samples_to_do = mcmc.num_samples + additional_samples
    with open(state_file, 'wb') as f:
      pickle.dump({
        'warmup_done': warmup_done,
        'num_samples_drawn': num_samples_drawn,
        'mcmc_state': mcmc_state,
      }, f)
    print(f"Initial state saved to {state_file}")
    idata = None

  # Calculate how many iterations are needed to do the samples and how mnay samples per iteration
  num_iterations = (num_samples_to_do + checkpoint_every - 1) // checkpoint_every
  num_samples_per_iteration = checkpoint_every

  # Istantiate post-warmup smapler
  mcmc_postwarmup = MCMC(mcmc.sampler,
                      num_warmup=0,
                      num_samples=num_samples_per_iteration,
                      thinning=mcmc.thinning,
                      num_chains=mcmc.num_chains,
                      postprocess_fn=mcmc.postprocess_fn,
                      chain_method=mcmc.chain_method,
                      jit_model_args=True,
                      progress_bar=mcmc.progress_bar,
                    )
  mcmc_postwarmup.post_warmup_state = mcmc_state # start after the warmup of from the loaded state

  # Main loop
  sampling_keys = jax.random.split(sampling_key, num_iterations)
  for i in range(num_iterations):

    # Run sampler with num_samples_per_iteration
    print(f"Running iteration {i+1} with {num_samples_per_iteration} samples")
    mcmc_postwarmup.run(sampling_keys[i], **model_args)

    # Save ArviZ InferenceData
    new_idata = az.from_numpyro(mcmc_postwarmup)
    if idata is None:
      idata = new_idata
    else:
      idata = az.concat(idata, new_idata, dim="draw")
    idata.to_netcdf(arviz_file, overwrite_existing=True)

    # Save MCMC state separately
    num_samples_drawn += num_samples_per_iteration
    with open(state_file, 'wb') as f:
      pickle.dump({
        'warmup_done': warmup_done,
        'num_samples_drawn': num_samples_drawn,
        'mcmc_state': mcmc_postwarmup.last_state
      }, f)

    # Update mcmc state for next iteration
    mcmc_postwarmup.post_warmup_state = mcmc_postwarmup.last_state

  print(f"MCMC finished. Total samples: {num_samples_drawn}")
  return idata
