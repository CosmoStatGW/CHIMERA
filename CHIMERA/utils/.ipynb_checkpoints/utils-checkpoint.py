from functools import partial
from .jax_config import jax, jnp
import h5py

@partial(jax.jit, static_argnames=['Ndraw'])
def get_neff(weights, mu, Ndraw=None):
    
    nsamples = weights.shape[-1]
    if Ndraw is None:
        Ndraw = nsamples  # true if weights have shape (nsamples,), (nevents, nsamples) or (nparams, nevents, nsamples)
    
    s2   = jnp.sum(weights**2, axis=-1) / Ndraw**2
    sig2 = s2 - mu**2 / Ndraw
    
    return jax.lax.cond(jnp.abs(sig2) <= 1.e-15, lambda _ : 0.0, lambda _ : mu**2 / sig2, operand=None)


def load_data_h5(fname, group_h5=None):
    """Generic function to load data from h5 files

    Args:
        fname (str): path to the h5 file

    Returns:
        h5py.File: h5py file
    """
    events={}
    if group_h5 is None:
        with h5py.File(fname, 'r') as f:
            for key in f.keys(): 
                events[key] = jnp.array(f[key][:])
        return events
    else:
        with h5py.File(fname, 'r') as f:
            for key in f[group_h5].keys(): 
                events[key] = jnp.array(f[group_h5][key][:])
        return events


