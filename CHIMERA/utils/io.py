import h5py
from ..utils.config import jnp
import numpy as np
import equinox as eqx
from dataclasses import fields, replace

def save_set(obj, dir_file, attrs=[], datasets=[], groups=[]):
  """Save attributes and datasets of an object to dir_file."""
  with h5py.File(dir_file, 'w') as f:
    for a in attrs:
      f.attrs[a] = getattr(obj, a)
    for d in datasets:
      f.create_dataset(d, data=jnp.array(getattr(obj, d)))
    for g in groups:
      dg = f.create_group(g)
      dict_to_save = getattr(obj, g)
      for k, v in dict_to_save.items():
        dg.create_dataset(k, data=v)

def load_set(obj, dir_file, attrs=[], datasets=[], groups=[]):
  """Load attributes and datasets into a new Equinox object (or update a mutable object)."""
  new_fields = {}
  with h5py.File(dir_file, 'r') as f:
    for a in attrs:
      new_fields[a] = f.attrs[a]
    for d in datasets:
      new_fields[d] = jnp.array(f[d][:])
    for g in groups:
      group_data = {}
      for k in f[g].keys():
        group_data[k] = jnp.array(f[g][k][:])
      new_fields[g] = group_data
  if isinstance(obj, eqx.Module):
    field_names = {f.name for f in fields(obj)}
    current_fields = {name: getattr(obj, name) for name in field_names}
    current_fields.update(new_fields)
    return obj.__class__(**current_fields)
  else:
    for k, v in new_fields.items():
      setattr(obj, k, v)
    return obj


def load_data_h5(fname, group_h5=None, backend='jax', require_keys=None):
  """Generic function to load data from h5 files with optional validation.

  Args:
      fname: Path to HDF5 file
      group_h5: Optional group within HDF5 file
      backend: 'jax' or 'numpy' for array types
      require_keys: List of keys that must be present

  Returns:
      Dictionary of arrays
  """
  xp = jnp if backend == 'jax' else np
  data = {}
  with h5py.File(fname, 'r') as f:
    target = f if group_h5 is None else f[group_h5]
    if require_keys:
      missing = [k for k in require_keys if k not in target]
      if missing:
        raise ValueError(f"Missing required keys in {fname}: {missing}")
    for key in target.keys():
      data[key] = xp.array(target[key][:])
  return data
