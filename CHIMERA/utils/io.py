"""HDF5 input/output utilities for CHIMERA objects.

This module provides functions for saving and loading CHIMERA objects to/from
HDF5 files, with support for:
- Equinox module serialization/deserialization
- Selective attribute, dataset, and group saving
- JAX/NumPy backend selection
- Required key validation
"""

import h5py
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from dataclasses import fields

def save_set(obj, dir_file, attrs=[], datasets=[], groups=[]):
  """Saves selected attributes, datasets, and groups from an object to HDF5.
  
  Args:
    obj: Object containing attributes/datasets/groups to save
    dir_file (str): Output HDF5 file path
    attrs (list[str], optional): Attribute names to save as HDF5 attributes
    datasets (list[str], optional): Attribute names to save as HDF5 datasets
    groups (list[str], optional): Dictionary attributes to save as HDF5 groups
  
  Example:
    >>> obj.alpha = 1.5
    >>> obj.data = jnp.array([1, 2, 3])
    >>> obj.params = {'x': 0.1, 'y': 0.2}
    >>> save_set(obj, 'output.h5', attrs=['alpha'], 
    ...          datasets=['data'], groups=['params'])
  """
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
  """Loads attributes, datasets, and groups from HDF5 into an object.
  
  For Equinox modules, creates a new immutable instance with updated fields.
  For mutable objects, updates fields in-place.
  
  Args:
    obj: Object to load data into (Equinox module or mutable object)
    dir_file (str): Input HDF5 file path
    attrs (list[str], optional): Attribute names to load from HDF5 attributes
    datasets (list[str], optional): Dataset names to load as object attributes
    groups (list[str], optional): Group names to load as dictionary attributes
  
  Returns:
    object: New instance (Equinox) or updated object (mutable)
  
  Example:
    >>> loaded_obj = load_set(obj_template, 'output.h5',
    ...                       attrs=['alpha'], datasets=['data'])
  """
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
    return replace(obj, **new_fields)
  else:
    for k, v in new_fields.items():
      setattr(obj, k, v)
    return obj


def load_data_h5(fname, group_h5=None, backend='jax', require_keys=None):
  """Loads all datasets from an HDF5 file with validation.
  
  Args:
    fname (str): Path to HDF5 file
    group_h5 (str, optional): HDF5 group path. If None, loads from root.
    backend (str, optional): Array backend - 'jax' or 'numpy'. Defaults to 'jax'.
    require_keys (list[str], optional): Keys that must be present. Raises ValueError if missing.
  
  Returns:
    dict[str, array]: Dictionary mapping dataset names to arrays
  
  Raises:
    ValueError: If any required keys are missing from the file
  
  Example:
    >>> data = load_data_h5('catalog.h5', group_h5='galaxies',
    ...                     require_keys=['ra', 'dec', 'redshift'])
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
