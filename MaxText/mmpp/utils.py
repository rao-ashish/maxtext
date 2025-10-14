"""Various utilities for mmpp."""

import contextlib
import dataclasses
from functools import wraps
from typing import Callable, Sequence

import jax
import numpy as np

# vjp-related utils imports
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info
from jax._src.api import VJP
from jax._src.util import tuple_update
from jax._src.api import NotNeeded

# Profiling imports
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')
import nvtx


### vjp-related utilities

# def fwd_and_bwd(
#     fun: Callable, argnums: Sequence[int], caller_saved_among_argnums: Sequence[bool],
#     has_aux: bool = False, jitted: bool = True,
# ) -> tuple[Callable, Callable]:
#   def fwd(*args, **kwargs):
#     dbg = debug_info('fwd_and_bwd', fun, args, kwargs)
#     f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
#     f_partial, dyn_args = argnums_partial(
#         f, argnums, args, require_static_args_hashable=False)
#     # return jax._src.api._saved_input_vjp(
#     #     f_partial, caller_saved_among_argnums, *dyn_args, has_aux=has_aux)
#     return jax._src.api.saved_input_vjp(
#         f_partial, caller_saved_among_argnums, *dyn_args)
#   def bwd(f_vjp, *outgrad_and_saved):
#     assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1
#     return f_vjp(*outgrad_and_saved)
#   if jitted:
#     fwd = jax.jit(fwd)
#     bwd = jax.jit(bwd)
#   return fwd, bwd


def fwd_and_bwd(
    fun: Callable,
    argnums: Sequence[int],
    caller_saved_among_argnums: Sequence[bool],
    has_aux: bool = False,
    jitted: bool = True,
) -> tuple[Callable, Callable]:
    def fwd(*args, **kwargs):
      # f_partial is a partially evaluated version of fun that only takes in
      # the arguments at argnums as input.
      def f_partial(*remaining_args):
        assert len(remaining_args) == len(argnums)
        new_args = []
        j = 0
        for i in range(len(args)):
          if i in argnums:
            new_args.append(remaining_args[j])
            j += 1
          else:
            new_args.append(args[i])

        return fun(*new_args)
      
      # Compute the VJP of f_partial.
      curr_remaining_args = [args[argnum] for argnum in argnums]
      out_primals, raw_vjp, *rest = jax._src.api.vjp3(
        f_partial, *curr_remaining_args, has_aux=has_aux
      )

      # The inputs are saved inside raw_vjp.args_res, and the cached 
      # activations are saved inside raw_vjp.opaque_residuals. Filter 
      # raw_vjp.args_res to only store the inputs that will not be explicitly 
      # provided by the caller.
      raw_vjp.args_res = [
        res if not caller_saved else NotNeeded() 
        for res, caller_saved in zip(raw_vjp.args_res, caller_saved_among_argnums)
      ]
      return out_primals, raw_vjp, *rest

    def bwd(f_vjp, *outgrad_and_saved):
      assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1

      outgrad = outgrad_and_saved[0]
      caller_saved_vals = outgrad_and_saved[1:]

      # Reconstruct the full args_res by inserting caller saved values.
      full_args_res = []
      j = 0
      for i, caller_saved in enumerate(caller_saved_among_argnums):
        if caller_saved:
          full_args_res.append(caller_saved_vals[j])
          j += 1
        else:
          full_args_res.append(f_vjp.args_res[i])

      full_vjp = VJP(fun, f_vjp.in_tree, f_vjp.out_tree, full_args_res, 
                     f_vjp.opaque_residuals)

      return full_vjp(outgrad)

    if jitted:
      fwd = jax.jit(fwd)
      bwd = jax.jit(bwd)
    return fwd, bwd


def vjp_unpack(f_vjp):
  # assert isinstance(f_vjp, VJP)  
  # data = (f_vjp.args_res, f_vjp.opaque_residuals)
  # metadata = (f_vjp.fun, f_vjp.in_tree, f_vjp.out_tree)
  # return (data, metadata)
  return f_vjp

def vjp_pack(f_vjp_unpacked):
  return f_vjp_unpacked
  # assert isinstance(f_vjp_unpacked, tuple)
  # assert len(f_vjp_unpacked) == 2

  # data, metadata = f_vjp_unpacked
  # args_res, opaque_residuals = data
  # fun, in_tree, out_tree = metadata
  
  # return VJP(fun, in_tree, out_tree, args_res, opaque_residuals)

def with_vjp_unpack(fwd):
  @wraps(fwd)
  def wrapper(*args, **kwargs):
    out = fwd(*args, **kwargs)
    assert 2 <= len(out) <= 3
    return tuple_update(out, 1, vjp_unpack(out[1]))
  return wrapper

def with_vjp_pack(bwd):
  @wraps(bwd)
  def wrapper(*args, **kwargs):
    assert len(args) == 3
    args = tuple_update(args, 0, vjp_pack(args[0]))
    return bwd(*args, **kwargs)
  return wrapper


### Various debugging utilities

def dump_memory_usage_snapshot(state):
  def jax_tree_size_bytes(tree):
    size_bytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
      if isinstance(leaf, (jax.Array, np.ndarray)):
        size_bytes += leaf.size * leaf.dtype.itemsize
    return size_bytes

  def gb(size_bytes):
    return size_bytes / 1024**3

  record = {}

  print("Memory usage:")
  print("  by device:")
  for i, device in enumerate(jax.devices()):
    stats = device.memory_stats()
    used = gb(stats["bytes_in_use"])
    limit = gb(stats["bytes_limit"])
    peak = gb(stats["peak_bytes_in_use"])
    print(
        f"    {device}: {used:7.01f}/{limit:7.01f}GB ({used/limit*100:4.1f}%)"
        f"  |  peak {peak:7.01f}GB ({peak/limit*100:4.1f}%)"
    )
    record[f"device{i}_used_gb"] = used
    record[f"device{i}_limit_gb"] = limit
    record[f"device{i}_peak_gb"] = peak

  print("  by known state:")
  is_leaf = lambda x: not isinstance(x, dict) or "params_by_stage" not in x
  flat_state, _ = jax.tree_util.tree_flatten_with_path(state, is_leaf=is_leaf)
  total_size_bytes = 0
  for path, value in flat_state:
    size_bytes = jax_tree_size_bytes(value)
    total_size_bytes += size_bytes
    print(f"    state{jax.tree_util.keystr(path):24}: {gb(size_bytes):7.01f}GB")
    assert len(path) == 1 and isinstance(path[0], jax.tree_util.DictKey)
    record[f"state_{path[0].key}_gb"] = gb(size_bytes)
  print(f"  => total size                   : {gb(total_size_bytes):7.01f}GB")

  return record


@contextlib.contextmanager
def annotate(name, color):
  with nvtx.annotate(name, color=color):
    yield


def annotate_step(fn, start_step=4, end_step=6):
  step = 0
  @wraps(fn)
  def wrapper(*args, **kwargs):
    nonlocal step
    if step == start_step:
      libcudart.cudaProfilerStart()
    with annotate(f"step{step}", color="white"):
      res = fn(*args, **kwargs)
    if step == end_step:
      libcudart.cudaProfilerStop()
    step += 1
    return res
  return wrapper
