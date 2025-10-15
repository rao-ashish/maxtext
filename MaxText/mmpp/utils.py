"""Various utilities for mmpp."""

import contextlib
import dataclasses
from functools import wraps
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
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
    # Constructs a partially evaluated version of fun that only takes in the
    # arguments at argnums as input. 
    # 
    # Returns:
    #   f_partial: The partially evaluated function as an lu.WrappedFun.
    #   dyn_args: The remaining arguments to pass into f_partial to fully 
    #     evaluate it. This is a subset of *args.
    #   bound_args: The elements of *args captured inside f_partial.
    def make_f_partial(*args, **kwargs):
      dbg = debug_info('fwd_and_bwd', fun, args, kwargs)
      f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
      f_partial, dyn_args = argnums_partial(
          f, argnums, args, require_static_args_hashable=False)
      return f_partial, dyn_args

    def argnum_is_caller_saved(argnum):
        return (
          argnum in argnums and 
          caller_saved_among_argnums[argnums.index(argnum)]
        )

    def fwd(*args, **kwargs):
      # bwd needs enough metadata to construct the vjp of f_partial. We just
      # need to save the args that are not user provided and the kwargs.
      vjp_metadata = {
          "saved_args": [
            args[i] for i in range(len(args)) if not argnum_is_caller_saved(i)
          ],
          "all_args": args,
          "kwargs": kwargs,
        }

      if has_aux:
        return vjp_metadata, *fun(*args, **kwargs)
      return vjp_metadata, fun(*args, **kwargs)

    def bwd(vjp_metadata, *outgrad_and_saved):
      assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1

      outgrad = outgrad_and_saved[0]
      caller_saved_vals = outgrad_and_saved[1:]

      # Reconstruct the full args list used to create f_partial.
      args = vjp_metadata["all_args"]
      # args = []
      # num_args = len(vjp_metadata["saved_args"]) + sum(caller_saved_among_argnums)

      # curr_caller_saved_idx = 0
      # curr_saved_idx = 0

      # for i in range(num_args):
      #   if argnum_is_caller_saved(i):
      #     args.append(caller_saved_vals[curr_caller_saved_idx])
      #     curr_caller_saved_idx += 1
      #   else:
      #     args.append(vjp_metadata["saved_args"][curr_caller_saved_idx])
      #     curr_saved_idx += 1

      # Construct f_partial, and take its vjp.
      f_partial, dyn_args = make_f_partial(*args, **vjp_metadata["kwargs"])

      if has_aux:
        _, vjp_fn, _ = jax._src.api._vjp3(f_partial, *dyn_args, has_aux=has_aux)
      else:
        _, vjp_fn = jax._src.api._vjp3(f_partial, *dyn_args, has_aux=has_aux)

      return vjp_fn(outgrad)

    if jitted:
      fwd = jax.jit(fwd)
      bwd = jax.jit(bwd)
    return fwd, bwd


def vjp_unpack(f_vjp):
  # assert isinstance(f_vjp, VJP)
  # data = (f_vjp.args_res, f_vjp.opaque_residuals)
  # # metadata = (f_vjp.fun, f_vjp.in_tree, f_vjp.out_tree)
  
  # dummy_data = jax.tree.map(lambda x: 123 * jnp.ones_like(x), data)
  # dummy_vjp = VJP(f_vjp.fun, f_vjp.in_tree, f_vjp.out_tree, dummy_data[0], 
  #                 dummy_data[1])

  # return (data, dummy_vjp)
  return f_vjp

def vjp_pack(f_vjp_unpacked):

  # assert isinstance(f_vjp_unpacked, tuple)

  # data, dummy_vjp = f_vjp_unpacked

  # return VJP(
  #   dummy_vjp.fun, dummy_vjp.in_tree, dummy_vjp.out_tree, data[0], data[1]
  # )

  return f_vjp_unpacked
  # assert isinstance(f_vjp_unpacked, tuple)
  # assert len(f_vjp_unpacked) == 2

  # data, metadata = f_vjp_unpacked
  # args_res, opaque_residuals = data
  # fun, in_tree, out_tree = metadata
  
  # return VJP(fun, in_tree, out_tree, args_res, opaque_residuals)

def with_vjp_unpack(fwd):
  return fwd
  # @wraps(fwd)
  # def wrapper(*args, **kwargs):
  #   out = fwd(*args, **kwargs)
  #   assert 2 <= len(out) <= 3
  #   return tuple_update(out, 1, vjp_unpack(out[1]))
  # return wrapper

def with_vjp_pack(bwd):
  return bwd
  # @wraps(bwd)
  # def wrapper(*args, **kwargs):
  #   assert len(args) == 3
  #   args = tuple_update(args, 0, vjp_pack(args[0]))
  #   return bwd(*args, **kwargs)
  # return wrapper


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
