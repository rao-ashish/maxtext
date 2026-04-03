"""Debugging utilities for MPMD pipeline parallelism implementation."""

import os
from functools import wraps

import jax
import numpy as np
from jax.sharding import NamedSharding

from ctypes import cdll

libcudart = cdll.LoadLibrary("libcudart.so")


# ==== Memory usage debugging utils ==== #


def _dump_memory_usage_snapshot(data):
    """Given a dictionary mapping string keys to PyTree values, print summaries
    of memory usage by device and by dictionary key.

    Returns:
        record: A dictionary holding the printed memory usage info.
    """

    def jax_tree_size_bytes(tree, device_id=None):
        size_bytes = 0

        for leaf in jax.tree_util.tree_leaves(tree):
            if isinstance(leaf, jax.Array) and device_id is not None:
                for shard in leaf.global_shards:
                    if shard.device.id == device_id:
                        size_bytes += shard.data.size * shard.data.dtype.itemsize
            elif isinstance(leaf, (jax.Array, np.ndarray)):
                size_bytes += leaf.size * leaf.dtype.itemsize
        return size_bytes

    def gb(size_bytes):
        return size_bytes / 1000**3

    record = {}

    print("Memory usage:")
    print("  by device:")
    for i, device in enumerate(jax.devices()):
        if device not in jax.local_devices():
            continue

        stats = device.memory_stats()
        used = gb(stats["bytes_in_use"])
        limit = gb(stats["bytes_limit"])
        peak = gb(stats["peak_bytes_in_use"])
        print(
            f"    {device}: {used:7.01f}/{limit:7.01f}GB ({used / limit * 100:4.1f}%)"
            f"  |  peak {peak:7.01f}GB ({peak / limit * 100:4.1f}%)"
        )
        record[f"device{i}_used_gb"] = used
        record[f"device{i}_limit_gb"] = limit
        record[f"device{i}_peak_gb"] = peak

        # Also record how much of the known state is on this device.
        known_state_bytes = 0
        for key, value in data.items():
            known_state_bytes += jax_tree_size_bytes(value, device_id=i)
        record[f"device{i}_known_state_gb"] = gb(known_state_bytes)

    print("  by known state:")
    total_size_bytes = 0
    for path, value in data.items():
        size_bytes = jax_tree_size_bytes(value)
        total_size_bytes += size_bytes
        print(f"    {jax.tree_util.keystr(path):24}: {gb(size_bytes):7.01f}GB")
        record[f"state_{path}_gb"] = gb(size_bytes)
    print(f"  => total size                   : {gb(total_size_bytes):7.01f}GB")

    return record


def log_memory_usage(task_str, loop_state):
    """A wrapper over dump_memory_usage_snapshot that works with the loop_state
    maintained by mpmd_pp.train.mpmd_pp_train_step."""

    loop_state_to_log = {
        "params": tuple(stage_state.params for stage_state in loop_state["state"]),
        "opt_state": tuple(
            stage_state.opt_state for stage_state in loop_state["state"]
        ),
        "grads": tuple(loop_state["grads"]),
        "bf16_params": tuple(loop_state["bf16_params"]),
        "fwd_inputs": tuple(tuple(x) for x in loop_state["fwd_inputs"]),
        "stashed": tuple(tuple(x) for x in loop_state["stashed"]),
        "bwd_inputs": tuple(tuple(x) for x in loop_state["bwd_inputs"]),
    }

    jax.block_until_ready(loop_state_to_log)
    record = _dump_memory_usage_snapshot(loop_state_to_log)

    if task_str == "start":
        print("MEM,name," + ",".join(k for k in sorted(record.keys())), flush=True)
    print(
        f"MEM,{task_str}," + ",".join(str(record[k]) for k in sorted(record.keys())),
        flush=True,
    )


# ==== Section metadata utils ==== #
# The functions below allow us to dump the jaxpr, StableHLO, HLO, and
# input/output shardings for our section functions.


# ---- File utils ---- #


def _section_dir_path(section_name, base_dir):
    """Get the directory in which to save artifacts for the section named by
    section_name."""
    section_kind, logical_stage_idx = section_name
    section_dir_name = f"section(kind={section_kind},logical_idx={logical_stage_idx})"
    return os.path.join(base_dir, section_dir_name)


def _dump_text(path, text):
    """Dump text into a file at path, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ---- IR dumping ---- #

SECTION_NAME_TO_IRS = {}


def get_jaxpr(jitted_fn, *args, **kwargs):
    jax.config.update("jax_pprint_use_color", False)
    return str(jax.make_jaxpr(jitted_fn)(*args, **kwargs).jaxpr)


def get_unoptimized_stable_hlo(jitted_fn, *args, **kwargs):
    return jitted_fn.lower(*args, **kwargs).as_text()


def get_optimized_hlo(jitted_fn, *args, **kwargs):
    return (
        jitted_fn.lower(*args, **kwargs)
        .compile()
        .runtime_executable()
        .hlo_modules()[0]
        .to_string()
    )


def _dump_single_section_irs(section_name, section_irs, base_dir):
    """Dump the IRs for a section into a directory."""
    section_dir_path = _section_dir_path(section_name, base_dir)
    for ir_name, ir_text in section_irs.items():
        _dump_text(os.path.join(section_dir_path, f"{ir_name}.txt"), ir_text)


# ---- PSpec dumping ---- #


SECTION_NAME_TO_INPUT_SHAPES_AND_PSPECS = {}
SECTION_NAME_TO_OUTPUT_SHAPES_AND_PSPECS = {}


class ShapeAndPspec:
    def __init__(self, shape, pspec):
        self.shape = shape
        self.pspec = pspec


def shapes_and_pspecs_pytree_to_str(shapes_and_pspecs_pytree):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(
        shapes_and_pspecs_pytree
    )
    lines = [f"treedef: {treedef}", "leaves:"]

    for path, leaf in leaves_with_paths:
        key_path = jax.tree_util.keystr(path)

        shape, leaf_pspec = (
            (leaf.shape, leaf.pspec)
            if isinstance(leaf, ShapeAndPspec)
            else (None, None)
        )

        lines.append(f"  {key_path}: shape={shape} pspec={leaf_pspec}")

    return "\n".join(lines)


def _dump_section_pspec(
    section_name,
    shapes_and_pspecs_filename,
    section_shapes_and_pspecs,
    base_dir,
):
    """Dump the string representation of a PyTree of shapes and pspecs to a file."""
    section_dir_path = _section_dir_path(section_name, base_dir)
    _dump_text(
        os.path.join(section_dir_path, shapes_and_pspecs_filename),
        shapes_and_pspecs_pytree_to_str(section_shapes_and_pspecs),
    )


def debug_info_section_fn_wrapper(
    section_name,
    section_fn,
    base_dir,
):
    """Wraps a given section_fn so that when first called, its IRs and in/out
    shardings are dumped to files inside base_dir."""

    def get_shape_and_pspecs(x):
        pspec = None
        if hasattr(x, "sharding") and isinstance(x.sharding, NamedSharding):
            pspec = x.sharding.spec

        shape = tuple(x.shape) if hasattr(x, "shape") else None
        return ShapeAndPspec(shape, pspec)

    @wraps(section_fn)
    def wrapped_section_fn(*args):
        # Capture the input sharding specs.
        if section_name not in SECTION_NAME_TO_INPUT_SHAPES_AND_PSPECS:
            input_shapes_and_pspecs = jax.tree.map(get_shape_and_pspecs, args)
            SECTION_NAME_TO_INPUT_SHAPES_AND_PSPECS[section_name] = (
                input_shapes_and_pspecs
            )
            _dump_section_pspec(
                section_name,
                "input_shapes_and_pspecs.txt",
                input_shapes_and_pspecs,
                base_dir,
            )

        # Capture the section_fn IRs.
        if section_name not in SECTION_NAME_TO_IRS:
            section_irs = {
                "jaxpr": get_jaxpr(section_fn, *args),
                "stable_hlo": get_unoptimized_stable_hlo(section_fn, *args),
                "optimized_hlo": get_optimized_hlo(section_fn, *args),
            }
            SECTION_NAME_TO_IRS[section_name] = section_irs
            _dump_single_section_irs(
                section_name,
                section_irs,
                base_dir,
            )

        # Compute the section outputs.
        section_outputs = section_fn(*args)

        # Capture the output sharding specs.
        if section_name not in SECTION_NAME_TO_OUTPUT_SHAPES_AND_PSPECS:
            output_shapes_and_pspecs = jax.tree.map(
                get_shape_and_pspecs, section_outputs
            )
            SECTION_NAME_TO_OUTPUT_SHAPES_AND_PSPECS[section_name] = (
                output_shapes_and_pspecs
            )
            _dump_section_pspec(
                section_name,
                "output_shapes_and_pspecs.txt",
                output_shapes_and_pspecs,
                base_dir,
            )

        return section_outputs

    return wrapped_section_fn
