"""MPMD pipeline parallelism utilities to switch between the global mesh and
the meshes for individual pipeline stages."""

import jax
from jax.sharding import NamedSharding, Mesh


def slice_mesh(global_mesh, axis_name, slice_idx):
    """Get a slice of global_mesh along the axis specified by axis_name at the
    index slice_idx."""

    axis = global_mesh.axis_names.index(axis_name)
    devices = global_mesh.devices.take(indices=slice_idx, axis=axis)
    new_axis_names = global_mesh.axis_names[:axis] + global_mesh.axis_names[axis + 1 :]
    new_axis_types = global_mesh.axis_types[:axis] + global_mesh.axis_types[axis + 1 :]
    return Mesh(devices, new_axis_names, axis_types=new_axis_types)


def logical_stage_idx_to_mesh(global_mesh, logical_stage_idx):
    """Get the physical mesh on which a particular logical_stage_idx is supposed
    to run."""

    num_physical_stages = global_mesh.shape["stage"]
    return slice_mesh(global_mesh, "stage", logical_stage_idx % num_physical_stages)


def physical_stage_idx_to_mesh(global_mesh, physical_stage_idx):
    """Get the mesh corresponding to a physical_stage_idx."""

    return slice_mesh(global_mesh, "stage", physical_stage_idx)


def sharding_with_mesh(s, mesh):
    """Given a NamedSharding object, return a NamedSharding object with the
    same PartitionSpec but with a new mesh."""

    assert isinstance(s, NamedSharding), f"expected NamedSharding, got: {s}"
    return NamedSharding(mesh, s.spec, memory_kind=s.memory_kind)


def remove_mesh_axis(
    arr,
    mesh_axis_name,
    mesh_axis_slice_idx,
):
    """Given an array replicated over the mesh axis specified by mesh_axis_name,
    return an array sharded over a slice of the input array's mesh specified by
    mesh_axis_name and mesh_axis_slice_idx.

    This does not copy the array; the output array aliases shards of the input
    array.
    """

    if not isinstance(arr, jax.Array):
        return arr

    assert isinstance(arr.sharding, NamedSharding), (
        "remove_mesh_axis only supports NamedShardings."
    )

    curr_mesh = arr.sharding.mesh
    pspec = arr.sharding.spec

    # Make sure curr_mesh contains mesh_axis_name.
    assert mesh_axis_name in curr_mesh.axis_names, (
        "Supplied array is not sharded over a mesh "
        + f"containing mesh_axis_name={mesh_axis_name}."
    )

    # Make sure that arr is replicated along mesh_axis_name.
    assert mesh_axis_name not in pspec, (
        f"Supplied array is not replicated over mesh_axis_name={mesh_axis_name}."
    )

    # Form the sharding of the output array.
    new_mesh = Mesh(
        devices=curr_mesh.devices.take(
            indices=mesh_axis_slice_idx,
            axis=curr_mesh.axis_names.index(mesh_axis_name),
        ),
        axis_names=tuple(
            name for name in curr_mesh.axis_names if name != mesh_axis_name
        ),
    )
    new_sharding = NamedSharding(new_mesh, pspec)

    # Form the new array.
    device_to_buffer = {s.device: s.data for s in arr.addressable_shards}
    ordered_arrs = []

    for device in new_mesh.devices.flat:
        if device in device_to_buffer:
            ordered_arrs.append(device_to_buffer[device])

    return jax.make_array_from_single_device_arrays(
        arr.shape,
        new_sharding,
        ordered_arrs,
        dtype=arr.dtype,
    )
