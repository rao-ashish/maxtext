"""Train step implementation for experimental MPMD pipeline parallelism.

Some naming conventions generally followed in this code:
- A 'logical stage' is a contiguous chunk of model layers.
- A 'physical stage' / 'stage mesh' is a slice of our global mesh along the
    "stage" axis. Each logical stage executes on one physical stage. One
    physical stage might be responsible for executing several logical stages
    (this happens when we use circular repeat).
- A 'task' is a tuple (microbatch_idx, (section_kind, logical_stage_idx)) (see
    schedules.py).
- A 'section' is a function that executes initialization / forward / backward /
    optimizer update for some specific logical stage.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from flax import linen as nn

import optax

from MaxText.mpmd_pp import transformer
from MaxText.mpmd_pp.utils import (
    sharding_utils,
    vjp_utils,
    jaxpr_transforms,
    debug_utils,
)
from MaxText.mpmd_pp.schedules import SectionKind, task_to_str, make_pipeline_schedule


# ==== Building pipeline sections ==== #


def make_init_section(config, model, logical_stage_idx, stage_state_shardings):
    """Make the section that initializes various values needed for a
    pipeline-parallel train step. This section will run once for every
    logical_stage_idx on every invocation of the train step.

    Returns:
        _init_step: A jitted function with signature

            (stage_params) ->
            (stage_grads_acc, microbatch_ones, microbatch_idx_consts, bf16_params)
    """

    num_microbatches = config.num_pipeline_microbatches
    dp_factor = model.mesh.shape["data"]
    stage_mesh = sharding_utils.logical_stage_idx_to_mesh(model.mesh, logical_stage_idx)

    init_out_shardings = (
        stage_state_shardings.params,
        None,
        None,
        stage_state_shardings.params,
    )

    @partial(jax.jit, out_shardings=init_out_shardings)
    def _init_step(stage_params):
        stage_grads_acc = jax.tree.map(
            lambda param: jnp.zeros_like(param),
            stage_params,
        )

        microbatch_ones = tuple(
            jnp.ones(
                (dp_factor,),
                dtype=jnp.float32,
                device=NamedSharding(stage_mesh, PartitionSpec("data")),
            )
            for microbatch_idx in range(num_microbatches)
        )
        microbatch_idx_consts = tuple(
            jnp.full(
                (),
                microbatch_idx,
                device=NamedSharding(stage_mesh, PartitionSpec()),
            )
            for microbatch_idx in range(num_microbatches)
        )

        bf16_params = jax.tree.map(lambda p: p.astype(jnp.bfloat16), stage_params)

        return stage_grads_acc, microbatch_ones, microbatch_idx_consts, bf16_params

    return _init_step


def _model_fwd_and_bwd(model, stage_index):
    """Helper function used to create forward and backward stages.

    Returns:
        fwd: A non-jitted function with signature

            (params, input_activations, microbatch_data, rng) -> fwd_out

            Where fwd_out is (params, stashed, activations) if this is not
            the last stage, and (params, loss, aux) if it is.

        bwd: A non-jitted function with signature

            (stashed, out_cot, params) -> bwd_out

            Where bwd_out is (grads_acc, in_cotangents) if this is not
            the last stage, and grads_acc alone if it is.
    """

    num_stages = model.num_logical_stages
    fwd, bwd = vjp_utils.fwd_and_bwd(
        partial(transformer.forward, model, stage_index),
        # Take vjp wrt params and input activations
        argnums=(0, 1),
        # Caller saves params (to avoid duplicating in vjp residuals)
        caller_saved_among_argnums=(True, False),
        has_aux=(stage_index == num_stages - 1),
        jitted=False,
    )
    fwd.__name__ = f"forward{stage_index}"
    bwd.__name__ = f"backward{stage_index}"

    # DP-vmap-trick: vmap both fwd and bwd
    # TODO: Pass spmd_axis="data" to vmap?
    fwd = jax.vmap(
        fwd,
        in_axes=(0, 0, 0, None),  # params, acts, data, rng
        out_axes=0,  # acts, stashed, aux?
        spmd_axis_name="data",
    )
    bwd = jax.vmap(
        bwd,
        in_axes=(0, 0, 0),  # stashed, out_cot, params
        out_axes=0,  # grads, in_cot
        spmd_axis_name="data",
    )

    return fwd, bwd


def make_forward_section(model, logical_stage_idx, fwd_fn, stage_state_shardings):
    """Build the forward section for a specific logical stage of a model.

    Args:
        model: An instance of mpmd_pp.transformer.PipelineParallelTransformer.
        logical_stage_idx: The logical stage for which we are building the
            forward section.
        fwd_fn: The fwd_fn returned by _model_fwd_and_bwd.
        stage_state_shardings: A PyTree giving the shardings for all values in
            the train state.

    Returns:
        _forward_section: A function with signature

            (bf16_params, params, input_activations, data, rng, microbatch_idx)
            -> fwd_out

            Where fwd_out is (stashed, activations) if this is not
            the last stage, and (loss, aux) if it is.
    """

    if logical_stage_idx != model.num_logical_stages - 1:
        fwd_out_shardings = (None, None)
    else:
        fwd_out_shardings = (None, None, None)

    @partial(jax.jit, out_shardings=fwd_out_shardings, donate_argnums=(1, 2))
    def _forward_section(
        bf16_params,
        params,
        input_activations,
        data,
        rng,
        microbatch_idx,
    ):
        def fwd_step_with_casts(
            params,
            input_activations,
            data,
            rng,
            microbatch_idx,
            fwd_fn=fwd_fn,
        ):
            microbatch_data = jax.tree.map(lambda x: x[microbatch_idx], data)
            res = fwd_fn(params, input_activations, microbatch_data, rng)
            return res

        args = (params, input_activations, data, rng, microbatch_idx)
        return jaxpr_transforms.remove_casts(fwd_step_with_casts, *args)(
            *args, bf16_params
        )

    return _forward_section


def make_backward_section(logical_stage_idx, bwd_fn, stage_state_shardings):
    """Build the backward section for a specific logical stage of a model.

    Args:
        logical_stage_idx: The logical stage for which we are building the
            backward section.
        bwd_fn: The bwd_fn returned by _model_fwd_and_bwd.
        stage_state_shardings: A PyTree giving the shardings for all values in
            the train state.

    Returns:
        _backward_section: A function with signature

            (bf16_params, params, stashed, out_cot, grads_acc, microbatch_idx)
            -> bwd_out

            Where bwd_out is (params, grads_acc, in_cotangents) if this is not
            the last stage, and (params, grads_acc) if it is.
    """

    if logical_stage_idx != 0:
        bwd_out_shardings = (stage_state_shardings.params, None)
    else:
        bwd_out_shardings = stage_state_shardings.params

    @partial(jax.jit, out_shardings=bwd_out_shardings, donate_argnums=(1, 2, 3, 4))
    def _backward_section(
        bf16_params,
        params,
        stashed,
        out_cot,
        grads_acc,
        microbatch_idx,
    ):
        def bwd_stage_with_casts(
            params,
            stashed,
            out_cot,
            grads_acc,
            microbatch_idx,
            bwd_fn=bwd_fn,
            logical_stage_idx=logical_stage_idx,
        ):
            grads, in_cot = bwd_fn(stashed, out_cot, params)
            grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
            if logical_stage_idx != 0:
                return grads_acc, in_cot
            return grads_acc

        args = (params, stashed, out_cot, grads_acc, microbatch_idx)
        return jaxpr_transforms.remove_casts(bwd_stage_with_casts, *args)(
            *args, bf16_params
        )

    return _backward_section


def make_update_section(model, logical_stage_idx, stage_state, stage_state_shardings):
    """Build the update section for a specific logical stage of a model.

    Returns:
        _update_section: A function with signature

            (stage_state, stage_grads_acc) -> new_stage_state
    """
    tx = stage_state.tx
    stage_mesh = sharding_utils.logical_stage_idx_to_mesh(model.mesh, logical_stage_idx)

    @partial(jax.jit, donate_argnums=(0, 1), out_shardings=stage_state_shardings)
    def _update_section(stage_state, stage_grads):
        assert nn.fp8_ops.OVERWRITE_WITH_GRADIENT not in stage_grads

        params_specs = jax.tree.map(lambda s: s.spec, stage_state_shardings.params)
        opt_state_specs = jax.tree.map(
            lambda s: s.spec, stage_state_shardings.opt_state
        )
        grad_specs = params_specs
        io_specs = (params_specs, opt_state_specs, grad_specs)

        @partial(
            jax.shard_map,
            in_specs=io_specs,
            out_specs=io_specs,
            mesh=stage_mesh,
            check_vma=False,
        )
        def shard_mapped_update(ps, os, gs):
            reduced_grads = jax.lax.psum(gs, axis_name="data")
            updates, new_opt_state = tx.update(reduced_grads, os, ps)
            new_params = optax.apply_updates(ps, updates)
            return new_params, new_opt_state, reduced_grads

        new_params, new_opt_state, reduced_grads = shard_mapped_update(
            stage_state.params, stage_state.opt_state, stage_grads
        )

        return stage_state.replace(
            step=stage_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

    return _update_section


def make_fused_backward_update_section(bwd_fn, update_fn):
    """Build a section that executes a backward pass and an optimizer update
    for a logical stage in one jitted function.

    Returns:
        _fused_bwd_update_step: A function with signature

            (bf16_params, stage_state, stashed, out_cot, grads_acc, microbatch_idx)
            -> new_stage_state
    """

    @partial(jax.jit, donate_argnums=(1, 2, 3, 4))
    def _fused_bwd_update_step(
        bf16_params,
        stage_state,
        stashed,
        out_cot,
        grads_acc,
        microbatch_idx,
    ):
        grads_acc = bwd_fn(
            bf16_params,
            stage_state.params,
            stashed,
            out_cot,
            grads_acc,
            microbatch_idx,
        )
        return update_fn(stage_state, grads_acc)

    return _fused_bwd_update_step


def make_section_name_to_section(config, model, state):
    """Build a dictionary mapping tuples of (section_kind, logical_stage_idx)
    to functions that execute those sections."""

    section_name_to_section = {}

    for logical_stage_idx, stage_state in enumerate(state):
        # We can optimize for compile times by making every logical stage on
        # the same physical stage reuse the same init / fwd / bwd / update
        # functions.
        if (
            config.mpmd_pp_optimize_for_compile_times
            and logical_stage_idx != 0
            and logical_stage_idx != model.num_logical_stages - 1
        ):
            physical_stage_idx = logical_stage_idx % model.num_physical_stages
            first_matching_logical_idx = (
                physical_stage_idx
                if physical_stage_idx != 0
                else model.num_physical_stages
            )
            if logical_stage_idx > first_matching_logical_idx:
                for section_kind in [
                    SectionKind.STAGE_INIT,
                    SectionKind.FORWARD,
                    SectionKind.BACKWARD,
                    SectionKind.UPDATE,
                ]:
                    section_name_to_section[(section_kind, logical_stage_idx)] = (
                        section_name_to_section[
                            (section_kind, first_matching_logical_idx)
                        ]
                    )
                continue

        stage_state_shardings = jax.tree.map(lambda s: s.sharding, stage_state)

        section_name_to_section[(SectionKind.STAGE_INIT, logical_stage_idx)] = (
            make_init_section(config, model, logical_stage_idx, stage_state_shardings)
        )

        fwd, bwd = _model_fwd_and_bwd(model, logical_stage_idx)

        section_name_to_section[(SectionKind.FORWARD, logical_stage_idx)] = (
            make_forward_section(model, logical_stage_idx, fwd, stage_state_shardings)
        )

        bwd_fn = make_backward_section(logical_stage_idx, bwd, stage_state_shardings)
        section_name_to_section[(SectionKind.BACKWARD, logical_stage_idx)] = bwd_fn

        update_fn = make_update_section(
            model,
            logical_stage_idx,
            stage_state,
            stage_state_shardings,
        )

        if logical_stage_idx == 0:
            section_name_to_section[
                (SectionKind.FUSED_BACKWARD_UPDATE, logical_stage_idx)
            ] = make_fused_backward_update_section(bwd_fn, update_fn)
        else:
            section_name_to_section[(SectionKind.UPDATE, logical_stage_idx)] = update_fn

    # Wrap all sections with debug_info_section_fn_wrapper.
    if config.mpmd_pp_section_fn_debug_info_dir:
        for section_name, section_fn in section_name_to_section.items():
            section_name_to_section[section_name] = (
                debug_utils.debug_info_section_fn_wrapper(
                    section_name,
                    section_fn,
                    base_dir=config.mpmd_pp_section_fn_debug_info_dir,
                )
            )

    return section_name_to_section


# ==== Train step ==== #


@partial(jax.jit, static_argnums=(1, 2), donate_argnums=(0,))
def _reshape_reshard_data(data, num_microbatches, dp_factor):
    """Helper function for prepare_input_data."""

    def reshape_reshard_arr(arr):
        arr = arr.reshape((num_microbatches, dp_factor, -1, *arr.shape[1:]))
        return nn.with_logical_constraint(
            arr,
            (
                None,
                "activation_embed_and_logits_batch",
            ),
        )

    return jax.tree.map(reshape_reshard_arr, data)


def prepare_input_data(
    data,
    global_mesh,
    num_microbatches,
    dp_factor,
    num_logical_stages,
    num_physical_stages,
):
    """Reshape, reshard, and replicate input tokens, segmentation mask,
    targets, etc across stages for MPMD PP train step.

    This is done to make the implementation simpler; ideally these inputs
    would only be copied onto and given to ranks that really need them (the
    first and last pipeline stages).

    For each pipeline stage's replica, the tokens are reshaped to
    (num_microbatches, dp_factor, per_device_microbatch_size).
    """
    data = _reshape_reshard_data(data, num_microbatches, dp_factor)

    def prepare_data_on_stage(logical_stage_idx):
        after_mesh_axis_removed = jax.tree.map(
            lambda x: sharding_utils.remove_mesh_axis(
                x,
                mesh_axis_name="stage",
                mesh_axis_slice_idx=logical_stage_idx % num_physical_stages,
            ),
            data,
        )
        # TODO: Do we really need this device_put?
        target_shardings = jax.tree.map(
            lambda x: sharding_utils.sharding_with_mesh(
                x.sharding,
                sharding_utils.logical_stage_idx_to_mesh(
                    global_mesh, logical_stage_idx
                ),
            ),
            after_mesh_axis_removed,
        )
        return jax.device_put(
            after_mesh_axis_removed,
            target_shardings,
        )

    return tuple(
        prepare_data_on_stage(logical_stage_idx)
        for logical_stage_idx in range(num_logical_stages)
    )


def prepare_dropout_rngs(dropout_rng, num_logical_stages, global_mesh):
    """Prepare new dropout RNGs for each pipeline stage."""

    num_physical_stages = global_mesh.shape["stage"]

    dropout_rngs = jax.device_put(
        jax.random.split(dropout_rng, num_logical_stages),
        NamedSharding(global_mesh, PartitionSpec()),
    )

    return [
        sharding_utils.remove_mesh_axis(
            rng,
            mesh_axis_name="stage",
            mesh_axis_slice_idx=logical_stage_idx % num_physical_stages,
        )
        for logical_stage_idx, rng in enumerate(dropout_rngs)
    ]


def init_loop_state(
    data,
    state,
    global_mesh,
    dropout_rng,
    num_logical_stages,
    num_microbatches,
):
    """Initialize a loop state that we read and update as we execute tasks
    as part of our pipeline parallel train step."""

    return {
        # Create a new dropout RNG for each logical stage.
        "dropout_rngs": prepare_dropout_rngs(
            dropout_rng, num_logical_stages, global_mesh
        ),
        # Reshape, reshard, and replicate input data across stages.
        "replicated_data": prepare_input_data(
            data,
            global_mesh=global_mesh,
            num_microbatches=num_microbatches,
            dp_factor=global_mesh.shape["data"],
            num_logical_stages=num_logical_stages,
            num_physical_stages=global_mesh.shape["stage"],
        ),
        # Forward pass inputs (primals), stashed activations fo backward pass,
        # backward pass inputs (cotanges).
        "fwd_inputs": [
            [None for _ in range(num_microbatches)] for _ in range(num_logical_stages)
        ],
        "stashed": [
            [None for _ in range(num_microbatches)] for _ in range(num_logical_stages)
        ],
        "bwd_inputs": [
            [None for _ in range(num_microbatches)] for _ in range(num_logical_stages)
        ],
        "microbatch_idx_consts_by_stage": [None for _ in range(num_logical_stages)],
        "state": list(state),
        "bf16_params": [None for _ in range(num_logical_stages)],
        "grads": [None for _ in range(num_logical_stages)],
        "loss": [None for _ in range(num_microbatches)],
        "aux": [None for _ in range(num_microbatches)],
    }


@partial(jax.jit, donate_argnums=(0, 1))
def stack_metrics(loss, aux):
    """Aggregate loss and aux values across multiple microbatches."""

    def _stack_mean(x):
        return jnp.mean(jnp.stack(x), axis=(0, 1))

    loss = _stack_mean(loss)
    aux = jax.tree.map(lambda *xs: _stack_mean(xs), *aux)
    return loss, aux


# ==== Executing pipeline sections ==== #
# The functions below take in the loop state (+ some auxiliary info), execute
# a task, and update the loop state.


def execute_init_task(
    logical_stage_idx,
    num_logical_stages,
    section_fn,
    loop_state,
    global_mesh,
):
    """Execute an init task given the current loop state, and update the loop
    state."""
    (
        loop_state["grads"][logical_stage_idx],
        microbatch_const_ones,
        loop_state["microbatch_idx_consts_by_stage"][logical_stage_idx],
        loop_state["bf16_params"][logical_stage_idx],
    ) = section_fn(loop_state["state"][logical_stage_idx].params)

    if logical_stage_idx == num_logical_stages - 1:
        for microbatch_idx, const_ones in enumerate(microbatch_const_ones):
            loop_state["bwd_inputs"][logical_stage_idx][microbatch_idx] = const_ones


def execute_forward_task(
    microbatch_idx,
    logical_stage_idx,
    num_logical_stages,
    section_fn,
    loop_state,
    global_mesh,
):
    """Execute a forward task given the current loop state, and update the loop
    state."""

    stage_input_activations = loop_state["fwd_inputs"][logical_stage_idx][
        microbatch_idx
    ]
    loop_state["fwd_inputs"][logical_stage_idx][microbatch_idx] = None
    if stage_input_activations is None and logical_stage_idx == 0:
        stage_input_activations = loop_state["replicated_data"][logical_stage_idx][
            "inputs"
        ][microbatch_idx]

    fwd_out = section_fn(
        loop_state["bf16_params"][logical_stage_idx],
        loop_state["state"][logical_stage_idx].params,
        stage_input_activations,
        loop_state["replicated_data"][logical_stage_idx],
        loop_state["dropout_rngs"][logical_stage_idx],
        loop_state["microbatch_idx_consts_by_stage"][logical_stage_idx][microbatch_idx],
    )

    del stage_input_activations

    loop_state["stashed"][logical_stage_idx][microbatch_idx] = fwd_out[0]

    if logical_stage_idx == num_logical_stages - 1:
        loop_state["loss"][microbatch_idx] = fwd_out[1]
        loop_state["aux"][microbatch_idx] = fwd_out[2]
    else:
        next_fwd_input = jax.device_put(
            fwd_out[1],
            sharding_utils.sharding_with_mesh(
                fwd_out[1].sharding,
                sharding_utils.logical_stage_idx_to_mesh(
                    global_mesh, logical_stage_idx + 1
                ),
            ),
            donate=True,
        )

        loop_state["fwd_inputs"][logical_stage_idx + 1][microbatch_idx] = next_fwd_input


def execute_backward_task(
    microbatch_idx,
    logical_stage_idx,
    section_fn,
    loop_state,
    global_mesh,
):
    """Execute a backward task given the current loop state, and update the loop
    state."""

    stashed = loop_state["stashed"][logical_stage_idx][microbatch_idx]
    out_cot = loop_state["bwd_inputs"][logical_stage_idx][microbatch_idx]
    loop_state["stashed"][logical_stage_idx][microbatch_idx] = None
    loop_state["bwd_inputs"][logical_stage_idx][microbatch_idx] = None

    bwd_out = section_fn(
        loop_state["bf16_params"][logical_stage_idx],
        loop_state["state"][logical_stage_idx].params,
        stashed,
        out_cot,
        loop_state["grads"][logical_stage_idx],
        loop_state["microbatch_idx_consts_by_stage"][logical_stage_idx][microbatch_idx],
    )

    del stashed
    del out_cot

    if logical_stage_idx == 0:
        loop_state["grads"][logical_stage_idx] = bwd_out
    else:
        loop_state["grads"][logical_stage_idx] = bwd_out[0]
        next_bwd_input = jax.device_put(
            bwd_out[1],
            sharding_utils.sharding_with_mesh(
                bwd_out[1].sharding,
                sharding_utils.logical_stage_idx_to_mesh(
                    global_mesh, logical_stage_idx - 1
                ),
            ),
            donate=True,
        )

        loop_state["bwd_inputs"][logical_stage_idx - 1][microbatch_idx] = next_bwd_input


def execute_fused_backward_update_task(
    microbatch_idx,
    logical_stage_idx,
    section_fn,
    loop_state,
):
    """Execute a fused backward/update task given the current loop state, and
    update the loop state."""

    stashed = loop_state["stashed"][logical_stage_idx][microbatch_idx]
    out_cot = loop_state["bwd_inputs"][logical_stage_idx][microbatch_idx]
    loop_state["stashed"][logical_stage_idx][microbatch_idx] = None
    loop_state["bwd_inputs"][logical_stage_idx][microbatch_idx] = None

    loop_state["state"][logical_stage_idx] = section_fn(
        loop_state["bf16_params"][logical_stage_idx],
        loop_state["state"][logical_stage_idx],
        stashed,
        out_cot,
        loop_state["grads"][logical_stage_idx],
        loop_state["microbatch_idx_consts_by_stage"][logical_stage_idx][microbatch_idx],
    )

    loop_state["grads"][logical_stage_idx] = None
    loop_state["bf16_params"][logical_stage_idx] = None

    del stashed
    del out_cot


def execute_update_task(
    logical_stage_idx,
    section_fn,
    loop_state,
):
    """Execute an update task given the current loop state, and update the loop
    state."""

    loop_state["state"][logical_stage_idx] = section_fn(
        loop_state["state"][logical_stage_idx],
        loop_state["grads"][logical_stage_idx],
    )
    loop_state["grads"][logical_stage_idx] = None
    loop_state["bf16_params"][logical_stage_idx] = None


def execute_task(
    task,
    section_name_to_section,
    loop_state,
    global_mesh,
    num_logical_stages,
):
    """Execute a task given the current loop state, and update the loop state
    according to the task's outputs."""

    microbatch_idx, section_name = task
    section_fn = section_name_to_section[section_name]
    section_kind, logical_stage_idx = section_name
    stage_mesh = sharding_utils.logical_stage_idx_to_mesh(
        global_mesh, logical_stage_idx
    )

    # print(f"Executing task: {microbatch_idx=} {section_kind=} {logical_stage_idx=}")

    with jax.set_mesh(stage_mesh):
        match section_kind:
            case SectionKind.STAGE_INIT:
                execute_init_task(
                    logical_stage_idx,
                    num_logical_stages,
                    section_fn,
                    loop_state,
                    global_mesh,
                )

            case SectionKind.FORWARD:
                execute_forward_task(
                    microbatch_idx,
                    logical_stage_idx,
                    num_logical_stages,
                    section_fn,
                    loop_state,
                    global_mesh,
                )

            case SectionKind.BACKWARD:
                execute_backward_task(
                    microbatch_idx,
                    logical_stage_idx,
                    section_fn,
                    loop_state,
                    global_mesh,
                )

            case SectionKind.FUSED_BACKWARD_UPDATE:
                execute_fused_backward_update_task(
                    microbatch_idx,
                    logical_stage_idx,
                    section_fn,
                    loop_state,
                )

            case SectionKind.UPDATE:
                execute_update_task(
                    logical_stage_idx,
                    section_fn,
                    loop_state,
                )

            case _:
                raise ValueError(f"Unrecognized section_kind {section_kind}.")


# ==== Main train step function ==== #


def train_step(
    model,
    config,
    state_mesh_shardings,
    params_shardings,
    section_name_to_section,
    state,
    data,
    dropout_rng,
):
    """Eager-mode MPMD implementation of pipeline-parallel train step.

    Launches different pipeline stages as different jitted functions executing on
    different devices, and moves data between pipeline stages using
    `jax.device_put`.
    """

    num_logical_stages = model.num_logical_stages
    num_physical_stages = model.num_physical_stages
    num_microbatches = config.num_pipeline_microbatches

    pipeline_schedule = make_pipeline_schedule(
        num_logical_stages,
        num_physical_stages,
        num_microbatches,
        config.mpmd_pp_schedule,
    )

    loop_state = init_loop_state(
        data,
        state,
        model.mesh,
        dropout_rng,
        num_logical_stages,
        num_microbatches,
    )
    # Delete references to things captured by loop_state.
    del data, state, dropout_rng

    if config.mpmd_pp_print_memory_usage:
        debug_utils.log_memory_usage("start", loop_state)

    for task_idx, task in enumerate(pipeline_schedule):
        execute_task(
            task,
            section_name_to_section,
            loop_state,
            model.mesh,
            num_logical_stages,
        )

        if config.mpmd_pp_print_memory_usage:
            debug_utils.log_memory_usage(task_to_str(task), loop_state)

    with jax.set_mesh(
        sharding_utils.logical_stage_idx_to_mesh(model.mesh, num_logical_stages - 1)
    ):
        loss, aux = stack_metrics(loop_state["loss"], loop_state["aux"])

    scalar_metrics = {
        "learning/loss": loss,
        "learning/moe_lb_loss": aux["moe_lb_loss"],
        "learning/total_weights": aux["total_weights"],
    }
    metrics = {
        "scalar": scalar_metrics,
        "scalars": {},
    }
    return loop_state["state"], metrics


def make_train_step(
    model,
    config,
    state_mesh_shardings,
    params_shardings,
    initial_state,
):
    """Build an MPMD pipeline-parallel train step function for the specified
    model and config."""

    section_name_to_section = make_section_name_to_section(
        config,
        model,
        initial_state,
    )
    return partial(
        train_step,
        model,
        config,
        state_mesh_shardings,
        params_shardings,
        section_name_to_section,
    )
