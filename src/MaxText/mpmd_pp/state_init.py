"""Train state initialization for MPMD pipeline parallelism."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from flax import linen as nn
from flax.training import train_state
from flax.linen import partitioning as nn_partitioning

from MaxText.mpmd_pp import transformer
from MaxText.mpmd_pp.utils import sharding_utils

from maxtext.utils import max_utils
from maxtext.utils import max_logging


def _prepend_data_axis_sharding(sharding):
    """Add the 'data' mesh axis as the leading entry of the given sharding's
    partition spec."""

    if not isinstance(sharding, NamedSharding):
        return sharding

    return NamedSharding(
        sharding.mesh,
        PartitionSpec("data", *sharding.spec),
        memory_kind=sharding.memory_kind,
    )


def _broadcast_tree_over_data_axis(tree, data_dim):
    """Broadcast arrays inside a given pytree to have a leading dimension with
    size data_dim."""

    def _broadcast_leaf(x):
        if isinstance(x, jax.Array):
            return jnp.broadcast_to(jnp.expand_dims(x, axis=0), (data_dim, *x.shape))

        if isinstance(x, (int, float)):
            x = jnp.array(x)
            return jnp.broadcast_to(jnp.expand_dims(x, axis=0), (data_dim, *x.shape))

        return x

    return jax.tree.map(_broadcast_leaf, tree)


def init_state(model, tx, config, rng_key):
    """MPMD pipeline parallelism train state initialization function."""

    assert isinstance(model, transformer.PipelineParallelTransformer)

    input_shape = (
        config.micro_batch_size_to_train_on,
        config.max_target_length,
    )

    activation_shape = (
        *input_shape,
        config.base_emb_dim,
    )

    state = [None] * model.num_logical_stages
    stage_rng_keys = jax.random.split(rng_key, model.num_logical_stages)

    shared_apply_fn = partial(transformer.forward, model=model)

    for physical_idx in range(model.num_physical_stages):
        max_logging.log(
            "==== MPMD_PP.INIT_STATE INITIALIZING FOR PROCESS_IDX = "
            f"{jax.process_index()}, PHYSICAL_IDX = {physical_idx} ===="
        )
        my_logical_stage_idxs = list(
            range(
                physical_idx,
                model.num_logical_stages,
                model.num_physical_stages,
            )
        )
        stage_mesh = sharding_utils.physical_stage_idx_to_mesh(model.mesh, physical_idx)

        # Derive shardings from logical axis rules.
        input_logical_spec = PartitionSpec(*config.input_data_sharding_logical_axes)
        input_sharding = nn.logical_to_mesh_sharding(
            input_logical_spec, stage_mesh, config.logical_axis_rules
        )

        activation_logical_spec = PartitionSpec(
            "activation_embed_and_logits_batch_outside_vmap",
            "activation_length",
            "activation_embed",
        )
        activation_sharding = nn.logical_to_mesh_sharding(
            activation_logical_spec, stage_mesh, config.logical_axis_rules
        )

        dummy_input = (
            jnp.ones(input_shape, dtype=jnp.int32, device=input_sharding),  # inputs
            jnp.ones(
                activation_shape, dtype=config.dtype, device=activation_sharding
            ),  # activations
            jnp.ones(
                input_shape, dtype=jnp.int32, device=input_sharding
            ),  # decoder_segment_ids
            jnp.ones(
                input_shape, dtype=jnp.int32, device=input_sharding
            ),  # decoder_positions
            jnp.ones(
                input_shape, dtype=jnp.int32, device=input_sharding
            ),  # decoder_targets
            jnp.ones(
                input_shape, dtype=jnp.int32, device=input_sharding
            ),  # decoder_targets_segmentation
        )

        with jax.set_mesh(stage_mesh):
            # Create new init rngs for each logical stage.
            init_rngs = tuple(
                jax.tree.map(
                    lambda x: jax.device_put(
                        x, NamedSharding(stage_mesh, PartitionSpec())
                    ),
                    {
                        "params": stage_rng_keys[logical_idx],
                        "dropout": stage_rng_keys[logical_idx],
                        "aqt": stage_rng_keys[logical_idx],
                    },
                )
                for logical_idx in my_logical_stage_idxs
            )

            # Single jitted function that initializes all logical stages for
            # this physical stage. _raw_init_fn produces the stage state without
            # broadcast / reshard.
            def _raw_init_fn(
                rngs,
                inputs,
                activations,
                decoder_segment_ids,
                decoder_positions,
                decoder_targets,
                decoder_targets_segmentation,
            ):
                out_train_states = []
                for i, logical_idx in enumerate(my_logical_stage_idxs):
                    stage_params = model.init(
                        rngs[i],
                        logical_idx,
                        inputs if logical_idx == 0 else activations,
                        decoder_segment_ids,
                        decoder_positions,
                        decoder_targets,
                        decoder_targets_segmentation,
                        method=model._stage,
                    )
                    out_train_states.append(
                        train_state.TrainState.create(
                            apply_fn=shared_apply_fn,
                            params=stage_params,
                            tx=tx,
                        )
                    )
                return tuple(out_train_states)

            data_dim = stage_mesh.shape["data"]

            def _init_fn(*args):
                out_train_states = _raw_init_fn(*args)
                return _broadcast_tree_over_data_axis(out_train_states, data_dim)

            with nn_partitioning.axis_rules(config.logical_axis_rules):
                abstract_state = jax.eval_shape(_raw_init_fn, init_rngs, *dummy_input)
                state_logical_annotations = nn.get_partition_spec(abstract_state)
                raw_state_mesh_shardings = nn.logical_to_mesh_sharding(
                    state_logical_annotations,
                    stage_mesh,
                    config.logical_axis_rules,
                )
                state_mesh_shardings = jax.tree.map(
                    lambda s: _prepend_data_axis_sharding(s),
                    raw_state_mesh_shardings,
                )

                my_logical_stage_states = jax.jit(
                    _init_fn,
                    out_shardings=state_mesh_shardings,
                )(init_rngs, *dummy_input)

                jax.block_until_ready(my_logical_stage_states)
                max_logging.log(
                    "==== MPMD_PP.INIT_STATE DONE WITH PROCESS_IDX = "
                    f"{jax.process_index()}, PHYSICAL_IDX = {physical_idx}, "
                    f"LOGICAL_IDXS = {my_logical_stage_idxs} ===="
                )

                for i, logical_stage_idx in enumerate(my_logical_stage_idxs):
                    state[logical_stage_idx] = my_logical_stage_states[i]

                del my_logical_stage_states

    state = max_utils.unbox_logicallypartioned(tuple(state))

    # Form state_mesh_annotations, state_mesh_shardings.
    state_mesh_annotations = jax.tree.map(
        lambda x: (
            x.sharding.spec
            if hasattr(x, "sharding") and hasattr(x.sharding, "spec")
            else None
        ),
        state,
    )
    state_mesh_shardings = jax.tree.map(lambda x: getattr(x, "sharding", None), state)

    return state, state_mesh_annotations, state_mesh_shardings
