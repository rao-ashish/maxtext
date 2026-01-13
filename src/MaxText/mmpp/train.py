"""Training step and state management for mmpp.
Primarily concerned with data placement, transfers and pipelining."""

import dataclasses
from functools import partial, lru_cache
from typing import Any, Callable

from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp
import jax._src.core as core
from jax._src.named_sharding import UNSPECIFIED
from jax._src.linear_util import DebugInfo
from jax.sharding import NamedSharding, PartitionSpec, Mesh
from flax.linen import partitioning as nn_partitioning

import optax

from MaxText.mmpp import models
from MaxText.mmpp import mpmd
from MaxText.mmpp import utils
from MaxText import max_utils
from MaxText.mmpp.schedules import make_gpipe_schedule, make_jaxpp_1F1B_schedule


def model_fwd_and_bwd(model, stage_index):
  num_stages = model.num_logical_stages
  fwd, bwd = utils.fwd_and_bwd(
    partial(models.forward, model, stage_index),
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
    out_axes=0,                  # acts, stashed, aux?
    spmd_axis_name="data",
  )
  bwd = jax.vmap(
    bwd,
    in_axes=(0, 0, 0),        # stashed, out_cot, params
    out_axes=0,                  # grads, in_cot
    spmd_axis_name="data",
  )

  return fwd, bwd


@dataclasses.dataclass(frozen=True)
class ParamInfo:
  shape: Any
  dtype: Any
  sharding: Any


def make_init_stage(num_mubatches, dp_factor, params, stage_index):
  # NB: We compute param_infos outside init_stage, so it doesn't capture params!
  def make_param_info(p):
    # DP-vmap-trick: prepend DP dimension to grads_acc
    return ParamInfo(p.shape, p.dtype, p.sharding)

  param_infos = jax.tree.map(make_param_info, params)

  def init_stage(params):
    stage_mesh = mpmd.get_context().get_stage_mesh(stage_index)

    # Gradient accumulators
    def zeros_like_param(pi):
      zeros = jnp.zeros(pi.shape, dtype=pi.dtype)
      sharding = mpmd.sharding_with_mesh(pi.sharding, stage_mesh)
      return jax.lax.with_sharding_constraint(zeros, sharding)

    grads = jax.tree.map(zeros_like_param, param_infos)

    # Constants we need to feed to fwd/bwd steps throughout the training step
    mubatch_ones = tuple(
      # DP-vmap-trick: prepend DP dimension to loss_cot
      jnp.full((dp_factor,), 1.0, device=NamedSharding(stage_mesh, PartitionSpec("data")))
      for mubatch_idx in range(num_mubatches)
    )
    mubatch_idx_consts = tuple(
      jnp.full((), mubatch_idx, device=NamedSharding(stage_mesh, PartitionSpec()))
      for mubatch_idx in range(num_mubatches)
    )

    # BF-16 casted stage params.
    bf16_params = jax.tree.map(
      lambda p, pi: jax.lax.with_sharding_constraint(
        p.astype(jnp.bfloat16),
        mpmd.sharding_with_mesh(pi.sharding, stage_mesh)
      ),
      params,
      param_infos,
    )

    return grads, mubatch_ones, mubatch_idx_consts, bf16_params

  return init_stage


def fwd_stage(fwd, params, input_activations, data, rng, mubatch_idx):
  mubatch_data = jax.tree.map(lambda x: x[mubatch_idx], data)
  res = fwd(params, input_activations, mubatch_data, rng)
  return params, *res


def remove_casts(bwd_stage, params, *args):
    """
    Modify bwd_stage to remove casts of params to bf16, and take in those bf16
    values as inputs instead.

    bwd_stage is assumed to have signature:
        `params, *args -> *outs`.

    The returned callable will have signature:
        `params, *args, bf16_params -> *outs`
    """

    out_struct = jax.eval_shape(bwd_stage, params, *args)
    out_flat, out_treedef = jax.tree_util.tree_flatten(out_struct)

    closed_jaxpr = jax.make_jaxpr(bwd_stage)(params, *args)

    num_param_leaves = len(jax.tree_util.tree_leaves(params))
    fp32_param_vars = closed_jaxpr.jaxpr.invars[:num_param_leaves]

    bf16_param_vars = [
        core.Var(
            aval=core.ShapedArray(v.aval.shape, jnp.bfloat16),
            initial_qdd=v.initial_qdd,
            final_qdd=v.final_qdd,
        )
        for v in fp32_param_vars
    ]

    def transform_jaxpr(
        jaxpr: core.Jaxpr,
        bf16_vars: list[core.Var],
        fp32_vars: list[core.Var],
    ) -> core.Jaxpr:
        fp32_to_bf16 = dict(zip(fp32_vars, bf16_vars))
        
        aliases = {}
        def maybe_get_alias(v):
            if not isinstance(v, core.Var):
                return v
            return aliases.get(v, v)
        
        new_invars = jaxpr.invars + bf16_vars

        # Mutates `aliases`.
        def transform_eqn(eqn) -> core.JaxprEqn | None:
            if (
                eqn.primitive.name == "convert_element_type"
                and eqn.invars[0] in fp32_to_bf16
                and eqn.params.get("new_dtype") == jnp.bfloat16
            ):
                aliases[eqn.outvars[0]] = fp32_to_bf16[eqn.invars[0]]
                return None

            # Return early if the eqn does not have a sub-jaxpr.
            if "jaxpr" not in eqn.params:
                return eqn.replace(
                    invars=[maybe_get_alias(v) for v in eqn.invars],
                    outvars=[maybe_get_alias(v) for v in eqn.outvars],
                )

            # Recursively transform the sub-jaxpr.
            inner_is_closed = isinstance(eqn.params["jaxpr"], core.ClosedJaxpr)
            jaxpr_param_val = eqn.params["jaxpr"]
            inner_raw_jaxpr = jaxpr_param_val.jaxpr if inner_is_closed else jaxpr_param_val

            new_bf16_vars = []
            new_fp32_vars = []
            additional_eqn_invars = []

            for inner_invar, eqn_invar in zip(
                inner_raw_jaxpr.invars, eqn.invars
            ):
                if eqn_invar in fp32_vars:
                    new_fp32_vars.append(inner_invar)
                    additional_eqn_invars.append(fp32_to_bf16[eqn_invar])
                    new_bf16_vars.append(
                        core.Var(
                            aval=fp32_to_bf16[eqn_invar].aval,
                            initial_qdd=inner_invar.initial_qdd,
                            final_qdd=inner_invar.final_qdd,
                        )
                    )

            new_eqn_invars = [
                maybe_get_alias(v) for v in eqn.invars
            ] + additional_eqn_invars

            new_eqn_outvars = [maybe_get_alias(v) for v in eqn.outvars]

            new_eqn_params = eqn.params.copy()
            transformed_raw_jaxpr = transform_jaxpr(
                inner_raw_jaxpr, new_bf16_vars, new_fp32_vars
            )

            if inner_is_closed:
                new_eqn_params["jaxpr"] = core.ClosedJaxpr(
                    jaxpr=transformed_raw_jaxpr,
                    consts=jaxpr_param_val.consts,
                )
            else:
                new_eqn_params["jaxpr"] = transformed_raw_jaxpr

            if "in_shardings" in new_eqn_params:
                new_eqn_params["in_shardings"] = new_eqn_params["in_shardings"] + (
                    UNSPECIFIED,
                ) * len(additional_eqn_invars)

            if "donated_invars" in new_eqn_params:
                new_eqn_params["donated_invars"] = new_eqn_params["donated_invars"] + (
                    False,
                ) * len(additional_eqn_invars)

            if (
                "in_layouts" in new_eqn_params
                and new_eqn_params["in_layouts"] is not None
            ):
                new_eqn_params["in_layouts"] = new_eqn_params["in_layouts"] + (
                    None,
                ) * len(additional_eqn_invars)

            return eqn.replace(
                invars=new_eqn_invars,
                outvars=new_eqn_outvars,
                params=new_eqn_params,
            )

        new_eqns = []
        for eqn in jaxpr.eqns:
            new_eqn = transform_eqn(eqn)
            if new_eqn is not None:
                new_eqns.append(new_eqn)

        new_outvars = [maybe_get_alias(v) for v in jaxpr.outvars]

        new_debug_info = jaxpr.debug_info
        if jaxpr.debug_info.arg_names is not None:
            num_new_args = len(new_invars) - len(jaxpr.invars)
            new_arg_names = tuple(
                new_debug_info.arg_names + ("_bf16_created",) * num_new_args
            )
            new_debug_info = DebugInfo(
                traced_for=jaxpr.debug_info.traced_for,
                func_src_info=jaxpr.debug_info.func_src_info,
                arg_names=new_arg_names,
                result_paths=jaxpr.debug_info.result_paths,
            )

        return core.Jaxpr(
            constvars=jaxpr.constvars,
            invars=new_invars,
            outvars=new_outvars,
            eqns=new_eqns,
            effects=jaxpr.effects,
            debug_info=new_debug_info,
            is_high=jaxpr.is_high,
        )

    new_jaxpr = transform_jaxpr(closed_jaxpr.jaxpr, bf16_param_vars, fp32_param_vars)

    def new_bwd_stage(*args):
        flat_args = jax.tree_util.tree_leaves(args)
        return jax.tree.unflatten(
            out_treedef, core.eval_jaxpr(new_jaxpr, closed_jaxpr.consts, *flat_args)
        )

    return new_bwd_stage


def bwd_stage(bwd, stage_idx, params, stashed, out_cot, grads_acc):
  grads, in_cot = bwd(stashed, out_cot, params)
  grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
  if stage_idx != 0:
    return params, grads_acc, in_cot
  return params, grads_acc

def bwd_last_stage(bwd, update, bf16_params, params, stashed, out_cot, opt_state, grads_acc):
  params, grads_acc = bwd(bf16_params, params, stashed, out_cot, grads_acc)
  return update(params, opt_state, grads_acc)


def make_update_stage(tx, stage_idx, stage_state):

  # Check if any of the opt_state arrays have a single device sharding.
  def check_single(path, x):
    if hasattr(x, "sharding") and isinstance(x.sharding, jax.sharding.SingleDeviceSharding):
      print(f"WARNING: element at {path} has SingleDeviceSharding {x.sharding}")
  jax.tree.map_with_path(
    check_single,
    stage_state.opt_state
  )

  params_specs = jax.tree.map(lambda x: x.sharding.spec, stage_state.params)
  opt_state_specs = jax.tree.map(lambda x: x.sharding.spec, stage_state.opt_state)
  grad_specs = params_specs
  io_specs = (params_specs, opt_state_specs, grad_specs)


  def update_stage(params, opt_state, grads):
    stage_mesh = mpmd.get_context().get_stage_mesh(stage_idx)

    @partial(jax.shard_map,
      in_specs=io_specs,
      out_specs=io_specs,
      mesh=stage_mesh)
    def shard_mapped_update(ps, os, gs):
      # No OWG: https://github.com/google/flax/blob/240a5107c02d60c171098fbc3f2738d8b6f5ba75/flax/training/train_state.py#L108-L110
      assert nn.fp8_ops.OVERWRITE_WITH_GRADIENT not in grads

      reduced_grads = jax.lax.psum(gs, axis_name="data")
      updates, new_opt_state = tx.update(reduced_grads, os, ps)
      new_params = optax.apply_updates(ps, updates)
      return new_params, new_opt_state, reduced_grads

    return shard_mapped_update(params, opt_state, grads)
  
  return update_stage


def get_section_fns(model, state_by_stage) -> dict[mpmd.SectionName, Callable]:
  num_mubatches = model.config.num_pipeline_microbatches
  dp_factor = model.mesh.shape["data"]
  section_fns = {}

  def constrain_to_param_shardings(x, stage_idx):
    stage_mesh = mpmd.get_context().get_stage_mesh(stage_idx)

    return jax.lax.with_sharding_constraint(
      x,
      jax.tree.map(
        lambda x: mpmd.sharding_with_mesh(x.sharding, stage_mesh),
        state_by_stage[stage_idx].params,
      ),
    )

  # We do this to avoid problems with late-binding with Python closures.
  def make_fwd_step(fwd_fn, idx):

    def fwd_step(bf16_params, *args):
      fwd_fn_out = remove_casts(
        partial(fwd_stage, fwd_fn), *args
      )(*args, bf16_params)
      return constrain_to_param_shardings(fwd_fn_out[0], idx), *fwd_fn_out[1:]

    return fwd_step


  def make_bwd_step(bwd_fn, idx):

    def bwd_step(bf16_params, *args):
      bwd_fn_out = remove_casts(
        partial(bwd_stage, bwd_fn, idx), *args
      )(*args, bf16_params)

      if idx != 0:
        stage_params, stage_grads, activation_cot = bwd_fn_out
        return constrain_to_param_shardings(stage_params, idx), constrain_to_param_shardings(stage_grads, idx), activation_cot
      else:
        stage_params, stage_grads = bwd_fn_out
        return constrain_to_param_shardings(stage_params, idx), constrain_to_param_shardings(stage_grads, idx)
    
    return bwd_step

  for stage_index, state in enumerate(state_by_stage):
    init_stage = make_init_stage(num_mubatches, dp_factor, state.params, stage_index)
    fwd, bwd = model_fwd_and_bwd(model, stage_index)
    update_stage = make_update_stage(state.tx, stage_index, state)
    section_fns[(mpmd.SectionKind.Prologue, stage_index)] = init_stage
    section_fns[(mpmd.SectionKind.Forward, stage_index)] = make_fwd_step(fwd, stage_index)
    section_fns[(mpmd.SectionKind.Backward, stage_index)] = make_bwd_step(bwd, stage_index)
    
    if stage_index != 0:
      section_fns[(mpmd.SectionKind.Epilogue, stage_index)] = update_stage
    else:
      # params, stashed, out_cot, grads_acc
      section_fns[(mpmd.SectionKind.BackwardLast, stage_index)] = \
        partial(bwd_last_stage, make_bwd_step(bwd, 0), update_stage)
  return section_fns


### Managing flax and optax state

def remove_mesh_axis(
  arr: Any,
  mesh_axis_name: str,
  mesh_axis_slice_idx: int,
  ctx: Any = None,
) -> Any:
  if ctx is not None and ctx.tracing_for_inference:
    return arr

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

def split_params_by_stage(num_stages, all_params):
  # Assumption: no params are shared between stages; we specialize to Transformer.
  params_by_stage = []
  _all_params = all_params["params"]
  for stage_index in range(num_stages):
    _params = {}
    layers_name = f"stage{stage_index}_layers"
    _params[layers_name] = _all_params[layers_name]
    if stage_index == 0:
      _params["token_embedder"] = _all_params["token_embedder"]
    if stage_index == num_stages - 1:      
      _params["final_layer"] = _all_params["final_layer"]
    params_by_stage.append({"params": _params})
  return tuple(params_by_stage)


# def split_opt_state_by_stage(num_stages, opt_state):
#   # Assumption: Optimizer state consists of mu and nu.
#   # https://flax-linen.readthedocs.io/en/latest/guides/model_inspection/model_surgery.html#surgery-with-optimizers
#   mu_by_stage = split_params_by_stage(num_stages, opt_state[0].mu)
#   nu_by_stage = split_params_by_stage(num_stages, opt_state[0].nu)

#   opt_state_by_stage = [
#     utils.tuple_update(opt_state, 0, opt_state[0]._replace(mu=mu, nu=nu, step=))
#     for mu, nu in zip(mu_by_stage, nu_by_stage)
#   ]
#   return tuple(opt_state_by_stage)

def split_opt_state_by_stage(num_stages, opt_state):
  mu_by_stage = split_params_by_stage(num_stages, opt_state[0].mu)
  nu_by_stage = split_params_by_stage(num_stages, opt_state[0].nu)

  def clone_scalars(leaf):
    if isinstance(leaf, (int, float)) and not isinstance(leaf, jax.Array):
      return jnp.array(leaf)
    if hasattr(leaf, 'shape') and leaf.shape == () and leaf.dtype == jnp.int32:
      return leaf.copy() 
    return leaf

  opt_state_by_stage = []
  
  for mu, nu in zip(mu_by_stage, nu_by_stage):
    local_opt_state = jax.tree.map(clone_scalars, opt_state)
    new_adam_state = local_opt_state[0]._replace(mu=mu, nu=nu)
    stage_opt_state = utils.tuple_update(local_opt_state, 0, new_adam_state)
    
    opt_state_by_stage.append(stage_opt_state)

  return tuple(opt_state_by_stage)


def split_state_by_stage(num_logical_stages, num_physical_stages, state):
  params_by_stage = split_params_by_stage(num_logical_stages, state.params)
  opt_state_by_stage = split_opt_state_by_stage(num_logical_stages, state.opt_state)
  return tuple(
    jax.tree.map(
      lambda x: remove_mesh_axis(x, "stage", stage_idx % num_physical_stages),
      state.replace(step=state.step, params=params, opt_state=opt_state),
    )
    for stage_idx, (params, opt_state) in enumerate(
      zip(params_by_stage, opt_state_by_stage, strict=True)
    )
  )


# update_state is the stage-sharded equivalent of
#   new_state = old_state.apply_gradients(grads=grads)
#
# To work around flax and optax's API and complexity of cross-stage sharding we
# make some heavy-handed assumptions here:
# - params are owned by exactly one stage (i.e. no weight sharing across stages)
# - optimizer state is sharded analogously (no cross-stage dependencies)
def update_state(ctx, old_state_by_stage, grads_by_stage):
  new_state_by_stage = []
  for stage_index, (old_state, grads) in enumerate(
      zip(old_state_by_stage, grads_by_stage, strict=True)):

    if stage_index == 0:
      new_state_by_stage.append(old_state)
      continue

    params, opt_state = old_state.params, old_state.opt_state
    _update_stage_state = ctx.section((mpmd.SectionKind.Epilogue, stage_index),
      donate_argnums=(0, 1, 2))
    
    with utils.annotate(f"update{stage_index}", color="green"):
      new_params, new_opt_state, _ = _update_stage_state(params, opt_state, grads)

    new_state_by_stage.append(
      old_state.replace(
        step=old_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
      )
    )
  return tuple(new_state_by_stage)


### Transfer state and input data to the corresponding stages' meshes


def _constant(stage_idx, const, *, shape=(), spec=PartitionSpec()):
  ctx = mpmd.get_context()
  stage_mesh = ctx.get_stage_mesh(stage_idx)
  sharding = NamedSharding(stage_mesh, spec)
  arr = jnp.full(shape, const, device=sharding)
  if ctx.tracing_for_inference:
    return arr
  return arr


def constant(stage_idx, const, *, shape=(), spec=PartitionSpec()):
  with utils.annotate(f"const {stage_idx=} {const=} {shape=}", color="yellow"):
    return _constant(stage_idx, const, shape=shape, spec=spec)


def transfer(stage_idx, xs):
  ctx = mpmd.get_context()
  if ctx.tracing_for_inference:
    return xs
  stage_mesh = ctx.get_stage_mesh(stage_idx)

  def transfer_one(x):
    sharding = mpmd.sharding_with_mesh(x.sharding, stage_mesh)
    return jax.device_put(x, device=sharding)

  return jax.tree.map(transfer_one, xs)


def setup_state_for_dp(state_by_stage, mesh):
  def broadcast_and_shard(x, stage_index):
    # If x is not an array, convert to array if it is a scalar number, otherwise return.
    if not isinstance(x, jax.Array):
      if isinstance(x, (int, float)):
        x = jnp.array(x)
      else:
        return x

    stage_mesh = mpmd.get_stage_mesh(mesh, stage_index)
    data_dim = stage_mesh.shape["data"]
    
    # Determine specs
    current_spec = PartitionSpec()
    if isinstance(x.sharding, NamedSharding):
        current_spec = x.sharding.spec
    
    new_spec = PartitionSpec("data", *current_spec)
    new_sharding = NamedSharding(stage_mesh, new_spec)
    
    new_shape = (data_dim, *x.shape)
    
    x = jnp.expand_dims(x, 0)
    x = jnp.broadcast_to(x, new_shape)
    
    return jax.device_put(x, new_sharding)

  new_state_by_stage = []
  for stage_idx, state in enumerate(state_by_stage):

    new_state_by_stage.append(
      jax.tree.map(partial(broadcast_and_shard, stage_index=stage_idx), state)
    )
    # print(f"setup_state_for_dp stage {stage_idx}")
    # params = jax.tree.map(partial(broadcast_and_shard, stage_index=stage_idx), state.params)
    # opt_state = jax.tree.map(partial(broadcast_and_shard, stage_index=stage_idx), state.opt_state)
    
    # new_state_by_stage.append(state.replace(step=jnp.array(state.step), params=params, opt_state=opt_state))

  return tuple(new_state_by_stage)


def split_and_transfer_state(
  mesh, num_logical_stages, num_physical_stages, state, in_shard_train, out_shard_train
):
  # assert mesh.shape["stage"] == num_physical_stages
  # state_by_stage = split_state_by_stage(num_logical_stages, num_physical_stages, state)
  # with mpmd.set_context(mpmd.Context(mesh, tracing_for_inference=False)):
  #   state_by_stage = tuple(
  #     transfer(stage_idx, state) for stage_idx, state in enumerate(state_by_stage)
  #   )
  #   state_by_stage = setup_state_for_dp(state_by_stage)

  # assert in_shard_train[0] == out_shard_train[0]
  # assert isinstance(in_shard_train[0], train_state.TrainState)
  # state_shard_by_stage = split_state_by_stage(
  #   num_logical_stages, num_physical_stages, in_shard_train[0]
  # )
  # state_shard_by_stage = setup_state_for_dp(state_shard_by_stage)

  #   params_by_stage = split_params_by_stage(num_logical_stages, state.params)
  # opt_state_by_stage = split_opt_state_by_stage(num_logical_stages, state.opt_state)
  # return tuple(
  #   jax.tree.map(
  #     lambda x: remove_mesh_axis(x, "stage", stage_idx % num_physical_stages),
  #     state.replace(step=state.step, params=params, opt_state=opt_state),
  #   )
  #   for stage_idx, (params, opt_state) in enumerate(
  #     zip(params_by_stage, opt_state_by_stage, strict=True)
  #   )
  # )

  state = tuple(
    state.replace(step=state.step, params=stage_params, opt_state=stage_opt_state)
    for stage_idx, (stage_params, stage_opt_state) in enumerate(
      zip(state.params, state.opt_state, strict=True)
    )
  )
  state_shard_by_stage = setup_state_for_dp(state, mesh)

  # def _print_sharding_spec(path, leaf):
  #   spec = "N/A"
  #   # Check if it is a JAX array with sharding info
  #   if hasattr(leaf, 'sharding'):
  #     # For NamedSharding (typical with Meshes), the PartitionSpec is in .spec
  #     if hasattr(leaf.sharding, 'spec'):
  #       spec = leaf.sharding.spec
  #     else:
  #       spec = leaf.sharding
    
  #   print(f"{jax.tree_util.keystr(path)} : {spec}")

  # print("\n=== Inspecting State Sharding Specs ===")
  # jax.tree_util.tree_map_with_path(_print_sharding_spec, state_by_stage)
  # print("=======================================\n")

  in_shard_train = (state_shard_by_stage,) + in_shard_train[1:]
  out_shard_train = (state_shard_by_stage,) + out_shard_train[1:]

  return state, in_shard_train, out_shard_train


### Train step

def make_pipeline_schedule(num_logical_stages, num_physical_stages, num_mubatches, schedule_name="gpipe"):
  if schedule_name == "gpipe":
    return make_gpipe_schedule(num_logical_stages, num_physical_stages, num_mubatches)
  elif schedule_name == "1F1B":
    return make_jaxpp_1F1B_schedule(num_logical_stages, num_physical_stages, num_mubatches)


@lru_cache
def setup_value_and_grad(
  num_logical_stages, num_physical_stages, num_mubatches, schedule_name="eager_1F1B"
):
  fwd_input = [[None for _ in range(num_mubatches)] for _ in range(num_logical_stages)]
  stashed = [[None for _ in range(num_mubatches)] for _ in range(num_logical_stages)]
  bwd_input = [[None for _ in range(num_mubatches)] for _ in range(num_logical_stages)]

  return (
    make_pipeline_schedule(num_logical_stages, num_physical_stages, num_mubatches, schedule_name),
    fwd_input,
    stashed,
    bwd_input,
  )


@partial(jax.jit, donate_argnums=(0, 1))
def stack_metrics(loss, aux):
  def _stack_mean(x):
    return jnp.mean(jnp.stack(x), axis=(0, 1))

  loss = _stack_mean(loss)
  aux = jax.tree.map(lambda *xs: _stack_mean(xs), *aux)
  return loss, aux


# TODO: Make sure we only transfer inputs actually needed by a section
def value_and_grad(
    ctx,
    num_logical_stages,
    num_physical_stages,
    num_mubatches,
    state_by_stage,
    data_by_stage,
    dropout_rngs,
    print_memory_usage=False,
    schedule_name="gpipe",
):
  assert num_logical_stages == len(state_by_stage) == len(data_by_stage)

  params_by_stage = tuple(state.params for state in state_by_stage)

  tasks, fwd_input, stashed, bwd_input = setup_value_and_grad(
    num_logical_stages, num_physical_stages, num_mubatches, schedule_name,
  )

  ### State
  fwd_fns = [
    ctx.section(
          (mpmd.SectionKind.Forward, stage_idx),
          donate_argnums=(1,2),
    )
    for stage_idx in range(num_logical_stages)
  ]
  bwd_fns = [
    ctx.section(
          (mpmd.SectionKind.Backward, stage_idx),
          donate_argnums=(1,2,3,4),
    )
    for stage_idx in range(num_logical_stages)
  ]
  # params_by_stage : stage_idx -> params
  params_by_stage = list(params_by_stage)
  bf16_params_by_stage = [None] * num_logical_stages
  # grads_by_stage : stage_idx -> grads
  grads_by_stage = [None] * num_logical_stages
  # loss : mubatch_idx -> loss
  loss = [None] * num_mubatches
  aux = [None] * num_mubatches

  # dropout_rngs : stage_idx -> dropout_rng
  dropout_rngs = tuple(
    transfer(stage_idx, dropout_rngs[stage_idx])
    for stage_idx in range(num_logical_stages)
  )
  # mubatch_idx_consts_by_stage : stage_idx -> mubatch_idx -> const
  mubatch_idx_consts_by_stage = [None] * num_logical_stages

  ### Run initialization for each stage
  for stage_idx in range(num_logical_stages):
    _init_stage = ctx.section((mpmd.SectionKind.Prologue, stage_idx))
    with utils.annotate(f"init{stage_idx}", color="green"):
      grads_by_stage[stage_idx], mubatch_const_ones, mubatch_idx_consts_by_stage[stage_idx], bf16_params_by_stage[stage_idx] = \
        _init_stage(params_by_stage[stage_idx])
    assert num_mubatches == len(mubatch_const_ones) == len(mubatch_idx_consts_by_stage[stage_idx])
    if stage_idx == num_logical_stages - 1:
      for mubatch_idx, const_ones in enumerate(mubatch_const_ones):
        bwd_input[stage_idx][mubatch_idx] = const_ones

  def memory_usage_snapshot(name):
    if not print_memory_usage or ctx.tracing_for_inference:
      return
    state = {
      "params_by_stage": params_by_stage,
      "bf16_params_by_stage": bf16_params_by_stage,
      "opt_state_by_stage": tuple(state.opt_state for state in state_by_stage),
      "fwd_input": fwd_input,
      "stashed": stashed,
      "bwd_input": bwd_input,
      "grads_by_stage": grads_by_stage,
      "loss": loss,
      "aux": aux,
    }
    jax.block_until_ready(state)
    # time.sleep(1)
    record = utils.dump_memory_usage_snapshot(state)
    if name == "start":
      print("MEM,name," + ",".join(k for k in sorted(record.keys())))
    print(f"MEM,{name}," + ",".join(str(record[k]) for k in sorted(record.keys())))

  ### Microbatched forward+backward
  memory_usage_snapshot("start")
  # jax.profiler.save_device_memory_profile("memory_start.prof")

  # Find the last mubatch that the first stage executes a bwd pass on.
  last_bwd_mubatch_stage_0 = None
  for mubatch_idx, stage_idx, is_bwd in reversed(tasks):
    if stage_idx == 0 and is_bwd:
      last_bwd_mubatch_stage_0 = mubatch_idx
      break
  
  # Main pipeline schedule.
  for mubatch_idx, stage_idx, is_bwd in tasks:
    print(f"ON TASK {mubatch_idx=} {stage_idx=} {is_bwd=}")

    with utils.annotate(f"TASK m{mubatch_idx} s{stage_idx} {'BWD' if is_bwd else 'FWD'}", color="blue"):
      ### Forward
      if not is_bwd:
        curr_input = fwd_input[stage_idx][mubatch_idx]
        fwd_input[stage_idx][mubatch_idx] = None  # Clear reference.

        res = fwd_fns[stage_idx](
            bf16_params_by_stage[stage_idx],
            params_by_stage[stage_idx],
            curr_input,
            data_by_stage[stage_idx],
            dropout_rngs[stage_idx],
            mubatch_idx_consts_by_stage[stage_idx][mubatch_idx],
        )
        
        params_by_stage[stage_idx] = res[0]
        stashed[stage_idx][mubatch_idx] = res[1]
        activation = res[2]

        if stage_idx == num_logical_stages - 1:
          loss[mubatch_idx] = activation
          aux[mubatch_idx] = res[3]
        else:
          fwd_input[stage_idx + 1][mubatch_idx] = transfer(stage_idx + 1, activation)
        
        del res
        del activation
        del curr_input
      
      ### Backward
      else:
        curr_stashed = stashed[stage_idx][mubatch_idx]
        stashed[stage_idx][mubatch_idx] = None  # Clear reference.

        curr_bwd_input = bwd_input[stage_idx][mubatch_idx]
        bwd_input[stage_idx][mubatch_idx] = None  # Clear reference.

        # Run BackwardLast instead of Backward for the first stage backward.
        if stage_idx == 0 and mubatch_idx == last_bwd_mubatch_stage_0:
          bwd_last_fn = ctx.section(
            (mpmd.SectionKind.BackwardLast, stage_idx),
            donate_argnums=(1,2,3,4)
          )

          # print("ABOUT TO CALL BWD LAST. STATE SHARDINGS:")

          # jax.tree.map_with_path(
          #   lambda path, x: print(f"{path}: {x.sharding if hasattr(x, "sharding") else None}"),
          #   state_by_stage[stage_idx],
          # )

          bwd_fn_out = bwd_last_fn(
              bf16_params_by_stage[stage_idx],
              params_by_stage[stage_idx],
              curr_stashed,
              curr_bwd_input,
              state_by_stage[stage_idx].opt_state,
              grads_by_stage[stage_idx],
            )
          
          old = params_by_stage[stage_idx]
          params_by_stage[stage_idx] = None
          del old
          
          old = grads_by_stage[stage_idx]
          grads_by_stage[stage_idx] = None
          del old
          
          params_by_stage[stage_idx], new_opt_state, grads_by_stage[stage_idx] = bwd_fn_out
          del bwd_fn_out
          
          new_state_by_stage = tuple(
            state_by_stage[i] if i != 0 else state_by_stage[i].replace(opt_state=new_opt_state)
            for i in range(len(state_by_stage))
          )
          del new_opt_state
          del state_by_stage
          state_by_stage = new_state_by_stage
          del new_state_by_stage
        
        else:
          bwd_fn_out = bwd_fns[stage_idx](
              bf16_params_by_stage[stage_idx],
              params_by_stage[stage_idx],
              curr_stashed,
              curr_bwd_input,
              grads_by_stage[stage_idx],
          )
          
          old = params_by_stage[stage_idx]
          params_by_stage[stage_idx] = None
          del old
          
          old = grads_by_stage[stage_idx]
          grads_by_stage[stage_idx] = None
          del old
          
          if stage_idx != 0:
            params_by_stage[stage_idx], grads_by_stage[stage_idx], activation_cot = bwd_fn_out
            bwd_input[stage_idx - 1][mubatch_idx] = transfer(stage_idx - 1, activation_cot)
            del activation_cot
          else:
            params_by_stage[stage_idx], grads_by_stage[stage_idx] = bwd_fn_out
          
          del bwd_fn_out
        
        del curr_stashed
        del curr_bwd_input
    
    # all_locals = locals()
    # print(list(all_locals.keys()))
    memory_usage_snapshot(f"after_task(mubatch_idx={mubatch_idx}|stage_idx={stage_idx}|is_bwd={is_bwd})")

  loss, aux = stack_metrics(loss, aux)

  state_by_stage = tuple(
    state.replace(params=params)
    for state, params in zip(state_by_stage, params_by_stage)
  )
  
  return state_by_stage, grads_by_stage, (loss, aux)

@partial(jax.jit, static_argnums=(1, 2), donate_argnums=(0,))
def reshape_reshard_data(data, num_mubatches, dp_factor):
  def reshape_reshard_arr(arr):
    arr = arr.reshape((num_mubatches, dp_factor, -1, *arr.shape[1:]))
    return nn.with_logical_constraint(
      arr,
      (None,"activation_embed_and_logits_batch_outside_vmap",),
    )
  return jax.tree.map(reshape_reshard_arr, data)

# @utils.annotate_step
def train_step(model, config, _state_mesh_shardings, _params_shardings, state_by_stage, data, dropout_rng):
  assert config is model.config
  assert not config.gradient_clipping_threshold > 0
  assert not config.optimizer_memory_host_offload
  assert not config.use_dpo
  assert not config.use_multimodal
  assert not config.gradient_accumulation_steps > 1
  assert not config.record_internal_nn_metrics
  assert not config.enable_dropout

  assert nn.get_logical_axis_rules() != (), "expected some linen logical axis rules"

  ctx = mpmd.get_context()
  # TODO: When ctx.tracing_for_inference, only unroll microbatching loop for first iteration
  num_mubatches = config.num_pipeline_microbatches

  # TODO: Investigate whether this replication and resharding is a bottleneck
  data = reshape_reshard_data(data, num_mubatches, model.mesh.shape["data"])  
  data_by_stage = tuple(
    transfer(
      stage_index,
      jax.tree.map(
        lambda x: remove_mesh_axis(
          x, "stage", stage_index % model.num_physical_stages, ctx=mpmd.get_context()
        ),
        data,
      ),
    )  # replicate, it's (relatively) cheap and might overlap?
    for stage_index in range(model.num_logical_stages)
  )

  # TODO: Also replicate dropout_rng to all stages and donate?
  dropout_rngs = tuple(
    jax.tree.map(
      lambda x: remove_mesh_axis(x, "stage", stage_index % model.num_physical_stages, ctx=ctx),
      dropout_rng,
    )
    for stage_index in range(model.num_logical_stages)
  )

  # Note: value_and_grad donates params; the params_by_stage returned will merely be
  # fresh jax.Arrays containing the same data.
  state_by_stage, grads_by_stage, (loss, aux) = value_and_grad(
    ctx, model.num_logical_stages, model.num_physical_stages, num_mubatches, 
    state_by_stage, data_by_stage, dropout_rngs, schedule_name=config.mmpp_schedule,
    print_memory_usage=model.config.mmpp_print_memory_usage
  )

  new_state_by_stage = update_state(ctx, state_by_stage, grads_by_stage)

  scalar_metrics = {
    "learning/loss": loss,
    "learning/moe_lb_loss": aux["moe_lb_loss"],
    "learning/total_weights": aux["total_weights"],
  }
  metrics = {
    "scalar": scalar_metrics,
    "scalars": {},
  }
  return new_state_by_stage, metrics


def get_resident_bytes(params):
    """
    Computes the total bytes actually stored on addressable devices 
    for the current process (useful for distributed/sharded settings).
    """
    leaves = jax.tree_util.tree_leaves(params)
    total_bytes = 0
    
    for leaf in leaves:
        # Check if it is a JAX Array (which has shards)
        if isinstance(leaf, jax.Array):
            # Sum the size of the data buffers on addressable devices only
            for shard in leaf.addressable_shards:
                total_bytes += shard.data.nbytes
        elif hasattr(leaf, 'nbytes'):
            # Fallback for standard numpy arrays or scalars
            total_bytes += leaf.nbytes
            
    return total_bytes


def init_state_mmpp(model, tx, config, rng_key):
  assert isinstance(model, models.Transformer)

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

  for logical_idx in range(model.num_logical_stages):
    print(f"==== INIT_STATE_MMPP, LOGICAL_IDX = {logical_idx} ====")
    stage_mesh = mpmd.get_stage_mesh(model.mesh, logical_idx)
    
    # Derive shardings from logical axis rules.
    input_logical_spec = PartitionSpec(*config.input_data_sharding_logical_axes)
    input_sharding = nn.logical_to_mesh_sharding(
        input_logical_spec, stage_mesh, config.logical_axis_rules
    )

    activation_logical_spec = PartitionSpec(
        "activation_embed_and_logits_batch_outside_vmap",
        "activation_length",
        "activation_embed"
    )
    activation_sharding = nn.logical_to_mesh_sharding(
        activation_logical_spec, stage_mesh, config.logical_axis_rules
    )

    if logical_idx == 0:
      y_sharding = input_sharding
      y_shape = input_shape
      y_dtype = jnp.int32
    else:
      y_sharding = activation_sharding
      y_shape = activation_shape
      y_dtype = config.dtype

    dummy_input = (
      jnp.ones(y_shape, dtype=y_dtype, device=y_sharding),  # y
      jnp.ones(input_shape, dtype=jnp.int32, device=input_sharding),  # decoder_segment_ids
      jnp.ones(input_shape, dtype=jnp.int32, device=input_sharding),  # decoder_positions
      jnp.ones(input_shape, dtype=jnp.int32, device=input_sharding),  # decoder_targets
      jnp.ones(input_shape, dtype=jnp.int32, device=input_sharding),  # decoder_targets_segmentation
    )

    # print(f"dummy_input shardings, logical_idx = {logical_idx}")
    # print(dummy_input[0].sharding)
    # print(dummy_input[1].sharding)
    # print(dummy_input[2].sharding)
    # print(dummy_input[3].sharding)
    # print(dummy_input[4].sharding)
    
    with stage_mesh:
      init_rngs = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(stage_mesh, PartitionSpec())),
        {
          "params": stage_rng_keys[logical_idx],
          "dropout": stage_rng_keys[logical_idx],
          "aqt": stage_rng_keys[logical_idx],
        }
      )
      
      def init_fn(rngs, *args):
        with mpmd.set_context(mpmd.Context(model.mesh, tracing_for_inference=False)):
          stage_params = model.init(rngs, logical_idx, *args, method=model._stage)
          return train_state.TrainState.create(
            apply_fn=partial(models.forward, model=model, stage_index=logical_idx),
            params=stage_params,
            tx=tx
          )

      with nn_partitioning.axis_rules(config.logical_axis_rules):
        abstract_params = jax.eval_shape(init_fn, init_rngs, *dummy_input)
        state_logical_annotations = nn.get_partition_spec(abstract_params)
        state_mesh_shardings = nn.logical_to_mesh_sharding(
            state_logical_annotations, stage_mesh, config.logical_axis_rules
        )

        # print("Inside init_state_mmpp, stage_mesh is")
        # print(stage_mesh)
        # print()

        # print("Inside init_state_mmpp, state_logical_annotations:")
        # jax.tree.map_with_path(
        #   lambda path, x: print(f"\t{path}: {x}"),
        #   state_logical_annotations,
        # )
        # print()

        # print("Inside init_state_mmpp, state_mesh_shardings:")
        # jax.tree.map_with_path(
        #   lambda path, s: print(f"\t{path}: {s.spec if isinstance(s, NamedSharding) else None}"),
        #   state_mesh_shardings,
        # )
        # print()

        state[logical_idx] = jax.jit(
            init_fn,
            out_shardings=state_mesh_shardings
        )(init_rngs, *dummy_input)

  state = max_utils.unbox_logicallypartioned(tuple(state))
  state = setup_state_for_dp(state, model.mesh)

  # Form state_mesh_annotations, state_mesh_shardings.
  state_mesh_annotations = jax.tree.map(
    lambda x: x.sharding.spec if hasattr(x, "sharding") and hasattr(x.sharding, "spec") else None,
    state
  )  
  state_mesh_shardings = jax.tree.map(
    lambda x: getattr(x, "sharding", None), state
  )    

  return state, state_mesh_annotations, state_mesh_shardings


def prepare_state_and_train_step(
    mesh,
    model,
    state,
    init_rng,
    functional_train,
    in_shard_train,
    out_shard_train,
    example_batch,
):
  # state, in_shard_train, out_shard_train = split_and_transfer_state(
  #     mesh,
  #     model.num_logical_stages,
  #     model.num_physical_stages,
  #     state,
  #     in_shard_train,
  #     out_shard_train,
  # )

  # Replicate init_rng
  init_rng = jax.device_put(init_rng, device=NamedSharding(mesh, PartitionSpec()))

  p_train_step = mpmd.transform(
      mesh,
      get_section_fns(model, state),
      functional_train,
      in_shard_train,
      out_shard_train,
      state,
      example_batch,
      init_rng,
      model.config.logical_axis_rules,
  )

  return state, init_rng, p_train_step
