"""Utilities to transform section functions for MPMD pipeline parallelism
implementation."""

import jax
import jax.numpy as jnp
import jax._src.core as core
from jax._src.linear_util import DebugInfo
from jax._src.named_sharding import UNSPECIFIED


def remove_casts(section_fn, params, *args):
    """Modify section_fn to remove casts of params to bf16, and take in those
    bf16 values as inputs instead.

    section_fn is assumed to have signature:
        `params, *args -> *outs`.

    The returned callable will have signature:
        `params, *args, bf16_params -> *outs`
    """

    out_struct = jax.eval_shape(section_fn, params, *args)
    out_flat, out_treedef = jax.tree_util.tree_flatten(out_struct)

    closed_jaxpr = jax.make_jaxpr(section_fn)(params, *args)

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
            inner_raw_jaxpr = (
                jaxpr_param_val.jaxpr if inner_is_closed else jaxpr_param_val
            )

            new_bf16_vars = []
            new_fp32_vars = []
            additional_eqn_invars = []

            for inner_invar, eqn_invar in zip(inner_raw_jaxpr.invars, eqn.invars):
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

    def new_section_fn(*args):
        flat_args = jax.tree_util.tree_leaves(args)
        return jax.tree.unflatten(
            out_treedef, core.eval_jaxpr(new_jaxpr, closed_jaxpr.consts, *flat_args)
        )

    return new_section_fn


def test_remove_casts():
    """A small unit test for remove_casts()."""

    @jax.jit
    def section_fn(params, x, y):
        bf16_params = params.astype(jnp.bfloat16)
        x_new = bf16_params @ x
        y_new = bf16_params @ y

        @jax.jit
        def inner_jit(p, a, b):
            bf16_p = p.astype(jnp.bfloat16)
            a_new = bf16_p @ a
            b_new = bf16_p @ b
            return a_new + b_new

        return inner_jit(params, x_new, y_new)

    params_key, x_key, y_key = jax.random.split(jax.random.key(1), 3)

    params = jax.random.normal(params_key, (10, 10), dtype=jnp.float32)
    bf16_params = params.astype(jnp.bfloat16)
    x = jax.random.normal(x_key, (10, 10), dtype=jnp.bfloat16)
    y = jax.random.normal(y_key, (10, 10), dtype=jnp.bfloat16)

    new_section_fn = jax.jit(remove_casts(section_fn, params, x, y))

    old_output = section_fn(params, x, y)
    new_output = new_section_fn(params, x, y, bf16_params)

    # This should be close to 0.
    print(f"l_inf norm = {jnp.max(jnp.abs(old_output - new_output))}")

    all_close = jnp.allclose(old_output, new_output)

    # This should be True.
    print(f"all_close = {all_close}")


if __name__ == "__main__":
    test_remove_casts()
