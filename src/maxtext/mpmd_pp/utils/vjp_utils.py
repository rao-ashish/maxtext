"""Utilities to prepare forward / backward passes for pipeline stages."""

from typing import Callable, Sequence

import jax
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info
from jax._src.api import NotNeeded


DUMMY_VJP_FUN = None


def fwd_and_bwd(
    fun: Callable,
    argnums: Sequence[int],
    caller_saved_among_argnums: Sequence[bool],
    has_aux: bool = False,
    jitted: bool = True,
) -> tuple[Callable, Callable]:
    """Given a function, prepare callables to compute a forward and backward
    pass through that function.

    Args:
        fun: A function that computes outputs (fun_out, *rest) that we will
            prepare a forward and backward pass for.
        argnums: The indices of fun's arguments with respect to which bwd should
            compute a gradient.
        caller_saved_among_argnums: The indices of fun's arguments which the
            caller will provide again when calling the backward pass. These
            arguments will not be saved inside the stashed residuals returned
            by the forward pass.
        has_aux: Whether fun returns auxiliary outputs beyond the one that
            the backward pass should differentiate.
        jitted: Whether to jit the output fwd / bwd functions.

    Returns:
        fwd: A function that takes the same inputs as fun, but returns a tuple
            (stashed, fun_out, *rest), where fun_out is the output of `fun`,
            and stashed are arrays that should be stored for the backward pass.

        bwd: A function that takes in the stashed values produced by the
            forward pass, the output cotangents flowing into this section as
            part of reverse-mode autodiff, and the arguments that were given
            to the forward pass which were marked as caller saved via
            caller_saved_among_argnums.
    """

    def argnum_is_caller_saved(argnum):
        return argnum in argnums and caller_saved_among_argnums[argnums.index(argnum)]

    inner_vjp_fn = None

    def fwd(*args, **kwargs):
        nonlocal inner_vjp_fn

        # Partially evaluate f to only take in the arguments at argnums.
        # dyn_args is the subset of args corresponding to argnums.
        dbg = debug_info("fwd_and_bwd", fun, args, kwargs)
        f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )

        # Take the vjp.
        primals_out, vjp_fn, *rest = jax._src.api._vjp(
            f_partial, *dyn_args, has_aux=has_aux
        )

        # vjp_fn.args_res contains the saved arguments. vjp_fn.opaque_residuals
        # contains saved activations for the backward pass.
        #
        # For args which have caller_saved_among_argnums = True, we do not need
        # to save those inside args_res, because the caller will provide them
        # again when bwd is called. So, remove them.
        vjp_fn.args_res = [
            res if not caller_saved_among_argnums[idx] else NotNeeded()
            for idx, res in enumerate(vjp_fn.args_res)
        ]

        # Save vjp_fn.fun 'out of band' from bwd's args. Then, remove it from
        # vjp_fn.
        inner_vjp_fn = vjp_fn.fun
        vjp_fn.fun = DUMMY_VJP_FUN

        return vjp_fn, primals_out, *rest

    def bwd(vjp_fn, *outgrad_and_saved):
        assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1

        outgrad = outgrad_and_saved[0]
        caller_saved_vals = outgrad_and_saved[1:]

        # Reset vjp_fn.fun.
        assert inner_vjp_fn != DUMMY_VJP_FUN, "inner_vjp_fn was not properly captured."
        vjp_fn.fun = inner_vjp_fn

        # Reconstruct the full args list used to create f_partial.
        new_args_res = []
        curr_caller_saved_idx = 0
        for idx, arg_res in enumerate(vjp_fn.args_res):
            if caller_saved_among_argnums[idx]:
                new_args_res.append(caller_saved_vals[curr_caller_saved_idx])
                curr_caller_saved_idx += 1
            else:
                new_args_res.append(arg_res)
        vjp_fn.args_res = new_args_res

        return vjp_fn(outgrad)

    if jitted:
        fwd = jax.jit(fwd)
        bwd = jax.jit(bwd)
    return fwd, bwd
