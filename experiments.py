from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from jax._src.api import saved_input_vjp
from jax._src.api_util import debug_info


def fwd_and_bwd(
    fun: Callable,
    argnums: Sequence[int],
    caller_saved_among_argnums: Sequence[bool],
    has_aux: bool = False,
    jitted: bool = True,
) -> tuple[Callable, Callable]:
    """Create forward and backward pass closures for a given fun.

    Args:
        fun: The function representing a forward pass.
        argnums: The argnums of fun wrt to which the backward pass should
            compute gradients.
        caller_saved_among_argnums: Element i is true if the ith input to the
            fwd pass will be given to the generated bwd function at call time
            (the bwd closure will not have to save the value for later).
        has_aux: Whether fun returns auxiliary outputs beyond the output to
            differentiate.
        jitted: Whether to jit the generated fwd and bwd closures.

    Returns:
        fwd: A callable for the fwd pass.
        bwd: A callable for the bwd pass.
    """

    # The forward stage to return.
    def fwd(*args, **kwargs):
        # Partially evaluate f against args for which
        # caller_saved_among_argnums is false.
        #
        #   f_partial(*[args[i] for i in range(len(argnums))
        #               if caller_saved_among_argnums[i]])
        #
        # will return f(*args).
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

            return f(*new_args)

        return jax._src.api.saved_input_vjp(
            f_partial,
            caller_saved_among_argnums,
            *args,
        )

    def bwd(f_vjp, *outgrad_and_saved):
        assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1
        return f_vjp(*outgrad_and_saved)

    if jitted:
        fwd = jax.jit(fwd)
        bwd = jax.jit(bwd)
    return fwd, bwd


def f(a, b):
    return a * b


a = jnp.zeros((10,), dtype=jnp.float32)
b = jnp.zeros((10,), dtype=jnp.float32)
v = jnp.ones((10,), dtype=jnp.float32)

f_val_1, f_vjp_1 = jax.vjp(f, a, b)
print(f_vjp_1(v))

f_val_2, f_vjp_2 = saved_input_vjp(f, (True, True), a, b)
print(f_vjp_2(v, a, b))

fwd, bwd = fwd_and_bwd(
    f,
    argnums=(0, 1),
    caller_saved_among_argnums=(True, True),
    has_aux=False,
    jitted=False,
)

f_val_3, f_vjp_3 = fwd(a, b)
print(bwd(f_vjp_3, v, a, b))