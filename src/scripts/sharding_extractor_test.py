from functools import partial, wraps

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

from MaxText.mmpp import mpmd


def is_jax_partial(x):
    return x is None or isinstance(x, jax._src.tree_util.Partial)


def sharding_extractor():
    def get_shape(x):
        try:
            aval = jax.typeof(x)
            if type(aval) is jax.core.ShapedArray:
                return aval.shape
        except TypeError:
            pass
        return None

    def check_and_store(shape, shardings, index, sharding):
        sharding.shard_shape(shape)
        print(f"Inside check_and_store, setting shardings[{index}] = {sharding}")
        shardings[index] = sharding

    def register_store_callbacks(xs, curr_name):
        xs_flat, xs_tree = jax.tree.flatten(xs, is_leaf=is_jax_partial)

        print(f"Inside register_store_callbacks {curr_name}, xs = {xs}")

        shardings = [None] * len(xs_flat)
        for index, x in enumerate(xs_flat):
            if shape := get_shape(x):
                print(f"get_shape {curr_name} {index} is {shape}")
                callback = partial(check_and_store, shape, shardings, index)
                jax.debug.inspect_array_sharding(x, callback=callback)
            else:
                print(f"get_shape {curr_name} {index} is {shape}")
                # Note that we end up here for jax Partials (because we first treat them
                # as leaves and then decide not to inspect the sharding for them). Simply
                # setting the inferred sharding to None means the Partials do not show
                # up in the inferred shardings; jax will treat these as replicated, but
                # all the data packaged in the Partial will have been replaced by dummy
                # values anyways (see vjp_unpack).
                print(f"Inside register_store_callbacks {curr_name} {index}, setting shardings[{index}] = None")
                shardings[index] = None
        return shardings, xs_tree

    # TODO: Add support for static args (e.g. via linear_util.WrappedFun)
    def dump_shardings(fun, name):
        in_shardings, in_tree = None, None
        out_shardings, out_tree = None, None

        @wraps(fun)
        def wrapper(*args):
            nonlocal in_shardings, in_tree
            nonlocal out_shardings, out_tree

            print(f"{name} args: {args}")
            res = fun(*args)
            print(f"{name} res: {res}")

            print(f"out_shardings: {out_shardings}")
            
            # NOTE: Shardings are captured only from the first invocation!
            if out_shardings is None:
                print("Populating in_shardings, out_shardings")
                in_shardings, in_tree = register_store_callbacks(args, f"{name} in")
                out_shardings, out_tree = register_store_callbacks(res, f"{name} out")
            return res

        return (
            wrapper,
            lambda: jax.tree.unflatten(in_tree, in_shardings),
            lambda: jax.tree.unflatten(out_tree, out_shardings),
        )

    return dump_shardings


def test_sharding_extractor_1():
    mesh = Mesh(np.array(jax.devices()).reshape((4, 2)), ("a", "b"))
    s0 = NamedSharding(mesh, P())
    s1 = NamedSharding(mesh, P("a"))
    s2 = NamedSharding(mesh, P(None, "b"))
    s3 = NamedSharding(mesh, P("a", "b"))

    def foo(x, y):
        z = jax.lax.with_sharding_constraint(x[0] * x[1], s1)
        return z, z + y

    dump_shardings = sharding_extractor()
    foo, ins_thunk, outs_thunk = dump_shardings(foo, "foo")

    foo = jax.jit(foo, in_shardings=((s0, s1), s2))
    arr = jnp.ones((16, 16))
    foo((arr, arr), arr)[0].block_until_ready()

    def specs(ss):
        return jax.tree.map(lambda s: s.spec, ss)

    in_specs = specs(ins_thunk())
    out_specs = specs(outs_thunk())
    assert in_specs == specs(((s0, s1), s2)), f"unexpected in specs {in_specs=}"
    assert out_specs == specs((s1, s3)), f"unexpected out specs {out_specs=}"


def test_sharding_extractor_2():
    # ---- Setup mesh and specs ---- #
    mesh = Mesh(np.array(jax.devices()).reshape((4, 2)), ("data", "stage"))
    s0 = NamedSharding(mesh, P())

    # ---- Setup sections and step_fn ---- #
    def section_1(x, a):
        return x + a

    def section_2(x, b):
        return x * b

    section_fns = {
        "section_1": section_1,
        "section_2": section_2,
    }

    def step_fn(x, a, b):
        ctx = mpmd.get_context()
        x = ctx.section("section_1")(x, a)
        x = ctx.section("section_2")(x, b)
        return x

    # ---- Setup sharding extractor ---- #
    in_shardings_thunk = {}
    out_shardings_thunk = {}

    dump_shardings = sharding_extractor()

    def dump_section_shardings(section_name, section_fn, **kwargs):
        wrapped, in_shardings_thunk[section_name], out_shardings_thunk[section_name] = (
            dump_shardings(section_fn, section_name)
        )
        return wrapped

    # ---- Setup inputs ---- #
    x = jnp.zeros((8, 8), dtype=jnp.float32, device=s0)
    a = jnp.zeros((8, 8), dtype=jnp.float32, device=s0)
    b = jnp.zeros((8, 8), dtype=jnp.float32, device=s0)

    # ---- Run test ---- #
    ctx = mpmd.Context(
        mesh=mesh,
        tracing_for_inference=True,
        section_fns=section_fns,
        section_decorator=dump_section_shardings,
    )

    def make_shape_dtype(arr):
        return jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=arr.sharding)

    with mpmd.set_context(ctx), mesh:
        _ = jax.jit(
            step_fn,
            in_shardings=(s0, s0, s0),
            out_shardings=s0,
        ).lower(*jax.tree.map(make_shape_dtype, (x, a, b))).compile()

    section_in_shardings = {
        section_name: thunk() for section_name, thunk in in_shardings_thunk.items()
    }
    section_out_shardings = {
        section_name: thunk() for section_name, thunk in out_shardings_thunk.items()
    }

    # ---- Print outputs ---- #
    print("In shardings:")
    print(section_in_shardings)
    print()

    print("Out shardings:")
    print(section_out_shardings)
    print()


if __name__ == "__main__":
    test_sharding_extractor_2()
