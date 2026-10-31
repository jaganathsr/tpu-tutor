"""Generates the compiler-stage listings shown in Chapter 9.

Runs on CPU with 4 simulated devices, so the listings are real JAX/XLA output
but CPU-flavoured: TPU HLO differs in layout tiling (T(8,128)), fusion choices
and memory-space annotations. Needs JAX (tested with 0.8.2):

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        /path/to/venv/bin/python scripts/ch09_dump.py
"""
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

OUT = os.path.join(os.path.dirname(__file__), "..", "content", "chapters", "ch09", "dumps")
os.makedirs(OUT, exist_ok=True)


def save(name, text, keep=None, limit=40, from_entry=False):
    lines = text.splitlines()
    if from_entry:
        lines = lines[next(i for i, l in enumerate(lines) if l.startswith("ENTRY")):]
    if keep:
        lines = [l for l in lines if re.search(keep, l)]
    text = "\n".join(lines[:limit])
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text.rstrip() + "\n")
    print(f"--- {name} ({len(lines)} lines kept)\n{text}\n")


mesh = jax.make_mesh((2, 2), ("data", "model"))


def mlp(x, w1, w2):
    return jax.nn.gelu(x @ w1) @ w2


B, D, F = 8, 512, 2048
x = jax.device_put(jnp.ones((B, D), jnp.bfloat16), NamedSharding(mesh, P("data", None)))
w1 = jax.device_put(jnp.ones((D, F), jnp.bfloat16), NamedSharding(mesh, P(None, "model")))
w2 = jax.device_put(jnp.ones((F, D), jnp.bfloat16), NamedSharding(mesh, P("model", None)))

# Stage 1: jaxpr.
save("1_jaxpr.txt", str(jax.make_jaxpr(mlp)(x, w1, w2)), limit=30)

# Stage 2: StableHLO with sharding annotations (what jit hands to XLA).
f = jax.jit(mlp, out_shardings=NamedSharding(mesh, P("data", None)))
lowered = f.lower(x, w1, w2)
save("2_stablehlo.txt", lowered.as_text(), keep=r"func\.func|sharding|dot_general|stablehlo\.(tanh|multiply|add)|return|mesh", limit=24)

# Stage 3: compiled, partitioned HLO: per-device shapes and the inserted collective.
compiled = lowered.compile()
save("3_compiled_hlo.txt", compiled.as_text(), keep=r"ENTRY|all-reduce|all-gather|reduce-scatter|dot\(|ROOT|parameter\(", limit=30, from_entry=True)

# Explicit sharding mode: an ambiguous contraction is a trace-time error.
try:
    emesh = jax.make_mesh((2, 2), ("X", "Y"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
    with jax.sharding.use_mesh(emesh) if hasattr(jax.sharding, "use_mesh") else jax.set_mesh(emesh):
        a = jax.device_put(jnp.ones((8, 512)), NamedSharding(emesh, P("X", "Y")))
        b = jax.device_put(jnp.ones((512, 256)), NamedSharding(emesh, P("Y", None)))
        print(jax.typeof(a))
        jnp.dot(a, b)
        save("4_explicit_error.txt", "no error raised (behaviour differs in this JAX version)")
except Exception as e:  # noqa: BLE001
    save("4_explicit_error.txt", f"{type(e).__name__}: {e}", limit=6)

# shard_map: per-device view with an explicit collective.
from functools import partial

shard_map = getattr(jax, "shard_map", None)
if shard_map is None:
    from jax.experimental.shard_map import shard_map  # older JAX


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(P("data", "model"), P("model", None)), out_specs=P("data", None))
def matmul_psum(a_blk, b_blk):
    return jax.lax.psum(a_blk @ b_blk, "model")


a = jax.device_put(jnp.ones((B, D), jnp.bfloat16), NamedSharding(mesh, P("data", "model")))
b = jax.device_put(jnp.ones((D, D), jnp.bfloat16), NamedSharding(mesh, P("model", None)))
save("5_shard_map_hlo.txt", matmul_psum.lower(a, b).compile().as_text(), keep=r"ENTRY|all-reduce|dot\(|ROOT|parameter\(", limit=16, from_entry=True)
print("jax", jax.__version__)
