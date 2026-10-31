"""The Pallas TPU matmul shown in Chapter 10, checked against jnp.dot in
interpret mode (runs on CPU; on TPU drop interpret=True).

    JAX_PLATFORMS=cpu /path/to/venv/bin/python scripts/ch10_pallas_matmul.py
"""
import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_ref, w_ref, o_ref, acc_ref):
    # Grid axes (i, j, k): output tile (i, j), contraction step k (innermost).
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("bm", "bk", "bn", "interpret"))
def matmul(x, w, bm=256, bk=512, bn=256, interpret=False):
    m, k = x.shape
    _, n = w.shape
    return pl.pallas_call(
        matmul_kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(x, w)


if __name__ == "__main__":
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.normal(kx, (512, 1024), jnp.bfloat16)
    w = jax.random.normal(kw, (1024, 512), jnp.bfloat16)
    out = matmul(x, w, interpret=True)
    ref = jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    err = jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32)))
    print("jax", jax.__version__, "max abs err", float(err), "shape", out.shape)
    # VMEM per grid step: double-buffered x and w blocks, the output block and the f32 accumulator.
    bm, bk, bn = 256, 512, 256
    vmem = 2 * (bm * bk + bk * bn) * 2 + 2 * bm * bn * 2 + bm * bn * 4
    print(f"VMEM for blocks (double-buffered inputs and output, f32 acc): {vmem/2**20:.2f} MiB")
    print(f"Arithmetic intensity per step (HBM bytes of the two input blocks): {2*bm*bk*bn/((bm*bk+bk*bn)*2):.0f} FLOP/B")

    print("\nTile choice against Ironwood's 313 FLOP/B ridge (bf16, f32 accumulator):")
    for bm, bk, bn in ((256, 512, 256), (512, 512, 512), (1024, 512, 1024), (2048, 512, 2048)):
        ai = bm * bn / (bm + bn)                      # FLOPs per byte of streamed input blocks
        vmem = 2 * (bm * bk + bk * bn) * 2 + 2 * bm * bn * 2 + bm * bn * 4
        print(f"  {bm:5}x{bk}x{bn:<5}: intensity {ai:5.0f} FLOP/B, VMEM {vmem/2**20:5.1f} MiB of 64 MiB per TensorCore")
