# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

__all__ = ["init_mlp", "mlp_apply", "window_field", "field_and_dz", "make_field_fn"]

def init_mlp(key, sizes: Sequence[int]):
    params = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / (d_in + d_out))
        W = scale * jax.random.normal(sub, (d_in, d_out), dtype=jnp.float64)
        b = jnp.zeros(d_out, dtype=jnp.float64)
        params.append((W, b))
    return params

def mlp_apply(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b

def window_field(params, z, za, zb, u_a):
    N = u_a.shape[0]
    s = 2.0 * (z - za) / (zb - za) - 1.0
    out = mlp_apply(params, jnp.array([s]))
    return u_a + (z - za) * (out[:N] + 1j * out[N:])

def field_and_dz(params, z, za, zb, u_a):
    f = lambda zz: window_field(params, zz, za, zb, u_a)
    return jax.jvp(f, (jnp.asarray(z, dtype=jnp.float64),), (jnp.ones((), dtype=jnp.float64),))

def make_field_fn(params, za, zb, u_a) -> Callable:
    def fn(z):
        return field_and_dz(params, z, za, zb, u_a)

    return fn
