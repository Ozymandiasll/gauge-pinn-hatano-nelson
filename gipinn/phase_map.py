# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .integrate import rk4_psi_jax
from .lattice import Lattice, com_shift, disorder, ipr, p_trap

__all__ = ["sweep_observables"]

def sweep_observables(N: int, n0: int, h: float, W_grid, g_grid, n_real: int, z_max: float = 10.0,
                      n_steps: int = 4000, seed: int = 0, row_cache=None, log=None) -> dict:
    W_grid = np.asarray(W_grid, dtype=float)
    g_grid = np.asarray(g_grid, dtype=float)
    lat = Lattice(N=N, n0=n0, h=h)
    out_steps = jnp.array([n_steps])

    def one(g, eps):
        psi = rk4_psi_jax(h, g, eps, N, n0, z_max, n_steps, out_steps)[0]
        return jnp.stack([ipr(psi, jnp), p_trap(lat, psi, 2, jnp), com_shift(lat, psi, jnp)])

    batch = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))
    g_j = jnp.asarray(g_grid)
    means = np.zeros((len(W_grid), len(g_grid), 3))
    stds = np.zeros_like(means)
    for i, W in enumerate(W_grid):
        path = row_cache(i) if row_cache is not None else None
        if path is not None and path.exists():
            with np.load(path) as d:
                obs = d["obs"]
        else:
            eps = jnp.asarray(np.stack([disorder(N, W, seed + r) for r in range(n_real)]))
            obs = np.asarray(batch(g_j, eps))
            if path is not None:
                np.savez(path, obs=obs)
            if log is not None:
                log(f"row {i + 1}/{len(W_grid)} (W = {W:g}) done")
        means[i] = obs.mean(axis=1)
        stds[i] = obs.std(axis=1)
    return {"ipr": means[:, :, 0], "p_trap": means[:, :, 1], "com": means[:, :, 2],
            "ipr_std": stds[:, :, 0], "p_trap_std": stds[:, :, 1], "com_std": stds[:, :, 2],
            "W_grid": W_grid, "g_grid": g_grid, "h": h, "n_real": n_real, "z_max": z_max, "n_steps": n_steps, "seed": seed}
