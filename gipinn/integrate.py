# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .lattice import Lattice, eps_array, gauge_factor, initial_psi, phi_to_psi, psi_to_phi, rhs_phi

__all__ = ["solve_phi", "solve_psi", "expm_psi", "rk4_psi_jax"]

def solve_phi(lat: Lattice, z_eval: np.ndarray, rtol: float = 1e-12, atol: float = 1e-12) -> np.ndarray:
    z_eval = np.asarray(z_eval, dtype=float)
    N = lat.N
    phi0 = psi_to_phi(lat, initial_psi(lat))
    y0 = np.concatenate([phi0.real, phi0.imag])

    def f(_z, y):
        d = rhs_phi(lat, y[:N] + 1j * y[N:])
        return np.concatenate([d.real, d.imag])

    sol = solve_ivp(f, (0.0, float(z_eval[-1])), y0, method="DOP853", t_eval=z_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y.T
    return y[:, :N] + 1j * y[:, N:]

def solve_psi(lat: Lattice, z_eval: np.ndarray, **kw) -> np.ndarray:
    return phi_to_psi(lat, solve_phi(lat, z_eval, **kw))

def _H_matrix(lat: Lattice) -> np.ndarray:
    N = lat.N
    H = np.diag(eps_array(lat)).astype(complex)
    idx = np.arange(N - 1)
    H[idx + 1, idx] = np.exp(lat.h)
    H[idx, idx + 1] = np.exp(-lat.h)
    return H

def expm_psi(lat: Lattice, z_eval: np.ndarray) -> np.ndarray:
    if lat.g != 0:
        raise ValueError("expm_psi is only valid for g = 0")
    H = _H_matrix(lat)
    psi0 = initial_psi(lat)
    return np.stack([expm(1j * H * z) @ psi0 for z in np.asarray(z_eval, dtype=float)])

def rk4_psi_jax(h, g, eps, N: int, n0: int, z_max: float, n_steps: int, out_steps):
    lat = Lattice(N=N, n0=n0)
    dz = z_max / n_steps
    phi0 = jnp.asarray(initial_psi(lat))

    def f(phi):
        return rhs_phi(lat, phi, xp=jnp, h=h, g=g, eps=eps)

    def step(phi, _):
        k1 = f(phi)
        k2 = f(phi + 0.5 * dz * k1)
        k3 = f(phi + 0.5 * dz * k2)
        k4 = f(phi + dz * k3)
        phi = phi + dz / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return phi, phi

    _, traj = jax.lax.scan(step, phi0, None, length=n_steps)
    traj = jnp.concatenate([phi0[None, :], traj], axis=0)
    phi_out = jnp.take(traj, jnp.asarray(out_steps), axis=0)
    return phi_out * gauge_factor(lat, jnp, h)[None, :]
