# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import jax
import jax.numpy as jnp

from .lattice import H_phi, Lattice, P_phi, residual_frac, residual_phi, residual_psi

__all__ = [
    "site_residual_sq",
    "residual_loss",
    "site_losses",
    "causal_weights",
    "conservation_loss",
    "data_loss",
    "gradnorm_weights",
]

def _residual(frame):
    if isinstance(frame, str):
        if frame == "phi":
            return residual_phi
        if frame == "psi":
            return residual_psi
        raise ValueError(f"unknown frame {frame!r}")
    alpha = float(frame)
    return lambda lat, u, du, xp=jnp, h=None, g=None, eps=None: residual_frac(lat, u, du, alpha, xp, h, g, eps)

def site_residual_sq(field_fn, lat: Lattice, z_pts, frame, h, g, eps):
    res = _residual(frame)

    def one(z):
        u, du = field_fn(z)
        r = res(lat, u, du, xp=jnp, h=h, g=g, eps=eps)
        return jnp.abs(r) ** 2

    return jax.vmap(one)(z_pts)

def residual_loss(field_fn, lat: Lattice, z_pts, frame: str, h=None, g=None, eps=None):
    r2 = site_residual_sq(field_fn, lat, z_pts, frame, h, g, eps)
    per_point = jnp.mean(r2, axis=1)
    return jnp.mean(per_point), per_point

def site_losses(field_fn, lat: Lattice, z_pts, frame: str, weights_pts=None, h=None, g=None, eps=None):
    r2 = site_residual_sq(field_fn, lat, z_pts, frame, h, g, eps)
    if weights_pts is None:
        return jnp.mean(r2, axis=0)
    w = weights_pts[:, None]
    return jnp.sum(w * r2, axis=0) / jnp.sum(w)

def causal_weights(per_point, eps_c: float):
    cum = jnp.concatenate([jnp.zeros(1, dtype=per_point.dtype), jnp.cumsum(per_point)[:-1]])
    return jax.lax.stop_gradient(jnp.exp(-eps_c * cum))

def conservation_loss(field_fn, lat: Lattice, z_pts, P0: float, H0: float, h=None, g=None, eps=None):
    def one(z):
        phi, _ = field_fn(z)
        return (P_phi(phi, jnp) - P0) ** 2 + (H_phi(lat, phi, jnp, h=h, g=g, eps=eps) - H0) ** 2

    return jnp.mean(jax.vmap(one)(z_pts))

def data_loss(psi_pred, I_obs, mode: str):
    I_pred = jnp.abs(psi_pred) ** 2
    if mode == "abs":
        return jnp.mean((I_pred - I_obs) ** 2)
    if mode == "rel":
        floor = 1e-3 * jnp.max(I_obs, axis=1, keepdims=True)
        return jnp.mean((I_pred - I_obs) ** 2 / (I_obs**2 + floor**2 + 1e-300))
    if mode == "norm":
        p = I_pred / jnp.maximum(jnp.sum(I_pred, axis=1, keepdims=True), 1e-30)
        q = I_obs / jnp.maximum(jnp.sum(I_obs, axis=1, keepdims=True), 1e-30)
        return jnp.mean((p - q) ** 2)
    raise ValueError(f"unknown data mode {mode!r}")

def gradnorm_weights(grad_norms, old_w, alpha: float = 0.9):
    w_hat = jnp.max(grad_norms) / jnp.maximum(grad_norms, 1e-30)
    return alpha * old_w + (1.0 - alpha) * w_hat
