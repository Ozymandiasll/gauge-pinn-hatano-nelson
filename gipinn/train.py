# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .lattice import H_phi, Lattice, P_phi, frac_gauge_factor, initial_psi
from .losses import causal_weights, conservation_loss, gradnorm_weights, site_residual_sq
from .network import init_mlp, make_field_fn, window_field

__all__ = ["TrainConfig", "TrainResult", "train_window", "train_sequential", "predict_psi", "rel_l2", "window_index", "initial_u"]

METHODS = ("gipinn", "gipinn_phi", "vanilla", "vanilla_rebalanced")

@dataclass
class TrainConfig:
    method: str = "gipinn"
    alpha: float = 0.25
    hidden: tuple = (64, 64, 64)
    n_windows: int = 20
    z_max: float = 10.0
    n_colloc: int = 200
    adam_steps: int = 800
    adam_lr: float = 1e-3
    lbfgs_steps: int = 2500
    causal_eps: tuple = (1e-2, 1e-1, 1.0)
    lambda_cons: float = 0.0
    cons_start: float = 0.5
    rebalance_every: int = 100
    seed: int = 0

    def frame(self) -> float:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        return {"gipinn": float(self.alpha), "gipinn_phi": 1.0, "vanilla": 0.0, "vanilla_rebalanced": 0.0}[self.method]

@dataclass
class TrainResult:
    cfg: TrainConfig
    windows: list
    params: list
    starts: list
    history: list
    wall_time: float

def window_index(bounds, z: float) -> int:
    right = np.array([b for _, b in bounds])
    return int(min(len(bounds) - 1, np.searchsorted(right, z, side="left")))

def _collocation(key, za, zb, n):
    u = jax.random.uniform(key, (max(n - 2, 0),), dtype=jnp.float64)
    return jnp.sort(jnp.concatenate([jnp.array([za]), za + (zb - za) * u, jnp.array([zb])]))

def _make_loss(lat: Lattice, cfg: TrainConfig, alpha: float):
    use_cons = cfg.method == "gipinn_phi"

    def loss(params, za, zb, u_a, z_pts, eps_c, lam_cons, site_w, frozen_w):
        field_fn = make_field_fn(params, za, zb, u_a)
        r2 = site_residual_sq(field_fn, lat, z_pts, alpha, None, None, None)
        per_point = jnp.mean(r2, axis=1)
        w = jnp.where(frozen_w >= 0.0, frozen_w, causal_weights(per_point, eps_c))
        w = w / jnp.sum(w)
        site_l = jnp.sum(w[:, None] * r2, axis=0)
        total = jnp.sum(site_w * site_l) / jnp.sum(site_w)
        if use_cons:
            total = total + lam_cons * conservation_loss(field_fn, lat, z_pts, P_phi(u_a, jnp), H_phi(lat, u_a, jnp))
        return total, per_point

    def site_loss_vec(params, za, zb, u_a, z_pts, frozen_w):
        field_fn = make_field_fn(params, za, zb, u_a)
        r2 = site_residual_sq(field_fn, lat, z_pts, alpha, None, None, None)
        w = frozen_w / jnp.sum(frozen_w)
        return jnp.sum(w[:, None] * r2, axis=0)

    return loss, site_loss_vec

def train_window(lat: Lattice, cfg: TrainConfig, params0, za: float, zb: float, u_a, key):
    alpha = cfg.frame()
    t0 = time.time()
    N = lat.N
    z_pts = _collocation(key, za, zb, cfg.n_colloc)
    u_a = jnp.asarray(u_a)
    loss, site_loss_vec = _make_loss(lat, cfg, alpha)
    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    site_jac = jax.jit(jax.jacrev(site_loss_vec))

    opt = optax.adam(cfg.adam_lr)
    opt_state = opt.init(params0)

    @jax.jit
    def adam_step(params, opt_state, eps_c, lam, site_w, frozen_w):
        (val, per_point), grads = value_and_grad(params, za, zb, u_a, z_pts, eps_c, lam, site_w, frozen_w)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, val, per_point

    params = params0
    site_w = jnp.ones(N, dtype=jnp.float64)
    no_freeze = -jnp.ones(z_pts.shape[0], dtype=jnp.float64)
    n_stage = max(1, cfg.adam_steps // len(cfg.causal_eps))
    cons_on = int(cfg.cons_start * cfg.adam_steps)
    loss_adam = []
    per_point = jnp.zeros_like(z_pts)
    eps_c = cfg.causal_eps[-1]
    for step in range(cfg.adam_steps):
        eps_c = cfg.causal_eps[min(step // n_stage, len(cfg.causal_eps) - 1)]
        lam = cfg.lambda_cons if (cfg.method == "gipinn_phi" and step >= cons_on) else 0.0
        if cfg.method == "vanilla_rebalanced" and step > 0 and step % cfg.rebalance_every == 0:
            frozen = causal_weights(per_point, eps_c)
            J = site_jac(params, za, zb, u_a, z_pts, frozen)
            norms = jnp.sqrt(sum(jnp.sum(jnp.reshape(j, (N, -1)) ** 2, axis=1) for j in jax.tree_util.tree_leaves(J)))
            site_w = gradnorm_weights(norms, site_w, alpha=0.9)
        params, opt_state, val, per_point = adam_step(params, opt_state, eps_c, lam, site_w, no_freeze)
        if step % 10 == 0:
            loss_adam.append(float(val))

    frozen_w = causal_weights(per_point, eps_c)
    lam = cfg.lambda_cons if cfg.method == "gipinn_phi" else 0.0
    flat0, unravel = ravel_pytree(params)
    loss_lbfgs = []

    def fun(x):
        p = unravel(jnp.asarray(x))
        (val, _), grads = value_and_grad(p, za, zb, u_a, z_pts, eps_c, lam, site_w, frozen_w)
        loss_lbfgs.append(float(val))
        return float(val), np.asarray(ravel_pytree(grads)[0], dtype=np.float64)

    if cfg.lbfgs_steps > 0:
        sol = minimize(
            fun, np.asarray(flat0, dtype=np.float64), jac=True, method="L-BFGS-B",
            options={"maxiter": cfg.lbfgs_steps, "maxfun": 2 * cfg.lbfgs_steps, "maxcor": 50, "gtol": 1e-12, "ftol": 0.0},
        )
        if sol.fun <= (loss_lbfgs[0] if loss_lbfgs else np.inf):
            params = unravel(jnp.asarray(sol.x))
    (final_val, _), _ = value_and_grad(params, za, zb, u_a, z_pts, eps_c, lam, site_w, frozen_w)
    history = {
        "loss_adam": loss_adam,
        "loss_lbfgs": loss_lbfgs,
        "final_loss": float(final_val),
        "site_w": np.asarray(site_w),
        "wall_time": time.time() - t0,
    }
    return params, history

def initial_u(lat: Lattice, alpha: float):
    return jnp.asarray(initial_psi(lat)) / frac_gauge_factor(lat, alpha, jnp)

def train_sequential(lat: Lattice, cfg: TrainConfig) -> TrainResult:
    alpha = cfg.frame()
    t0 = time.time()
    N = lat.N
    L = cfg.z_max / cfg.n_windows
    windows = [(k * L, (k + 1) * L) for k in range(cfg.n_windows)]
    key = jax.random.PRNGKey(cfg.seed)
    u = initial_u(lat, alpha)
    params_list, starts, history = [], [], []
    for za, zb in windows:
        key, k_init, k_pts = jax.random.split(key, 3)
        params0 = init_mlp(k_init, [1, *cfg.hidden, 2 * N])
        starts.append(np.asarray(u))
        params, hist = train_window(lat, cfg, params0, za, zb, u, k_pts)
        params_list.append(params)
        history.append(hist)
        u = window_field(params, zb, za, zb, u)
    return TrainResult(cfg=cfg, windows=windows, params=params_list, starts=starts, history=history,
                       wall_time=time.time() - t0)

def predict_psi(res: TrainResult, lat: Lattice, z_eval: np.ndarray) -> np.ndarray:
    alpha = res.cfg.frame()
    s = np.asarray(frac_gauge_factor(lat, alpha))
    z_eval = np.asarray(z_eval, dtype=float)
    out = np.zeros((len(z_eval), lat.N), dtype=complex)
    idx = np.array([window_index(res.windows, z) for z in z_eval])
    for k, (za, zb) in enumerate(res.windows):
        sel = np.where(idx == k)[0]
        if sel.size == 0:
            continue
        u_a = jnp.asarray(res.starts[k])
        f = jax.vmap(lambda z: window_field(res.params[k], z, za, zb, u_a))
        out[sel] = np.asarray(f(jnp.asarray(z_eval[sel]))) * s[None, :]
    return out

def rel_l2(psi_pred: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(psi_pred - psi_ref, axis=1)
    den = np.linalg.norm(psi_ref, axis=1)
    return num / den
