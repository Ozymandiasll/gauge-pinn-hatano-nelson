# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .integrate import solve_psi
from .lattice import H_phi, Lattice, P_phi, frac_gauge_factor, gauge_factor, initial_psi, neighbors
from .losses import causal_weights, conservation_loss, data_loss, site_residual_sq
from .network import init_mlp, make_field_fn, window_field
from .train import window_index

__all__ = ["InverseConfig", "InverseResult", "synthetic_intensity", "train_inverse", "param_error", "eps_metrics"]

@dataclass
class InverseConfig:
    alpha: float = 0.5
    hidden: tuple = (64, 64, 64)
    n_windows: int = 20
    z_max: float = 10.0
    n_colloc: int = 150
    adam_steps: int = 4000
    adam_lr: float = 1e-3
    param_lr: float = 1e-2
    lbfgs_steps: int = 2000
    causal_eps: tuple = (1e-2, 1e-1, 1.0)
    lambda_cons: float = 0.0
    lambda_data: float = 1.0
    data_mode: str = "abs"
    learn_eps: bool = True
    h_init: float = 0.1
    g_init: float = 1.0
    warmup_frac: float = 0.3
    param_mode: str = "varpro"

    ridge_eps: float = 0.0
    eps_warmup_frac: float = 0.0
    seed: int = 0

@dataclass
class InverseResult:
    h: float
    g: float
    eps: np.ndarray
    history: dict
    wall_time: float
    params: Any = field(default=None, repr=False)

def synthetic_intensity(lat_true: Lattice, z_planes: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    I = np.abs(solve_psi(lat_true, np.asarray(z_planes, dtype=float))) ** 2
    if sigma > 0:
        rng = np.random.default_rng(seed)
        I = I * (1.0 + sigma * rng.normal(size=I.shape))
    return np.maximum(I, 0.0)

def param_error(est: float, true: float) -> float:
    return abs(est - true) / abs(true) if abs(true) >= 0.1 else abs(est - true)

def eps_metrics(eps_est, eps_true, mask=None) -> dict:
    a, b = np.asarray(eps_est, dtype=float), np.asarray(eps_true, dtype=float)
    if mask is not None:
        a, b = a[np.asarray(mask, bool)], b[np.asarray(mask, bool)]
    a, b = a - a.mean(), b - b.mean()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return {"corr": corr, "rmse": float(np.sqrt(np.mean((a - b) ** 2)))}

def _varpro_terms(field_fn, z_pts, s2, hop_m, hop_p):
    def one(z):
        u, du = field_fn(z)
        um, up = neighbors(u, jnp)
        return 1j * du + hop_m * um + hop_p * up, s2 * jnp.abs(u) ** 2 * u, u

    return jax.vmap(one)(z_pts)

def varpro_eps(Rp, B, w, ridge=0.0):
    wr = w[:, None]
    b = jnp.sum(wr * jnp.abs(B) ** 2, axis=0) + ridge
    q = jnp.sum(wr * jnp.real(jnp.conj(Rp) * B), axis=0)
    return -q / jnp.maximum(b, 1e-30)

def _collocation(key, za, zb, n):
    u = jax.random.uniform(key, (max(n - 2, 0),), dtype=jnp.float64)
    return jnp.sort(jnp.concatenate([jnp.array([za]), za + (zb - za) * u, jnp.array([zb])]))

def train_inverse(N: int, n0: int, z_planes: np.ndarray, I_obs: np.ndarray, cfg: InverseConfig,
                  eps_known=None) -> InverseResult:
    t0 = time.time()
    lat = Lattice(N=N, n0=n0)
    L = cfg.z_max / cfg.n_windows
    windows = [(k * L, (k + 1) * L) for k in range(cfg.n_windows)]
    key = jax.random.PRNGKey(cfg.seed)
    keys = jax.random.split(key, 2 * cfg.n_windows)
    nets = [init_mlp(keys[2 * k], [1, *cfg.hidden, 2 * N]) for k in range(cfg.n_windows)]
    z_pts = [_collocation(keys[2 * k + 1], za, zb, cfg.n_colloc) for k, (za, zb) in enumerate(windows)]
    z_all = jnp.concatenate(z_pts)
    phi0 = jnp.asarray(initial_psi(lat))
    z_planes = np.asarray(z_planes, dtype=float)
    I_obs = jnp.asarray(I_obs, dtype=jnp.float64)
    plane_window = np.array([window_index(windows, z) for z in z_planes])
    eps_fixed = None if cfg.learn_eps else jnp.asarray(np.zeros(N) if eps_known is None else eps_known, dtype=jnp.float64)

    params = {"nets": nets, "h": jnp.asarray(cfg.h_init, dtype=jnp.float64), "g": jnp.asarray(cfg.g_init, dtype=jnp.float64)}
    if cfg.learn_eps:
        params["eps"] = jnp.zeros(N, dtype=jnp.float64)

    def _eps(p):
        return p["eps"] if cfg.learn_eps else eps_fixed

    def _starts(p):
        starts, u = [], jnp.sqrt(p["g"]) * phi0
        for k, (za, zb) in enumerate(windows):
            starts.append(u)
            u = window_field(p["nets"][k], zb, za, zb, u)
        return starts

    varpro = cfg.param_mode == "varpro"
    if cfg.param_mode not in ("varpro", "gradient"):
        raise ValueError(f"unknown param_mode {cfg.param_mode!r}")
    alpha = float(cfg.alpha)
    if cfg.lambda_cons > 0 and alpha != 1.0:
        raise ValueError("the conservation penalty is only defined in the full gauge frame (alpha = 1)")

    def loss(p, eps_c, lam_cons, frozen_w, eps_on=1.0):
        h, g = p["h"], p["g"]
        starts = _starts(p)
        s2 = frac_gauge_factor(lat, alpha, jnp, h) ** 2
        hop_m, hop_p = jnp.exp((1.0 - alpha) * h), jnp.exp(-(1.0 - alpha) * h)
        RAB = [_varpro_terms(make_field_fn(p["nets"][k], za, zb, starts[k]), z_pts[k], s2, hop_m, hop_p)
               for k, (za, zb) in enumerate(windows)]
        Rp = jnp.concatenate([t[0] + t[1] for t in RAB])
        B = jnp.concatenate([t[2] for t in RAB])

        if varpro and cfg.learn_eps:
            pre = jnp.mean(jnp.abs(Rp + p["eps"][None, :] * B) ** 2, axis=1) / g
            w = jnp.where(frozen_w >= 0.0, frozen_w, causal_weights(pre, eps_c))
            w = w / jnp.sum(w)
            eps = eps_on * varpro_eps(Rp, B, w, cfg.ridge_eps * g)
            per_point = jnp.mean(jnp.abs(Rp + eps[None, :] * B) ** 2, axis=1) / g
            res_term = jnp.sum(w * per_point)
        else:
            eps = _eps(p)
            per_point = jnp.mean(jnp.abs(Rp + eps[None, :] * B) ** 2, axis=1) / g
            w = jnp.where(frozen_w >= 0.0, frozen_w, causal_weights(per_point, eps_c))
            res_term = jnp.sum(w * per_point) / jnp.sum(w)
        cons = 0.0
        gf = frac_gauge_factor(lat, alpha, jnp, h) / jnp.sqrt(g)
        psi_pred = jnp.stack([
            window_field(p["nets"][k], z, windows[k][0], windows[k][1], starts[k]) * gf
            for z, k in zip(z_planes, plane_window)
        ])
        d_term = data_loss(psi_pred, I_obs, cfg.data_mode)
        return res_term + lam_cons * cons + cfg.lambda_data * d_term, (per_point, res_term, d_term, g, eps)

    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    labels = {"nets": "net", "h": "phys", "g": "phys"}
    if cfg.learn_eps:
        labels["eps"] = "phys"
    opt = optax.multi_transform({"net": optax.adam(cfg.adam_lr), "phys": optax.adam(cfg.param_lr)}, labels)
    opt_state = opt.init(params)
    phys_mask = jax.tree_util.tree_map_with_path(lambda path, _: 0.0 if path[0].key in ("h", "g", "eps") else 1.0, params)

    def _store(p, aux):
        if varpro and cfg.learn_eps:
            p = dict(p); p["eps"] = aux[4]
        return p

    @jax.jit
    def adam_step(p, s, eps_c, lam, frozen_w, phys_on, eps_on):
        (val, aux), grads = value_and_grad(p, eps_c, lam, frozen_w, eps_on)
        grads = jax.tree_util.tree_map(lambda g_, m: g_ * (m + (1.0 - m) * phys_on), grads, phys_mask)
        updates, s = opt.update(grads, s, p)
        return _store(optax.apply_updates(p, updates), aux), s, val, aux

    no_freeze = -jnp.ones(z_all.shape[0], dtype=jnp.float64)
    n_stage = max(1, cfg.adam_steps // len(cfg.causal_eps))
    hist = {"loss": [], "res": [], "data": [], "h_traj": [], "g_traj": []}
    eps_c = cfg.causal_eps[-1]
    per_point = jnp.zeros_like(z_all)
    warmup = int(cfg.warmup_frac * cfg.adam_steps)
    eps_warmup = int(cfg.eps_warmup_frac * cfg.adam_steps)
    for step in range(cfg.adam_steps):
        eps_c = cfg.causal_eps[min(step // n_stage, len(cfg.causal_eps) - 1)]
        phys_on = 0.0 if step < warmup else 1.0
        eps_on = 0.0 if step < eps_warmup else 1.0
        params, opt_state, val, (per_point, r, d, _, _) = adam_step(params, opt_state, eps_c, cfg.lambda_cons, no_freeze, phys_on, eps_on)
        if step % 10 == 0:
            hist["loss"].append(float(val)); hist["res"].append(float(r)); hist["data"].append(float(d))
            hist["h_traj"].append(float(params["h"])); hist["g_traj"].append(float(params["g"]))

    frozen_w = causal_weights(per_point, eps_c)
    flat0, unravel = ravel_pytree(params)
    lbfgs_loss = []

    def fun(x):
        p = unravel(jnp.asarray(x))
        (val, aux), grads = value_and_grad(p, eps_c, cfg.lambda_cons, frozen_w)
        if varpro:
            grads = jax.tree_util.tree_map(lambda g_, m: g_ * m, grads, phys_mask_varpro)
        lbfgs_loss.append(float(val))
        return float(val), np.asarray(ravel_pytree(grads)[0], dtype=np.float64)

    phys_mask_varpro = jax.tree_util.tree_map_with_path(lambda path, _: 0.0 if path[0].key == "eps" else 1.0, params)
    if cfg.lbfgs_steps > 0:
        sol = minimize(fun, np.asarray(flat0, dtype=np.float64), jac=True, method="L-BFGS-B",
                       options={"maxiter": cfg.lbfgs_steps, "maxfun": 2 * cfg.lbfgs_steps, "maxcor": 50, "gtol": 1e-12, "ftol": 0.0})
        if sol.fun <= (lbfgs_loss[0] if lbfgs_loss else np.inf):
            params = unravel(jnp.asarray(sol.x))
    (final, aux), _ = value_and_grad(params, eps_c, cfg.lambda_cons, frozen_w)
    params = _store(params, aux)
    r, d = aux[1], aux[2]
    hist["loss_lbfgs"] = lbfgs_loss
    hist["final_loss"], hist["final_res"], hist["final_data"] = float(final), float(r), float(d)
    hist["h_traj"].append(float(params["h"])); hist["g_traj"].append(float(params["g"]))
    eps_out = np.asarray(_eps(params))
    return InverseResult(h=float(params["h"]), g=float(params["g"]), eps=eps_out, history=hist,
                         wall_time=time.time() - t0, params=params)

from .integrate import rk4_psi_jax

__all__ += ["BaselineResult", "baseline_rk4_fit"]

@dataclass
class BaselineResult:
    h: float
    g: float
    eps: np.ndarray
    per_restart: list
    success_rate: float
    wall_time: float

def baseline_rk4_fit(N: int, n0: int, z_planes, I_obs, mode: str, learn_eps: bool, true_params: dict | None = None,
                     n_restarts: int = 10, adam_steps: int = 2000, lr: float = 1e-2, lbfgs_steps: int = 500,
                     n_steps: int = 1000, seed: int = 0, eps_known=None, init_box=None) -> BaselineResult:
    t0 = time.time()
    z_planes = np.asarray(z_planes, dtype=float)
    z_max = float(z_planes[-1])
    dz = z_max / n_steps
    out_steps = jnp.asarray(np.rint(z_planes / dz).astype(int))
    I_obs = jnp.asarray(I_obs, dtype=jnp.float64)
    eps_fixed = jnp.asarray(np.zeros(N) if eps_known is None else eps_known, dtype=jnp.float64)
    box = {"h": (0.0, 0.4), "g": (0.2, 4.0)}
    if init_box:
        box.update(init_box)

    def unpack(p):
        return p["h"], p["g"], (p["eps"] if learn_eps else eps_fixed)

    def loss(p):
        h, g, eps = unpack(p)
        psi = rk4_psi_jax(h, g, eps, N, n0, z_max, n_steps, out_steps)
        return data_loss(psi, I_obs, mode)

    vg = jax.jit(jax.value_and_grad(loss))
    opt = optax.adam(lr)

    @jax.jit
    def step(p, s):
        val, grads = vg(p)
        upd, s = opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s, val

    rng = np.random.default_rng(seed)
    per_restart = []
    for r in range(n_restarts):
        p = {"h": jnp.asarray(rng.uniform(*box["h"])), "g": jnp.asarray(rng.uniform(*box["g"]))}
        if learn_eps:
            p["eps"] = jnp.zeros(N, dtype=jnp.float64)
        s = opt.init(p)
        for _ in range(adam_steps):
            p, s, val = step(p, s)
        flat, unravel = ravel_pytree(p)

        def fun(x):
            val, grads = vg(unravel(jnp.asarray(x)))
            return float(val), np.asarray(ravel_pytree(grads)[0], dtype=np.float64)

        if lbfgs_steps > 0:
            sol = minimize(fun, np.asarray(flat, dtype=np.float64), jac=True, method="L-BFGS-B",
                           options={"maxiter": lbfgs_steps, "maxfun": 2 * lbfgs_steps, "gtol": 1e-14, "ftol": 0.0})
            if sol.fun <= float(val):
                p, val = unravel(jnp.asarray(sol.x)), sol.fun
        h, g, eps = unpack(p)
        rec = {"h": float(h), "g": float(g), "eps": np.asarray(eps), "loss": float(val), "success": None}
        if true_params is not None:
            ok = param_error(rec["h"], true_params["h"]) <= 0.05 and param_error(rec["g"], true_params["g"]) <= 0.05
            if learn_eps and "eps" in true_params:
                ok = ok and eps_metrics(rec["eps"], true_params["eps"], true_params.get("eps_mask"))["corr"] >= 0.9
            rec["success"] = bool(ok)
        per_restart.append(rec)
    best = min(per_restart, key=lambda d: d["loss"])
    succ = [d["success"] for d in per_restart if d["success"] is not None]
    return BaselineResult(h=best["h"], g=best["g"], eps=best["eps"], per_restart=per_restart,
                          success_rate=float(np.mean(succ)) if succ else float("nan"), wall_time=time.time() - t0)
