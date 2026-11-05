# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gipinn.lattice import Lattice
from gipinn.network import init_mlp, window_field
from gipinn.train import TrainConfig, TrainResult, initial_u, train_window

from .common import RESULTS_DIR

def _ckpt_dir(results_dir):
    d = (Path(results_dir) if results_dir is not None else RESULTS_DIR) / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_window(path: Path, params, start, hist: dict):
    arrays = {"start": np.asarray(start)}
    for i, (W, b) in enumerate(params):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(W), np.asarray(b)
    arrays["loss_adam"] = np.asarray(hist["loss_adam"], dtype=float)
    arrays["loss_lbfgs"] = np.asarray(hist["loss_lbfgs"], dtype=float)
    arrays["site_w"] = np.asarray(hist["site_w"], dtype=float)
    arrays["scalars"] = np.array([hist["final_loss"], hist["wall_time"]], dtype=float)
    np.savez(path, **arrays)

def _load_window(path: Path, n_layers: int):
    with np.load(path) as d:
        params = [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n_layers)]
        start = d["start"]
        hist = {
            "loss_adam": d["loss_adam"].tolist(),
            "loss_lbfgs": d["loss_lbfgs"].tolist(),
            "site_w": d["site_w"],
            "final_loss": float(d["scalars"][0]),
            "wall_time": float(d["scalars"][1]),
        }
    return params, start, hist

def run_training_resumable(name: str, lat: Lattice, cfg: TrainConfig, results_dir=None,
                           time_budget_s: float = 480.0, log=print):
    alpha = cfg.frame()
    t0 = time.time()
    N = lat.N
    L = cfg.z_max / cfg.n_windows
    windows = [(k * L, (k + 1) * L) for k in range(cfg.n_windows)]
    n_layers = len(cfg.hidden) + 1
    ckpt = _ckpt_dir(results_dir)
    key = jax.random.PRNGKey(cfg.seed)
    u = initial_u(lat, alpha)
    params_list, starts, history = [], [], []
    trained_now = 0
    for k, (za, zb) in enumerate(windows):
        key, k_init, k_pts = jax.random.split(key, 3)
        path = ckpt / f"{name}_w{k}.npz"
        if path.exists():
            params, start, hist = _load_window(path, n_layers)
            u = jnp.asarray(start)
        else:
            if trained_now > 0 and (time.time() - t0) > time_budget_s:
                log(f"[{name}] time budget spent after window {k - 1}; call again to resume")
                return None
            params0 = init_mlp(k_init, [1, *cfg.hidden, 2 * N])
            params, hist = train_window(lat, cfg, params0, za, zb, u, k_pts)
            _save_window(path, params, u, hist)
            trained_now += 1
            log(f"[{name}] window {k} done: final loss {hist['final_loss']:.3e}, {hist['wall_time']:.0f} s")
        params_list.append(params)
        starts.append(np.asarray(u))
        history.append(hist)
        u = window_field(params, zb, za, zb, u)
    wall = sum(h["wall_time"] for h in history)
    return TrainResult(cfg=cfg, windows=windows, params=params_list, starts=starts, history=history, wall_time=wall)
