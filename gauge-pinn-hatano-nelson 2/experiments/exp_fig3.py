# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS_DIR, already_done, load_result, save_json, save_result
from gipinn.inverse import (InverseConfig, baseline_rk4_fit, eps_metrics, param_error, synthetic_intensity,
                            train_inverse)
from gipinn.lattice import Lattice, disorder

H_TRUE, G_TRUE, W, W_SEED, N, N0 = 0.2, 2.0, 1.0, 0, 41, 21
Z_MAX, N_WINDOWS = 2.5, 5
MODES = ("abs", "norm")
K_VALUES = (5, 10, 25)
SIGMAS = (0.0, 0.02, 0.05, 0.10)
SEEDS = (0, 1)
LEARN_EPS = True
LIT_MASK = np.abs(np.arange(1, N + 1) - N0) <= 4
TARGET_PARAM, TARGET_CORR = 0.05, 0.9
TARGET_SIGMA, TARGET_K, TARGET_SIGMA_EPS = 0.05, 10, 0.02

def truth() -> Lattice:
    return Lattice(N=N, n0=N0, h=H_TRUE, g=G_TRUE, eps=disorder(N, W, W_SEED))

def planes(K: int) -> np.ndarray:
    return np.linspace(Z_MAX / K, Z_MAX, K)

def pinn_name(mode, K, sigma, seed, learn_eps: bool = True) -> str:
    return f"fig3_pinn_{mode}_K{K}_sig{sigma:g}_s{seed}" + ("" if learn_eps else "_fixeps")

def baseline_name(mode, K, sigma, seed: int = 0) -> str:
    return f"fig3_baseline_{mode}_K{K}_sig{sigma:g}" + ("" if seed == 0 else f"_s{seed}")

def pinn_config(mode: str, seed: int, learn_eps: bool = True) -> InverseConfig:
    return InverseConfig(z_max=Z_MAX, n_windows=N_WINDOWS, n_colloc=150, data_mode=mode, learn_eps=learn_eps, seed=seed,
                         h_init=0.1, g_init=1.0, warmup_frac=0.0, eps_warmup_frac=0.3, ridge_eps=1e-3,
                         adam_steps=1500, lbfgs_steps=2500)

def run_pinn(mode: str, K: int, sigma: float, seed: int, learn_eps: bool = True) -> None:
    name = pinn_name(mode, K, sigma, seed, learn_eps)
    if already_done(name):
        return
    lat = truth()
    z = planes(K)
    I = synthetic_intensity(lat, z, sigma, seed=100 + seed)
    cfg = pinn_config(mode, seed, learn_eps)
    res = train_inverse(N, N0, z, I, cfg, eps_known=None if learn_eps else np.asarray(lat.eps))
    m = eps_metrics(res.eps, lat.eps)
    ml = eps_metrics(res.eps, lat.eps, LIT_MASK)
    meta = {"mode": mode, "K": K, "sigma": sigma, "seed": seed, "h_est": res.h, "g_est": res.g, "h_true": H_TRUE,
            "g_true": G_TRUE, "learn_eps": learn_eps, "err_h": param_error(res.h, H_TRUE), "err_g": param_error(res.g, G_TRUE),
            "eps_corr": m["corr"], "eps_rmse": m["rmse"], "eps_corr_lit": ml["corr"], "eps_rmse_lit": ml["rmse"],
            "final_loss": res.history["final_loss"],
            "final_res": res.history["final_res"], "final_data": res.history["final_data"], "wall_time": res.wall_time, "cfg": cfg}
    save_result(name, {"eps_est": res.eps, "eps_true": np.asarray(lat.eps), "hist_h": np.array(res.history["h_traj"]),
                       "hist_g": np.array(res.history["g_traj"]), "z_planes": z, "I_obs": I}, meta)
    print(f"{name}: h={res.h:.4f} g={res.g:.4f} err_h={meta['err_h']:.3f} err_g={meta['err_g']:.3f} eps corr={m['corr']:.3f} (lit {ml['corr']:.3f}) t={res.wall_time:.0f}s", flush=True)

def run_baseline(mode: str, K: int, sigma: float, seed: int = 0) -> None:
    name = baseline_name(mode, K, sigma, seed)
    if already_done(name):
        return
    lat = truth()
    z = planes(K)
    I = synthetic_intensity(lat, z, sigma, seed=100 + seed)
    res = baseline_rk4_fit(N, N0, z, I, mode, LEARN_EPS, {"h": H_TRUE, "g": G_TRUE, "eps": np.asarray(lat.eps), "eps_mask": LIT_MASK},
                           n_restarts=10, adam_steps=1000, lbfgs_steps=1000, n_steps=500, lr=5e-3, seed=seed)
    m = eps_metrics(res.eps, lat.eps)
    ml = eps_metrics(res.eps, lat.eps, LIT_MASK)
    meta = {"mode": mode, "K": K, "sigma": sigma, "seed": seed, "h": res.h, "g": res.g, "err_h": param_error(res.h, H_TRUE),
            "err_g": param_error(res.g, G_TRUE), "eps_corr": m["corr"], "eps_corr_lit": ml["corr"], "success_rate": res.success_rate,
            "wall_time": res.wall_time, "per_restart": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
                                                           for r in res.per_restart]}
    save_result(name, {"eps": res.eps, "restart_h": np.array([r["h"] for r in res.per_restart]),
                       "restart_g": np.array([r["g"] for r in res.per_restart]), "restart_loss": np.array([r["loss"] for r in res.per_restart])}, meta)
    print(f"{name}: h={res.h:.4f} g={res.g:.4f} success={res.success_rate:.2f} t={res.wall_time:.0f}s", flush=True)

def all_configs():
    pinn = [("pinn", m, K, s, seed) for seed in SEEDS for m in MODES for K in K_VALUES for s in SIGMAS]
    base = [("baseline", m, K, s, None) for m in MODES for K in K_VALUES for s in SIGMAS]

    fixeps = [("pinn_fixeps", "abs", K, s, seed) for seed in SEEDS for K in K_VALUES for s in SIGMAS]

    base2 = [("baseline", "abs", K, s, seed) for seed in SEEDS[1:] for K in K_VALUES for s in SIGMAS]
    return pinn + base + fixeps + base2

def summarize_fig3(results_dir=None) -> dict:
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    acc = {}
    for p in sorted(d.glob("fig3_pinn_*.json")):
        arrays, meta = load_result(p.stem, d)
        mode = meta["mode"] + ("" if meta.get("learn_eps", True) else "_fixeps")
        key = (mode, str(meta["K"]), f"{meta['sigma']:g}")
        m = eps_metrics(arrays["eps_est"], arrays["eps_true"])
        ml = eps_metrics(arrays["eps_est"], arrays["eps_true"], LIT_MASK)
        acc.setdefault(key, []).append((param_error(meta["h_est"], meta["h_true"]), param_error(meta["g_est"], meta["g_true"]),
                                        m["corr"], m["rmse"], ml["corr"], ml["rmse"], meta.get("wall_time", float("nan"))))
    out = {"err_h": {}, "err_g": {}, "eps_corr": {}, "eps_rmse": {}, "eps_corr_lit": {}, "eps_rmse_lit": {}, "wall_time": {}, "baseline": {}}
    for (mode, K, sig), rows in acc.items():
        arr = np.array(rows)
        for i, nm in enumerate(["err_h", "err_g", "eps_corr", "eps_rmse", "eps_corr_lit", "eps_rmse_lit", "wall_time"]):
            out[nm].setdefault(mode, {}).setdefault(K, {})[sig] = (float(np.mean(arr[:, i])), float(np.std(arr[:, i])), len(rows))
    bacc = {}
    for p in sorted(d.glob("fig3_baseline_*.json")):
        arrays, meta = load_result(p.stem, d)
        pr = meta.get("per_restart", [])
        hg_ok = [param_error(r["h"], H_TRUE) <= 0.05 and param_error(r["g"], G_TRUE) <= 0.05 for r in pr]
        gs = np.array([r["g"] for r in pr]) if pr else np.array([meta["g"]])
        bacc.setdefault((meta["mode"], str(meta["K"]), f"{meta['sigma']:g}"), []).append(
            (param_error(meta["h"], H_TRUE), param_error(meta["g"], G_TRUE), meta.get("eps_corr_lit", float("nan")),
             float(np.mean(hg_ok)) if pr else float("nan"), float((gs.max() - gs.min()) / G_TRUE), meta["wall_time"], len(pr)))
    for (mode, K, sig), rows in bacc.items():
        arr = np.array(rows)
        rec = {nm: (float(np.mean(arr[:, i])), float(np.std(arr[:, i])), len(rows))
               for i, nm in enumerate(["err_h", "err_g", "eps_corr_lit", "success_rate_hg", "restart_g_spread", "wall_time"])}
        rec["n_restarts"] = int(arr[0, 6])
        out["baseline"].setdefault(mode, {}).setdefault(K, {})[sig] = rec
    abs_recs = [r for K in out["baseline"].get("abs", {}).values() for r in K.values()]
    out["baseline_max_g_spread_abs"] = max([v["restart_g_spread"][0] + v["restart_g_spread"][1] for v in abs_recs], default=None)
    out["baseline_n_configs_abs"] = len(abs_recs)
    noisy = [(K, sig) for K in out["baseline"].get("abs", {}) for sig in out["baseline"]["abs"][K] if float(sig) > 0
             and K in out["eps_corr_lit"].get("abs", {}) and sig in out["eps_corr_lit"]["abs"][K]]
    out["n_noisy_configs_abs"] = len(noisy)
    out["pinn_eps_lit_better_abs"] = int(sum(out["eps_corr_lit"]["abs"][K][sig][0] > out["baseline"]["abs"][K][sig]["eps_corr_lit"][0] for K, sig in noisy))
    out["baseline_n_spread_below_3pct_abs"] = int(sum(v["restart_g_spread"][0] + v["restart_g_spread"][1] < 0.03 for v in abs_recs))
    try:
        t_param = out["err_h"]["abs"][str(TARGET_K)][f"{TARGET_SIGMA:g}"][0] <= TARGET_PARAM and \
            out["err_g"]["abs"][str(TARGET_K)][f"{TARGET_SIGMA:g}"][0] <= TARGET_PARAM
        t_eps = out["eps_corr_lit"]["abs"][str(TARGET_K)][f"{TARGET_SIGMA_EPS:g}"][0] >= TARGET_CORR
        out["pass_params"], out["pass_eps"], out["pass_targets"] = bool(t_param), bool(t_eps), bool(t_param and t_eps)
    except KeyError:
        out["pass_params"] = out["pass_eps"] = out["pass_targets"] = None
    save_json("fig3_summary", out, d)
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, default="abs")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--time-budget", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.summarize:
        s = summarize_fig3()
        for nm in ("err_h", "err_g", "eps_corr", "eps_corr_lit"):
            print(nm, s[nm])
        print("baseline", s["baseline"])
        print("pass", s["pass_params"], s["pass_eps"], s["pass_targets"])
        return
    if a.all:
        for i, (kind, mode, K, sig, seed) in enumerate(all_configs()):
            if i % a.n_workers != a.worker:
                continue
            t = time.time()
            if kind == "pinn":
                run_pinn(mode, K, sig, seed)
            elif kind == "pinn_fixeps":
                run_pinn(mode, K, sig, seed, learn_eps=False)
            else:
                run_baseline(mode, K, sig, seed or 0)
            print(f"[worker {a.worker}] {kind} {mode} K={K} sigma={sig} seed={seed} done in {time.time() - t:.0f}s", flush=True)
        return
    if a.baseline:
        run_baseline(a.mode, a.K, a.sigma, a.seed)
    else:
        run_pinn(a.mode, a.K, a.sigma, a.seed)

if __name__ == "__main__":
    main()
