# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS_DIR, already_done, load_result, save_json, save_result
from experiments.runner import run_training_resumable
from gipinn.integrate import solve_psi
from gipinn.lattice import Lattice, disorder
from gipinn.train import TrainConfig, predict_psi, rel_l2

VARIANTS = {
    "vanilla": ("vanilla", 0.0),
    "gipinn_a0.25": ("gipinn", 0.25),
    "gipinn_a0.5": ("gipinn", 0.5),
    "gipinn_a0.75": ("gipinn", 0.75),
    "gipinn_phi": ("gipinn_phi", 1.0),
    "vanilla_rebalanced": ("vanilla_rebalanced", 0.0),
}
KEYS = tuple(VARIANTS)
H_VALUES = (0.0, 0.1, 0.2, 0.3)
SEEDS = (0, 1, 2)
G, W, W_SEED = 2.0, 1.0, 0
Z_EVAL = np.linspace(0.0, 10.0, 101)

def result_name(key: str, h: float, seed: int) -> str:
    return f"fig1c_{key}_h{h:g}_s{seed}"

def lattice(h: float) -> Lattice:
    return Lattice(h=h, g=G, eps=disorder(41, W, W_SEED))

def run_one(key: str, h: float, seed: int, time_budget_s: float = 480.0) -> bool:
    name = result_name(key, h, seed)
    if already_done(name):
        return True
    lat = lattice(h)
    method, alpha = VARIANTS[key]
    cfg = TrainConfig(method=method, alpha=alpha, seed=seed)
    res = run_training_resumable(name, lat, cfg, time_budget_s=time_budget_s)
    if res is None:
        return False
    psi_pinn = predict_psi(res, lat, Z_EVAL)
    psi_ref = solve_psi(lat, Z_EVAL)
    err = rel_l2(psi_pinn, psi_ref)
    meta = {"key": key, "method": method, "alpha": alpha, "h": h, "seed": seed, "g": G, "W": W, "W_seed": W_SEED, "cfg": cfg,
            "err_z10": float(err[-1]), "err_max": float(err.max()), "wall_time": res.wall_time,
            "final_losses": [hh["final_loss"] for hh in res.history]}
    save_result(name, {"z_eval": Z_EVAL, "rel_l2": err, "psi_pinn_z10": psi_pinn[-1], "psi_ref_z10": psi_ref[-1]}, meta)
    print(f"{name}: err(z=10)={err[-1]:.3e} max={err.max():.3e} t={res.wall_time:.0f}s", flush=True)
    return True

def all_configs():
    main = ("vanilla", "gipinn_a0.5", "gipinn_phi")
    cfgs = [(k, h, s) for s in SEEDS for h in H_VALUES for k in main]
    cfgs += [(k, h, 0) for h in H_VALUES for k in ("gipinn_a0.25", "gipinn_a0.75")]
    cfgs += [("vanilla_rebalanced", h, s) for s in SEEDS[:2] for h in H_VALUES]

    cfgs += [(k, h, s) for s in SEEDS[1:] for h in H_VALUES for k in ("gipinn_a0.25", "gipinn_a0.75")]
    return cfgs

def summarize_fig1c(results_dir=None) -> dict:
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    err = {k: {} for k in KEYS}
    for p in sorted(d.glob("fig1c_*_h*_s*.json")):
        arrays, meta = load_result(p.stem, d)
        err[meta["key"]].setdefault(f"{meta['h']:g}", []).append(float(arrays["rel_l2"][-1]))
    out = {"err_z10": {}, "ratio_vanilla_over_gipinn": {}, "ratio_rebalanced_over_gipinn": {}, "ratio_phi_over_gipinn": {},
           "best_alpha": {}, "n_results": 0}
    for k in KEYS:
        out["err_z10"][k] = {h: (float(np.mean(v)), float(np.std(v)), len(v)) for h, v in err[k].items()}
        out["n_results"] += sum(len(v) for v in err[k].values())
    ref = "gipinn_a0.5"
    for h in err[ref]:
        g_mean = out["err_z10"][ref][h][0]
        if h in err["vanilla"]:
            out["ratio_vanilla_over_gipinn"][h] = out["err_z10"]["vanilla"][h][0] / g_mean
        if h in err["vanilla_rebalanced"]:
            out["ratio_rebalanced_over_gipinn"][h] = out["err_z10"]["vanilla_rebalanced"][h][0] / g_mean
        if h in err["gipinn_phi"]:
            out["ratio_phi_over_gipinn"][h] = out["err_z10"]["gipinn_phi"][h][0] / g_mean
        alphas = {k: out["err_z10"][k][h][0] for k in KEYS if k != "vanilla_rebalanced" and h in err[k]}
        out["best_alpha"][h] = min(alphas, key=alphas.get)
    save_json("fig1c_summary", out, d)
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--key", choices=KEYS)
    p.add_argument("--h", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--seeds", type=int, default=len(SEEDS), help="use only the first k seeds")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--time-budget", type=float, default=480.0)
    a = p.parse_args(argv)
    if a.summarize:
        s = summarize_fig1c()
        for k in KEYS:
            print(k, s["err_z10"][k])
        print("ratio vanilla/gipinn(0.5)", s["ratio_vanilla_over_gipinn"])
        print("ratio rebalanced/gipinn(0.5)", s["ratio_rebalanced_over_gipinn"])
        print("ratio gipinn_phi/gipinn(0.5)", s["ratio_phi_over_gipinn"])
        print("best variant per h", s["best_alpha"])
        return
    if a.all:
        cfgs = [c for i, c in enumerate(all_configs()) if c[2] < a.seeds and i % a.n_workers == a.worker]
        for k, h, s in cfgs:
            t = time.time()
            while not run_one(k, h, s, a.time_budget):
                pass
            print(f"[worker {a.worker}] {result_name(k, h, s)} finished in {time.time() - t:.0f}s", flush=True)
        return
    run_one(a.key, a.h, a.seed, a.time_budget)

if __name__ == "__main__":
    main()
