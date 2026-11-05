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
from gipinn.lattice import H_phi, Lattice, P_phi, disorder, initial_psi, ipr, psi_to_phi
from gipinn.train import TrainConfig, predict_psi, rel_l2

H, G, W, W_SEED = 0.2, 2.0, 1.0, 0
SEEDS = (0, 1, 2, 3, 4)
ALPHA = TrainConfig().alpha
Z_GRID = np.linspace(0.0, 10.0, 401)
TARGET_ERR, TARGET_DEV = 2e-2, 1e-3

def lattice() -> Lattice:
    return Lattice(h=H, g=G, eps=disorder(41, W, W_SEED))

def _observables(lat, psi):
    phi = psi_to_phi(lat, psi)
    return (np.array([ipr(p) for p in psi]), np.array([P_phi(p) for p in phi]), np.array([H_phi(lat, p) for p in phi]))

def run_reference():
    if already_done("fig2_reference"):
        return
    lat = lattice()
    psi = solve_psi(lat, Z_GRID)
    I, P, Hh = _observables(lat, psi)
    save_result("fig2_reference", {"z_grid": Z_GRID, "psi_ref": psi, "ipr_ref": I, "P_phi_ref": P, "H_phi_ref": Hh},
                {"h": H, "g": G, "W": W, "W_seed": W_SEED, "eps": lat.eps})

def run_seed(seed: int, time_budget_s: float = 480.0) -> bool:
    name = f"fig2_gipinn_a{ALPHA:g}_s{seed}"
    if already_done(name):
        return True
    run_reference()
    lat = lattice()
    cfg = TrainConfig(seed=seed)
    res = run_training_resumable(name, lat, cfg, time_budget_s=time_budget_s)
    if res is None:
        return False
    ref, _ = load_result("fig2_reference")
    psi = predict_psi(res, lat, Z_GRID)
    err = rel_l2(psi, ref["psi_ref"])
    I, P, Hh = _observables(lat, psi)
    phi0 = psi_to_phi(lat, initial_psi(lat))
    meta = {"seed": seed, "cfg": cfg, "err_z10": float(err[-1]), "err_max": float(err.max()),
            "max_dev_P": float(np.max(np.abs(P - P_phi(phi0)))), "max_dev_H": float(np.max(np.abs(Hh - H_phi(lat, phi0)))),
            "max_ipr_err": float(np.max(np.abs(I - ref["ipr_ref"]))), "wall_time": res.wall_time,
            "final_losses": [hh["final_loss"] for hh in res.history]}
    save_result(name, {"psi_pinn": psi, "rel_l2": err, "err_local": np.abs(psi - ref["psi_ref"]) ** 2, "ipr_pinn": I,
                       "P_phi_pinn": P, "H_phi_pinn": Hh, "windows": np.array(res.windows)}, meta)
    print(f"{name}: err(z=10)={err[-1]:.3e} max={err.max():.3e} devP={meta['max_dev_P']:.1e} devH={meta['max_dev_H']:.1e}", flush=True)
    return True

def summarize_fig2(results_dir=None) -> dict:
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    metas = [load_result(p.stem, d)[1] for p in sorted(d.glob(f"fig2_gipinn_a{ALPHA:g}_s*.json"))]
    e = np.array([m["err_z10"] for m in metas])
    out = {"alpha": ALPHA, "n_seeds": len(metas), "err_z10": (float(e.mean()), float(e.std())), "err_z10_per_seed": e.tolist(),
           "max_dev_P": max(m["max_dev_P"] for m in metas), "max_dev_H": max(m["max_dev_H"] for m in metas),
           "max_ipr_err": max(m["max_ipr_err"] for m in metas), "wall_time_mean": float(np.mean([m["wall_time"] for m in metas]))}
    try:
        H0 = abs(float(load_result("fig2_reference", d)[0]["H_phi_ref"][0]))
        out["H_phi_0"], out["max_dev_H_rel"] = H0, out["max_dev_H"] / H0
    except FileNotFoundError:
        pass
    out["pass_err"] = bool(e.mean() <= TARGET_ERR)
    out["pass_dev_P"] = bool(out["max_dev_P"] <= TARGET_DEV)
    out["pass_dev_H"] = bool(out["max_dev_H"] <= TARGET_DEV)
    save_json("fig2_summary", out, d)
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--time-budget", type=float, default=480.0)
    a = p.parse_args(argv)
    if a.reference:
        run_reference()
    if a.seed is not None:
        run_seed(a.seed, a.time_budget)
    if a.all:
        for s in [s for i, s in enumerate(SEEDS) if i % a.n_workers == a.worker]:
            t = time.time()
            while not run_seed(s, a.time_budget):
                pass
            print(f"[worker {a.worker}] seed {s} finished in {time.time() - t:.0f}s", flush=True)
    if a.summarize:
        print(summarize_fig2())

if __name__ == "__main__":
    main()
