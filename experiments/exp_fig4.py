# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS_DIR, already_done, load_json, load_result, save_json, save_result
from experiments.runner import run_training_resumable
from gipinn.integrate import solve_psi
from gipinn.lattice import Lattice, com_shift, disorder, ipr, p_trap
from gipinn.phase_map import sweep_observables
from gipinn.train import TrainConfig, predict_psi, rel_l2

W_GRID = np.linspace(0.0, 8.0, 21)
G_GRID = np.linspace(0.0, 6.0, 21)
N_REAL = 50
H_MAP = (0.0, 0.2)
H_REGIMES = 0.2
Z_EVAL = np.linspace(0.0, 10.0, 101)
REGIMES = ("skin", "anderson", "trapped")

def run_map(h: float) -> None:
    name = f"fig4_map_h{h:g}"
    if already_done(name):
        return
    t = time.time()
    ckpt = RESULTS_DIR / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    out = sweep_observables(41, 21, h, W_GRID, G_GRID, N_REAL, z_max=10.0, n_steps=4000, seed=1000,
                            row_cache=lambda i: ckpt / f"{name}_row{i}.npz", log=lambda m: print(f"[{name}] {m}", flush=True))
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    save_result(name, arrays, {"h": h, "n_real": N_REAL, "z_max": 10.0, "n_steps": 4000, "seed": 1000, "wall_time": time.time() - t})
    print(f"{name}: done in {time.time() - t:.0f} s", flush=True)

def pick_regimes() -> dict:
    a, _ = load_result(f"fig4_map_h{H_REGIMES:g}")
    W, g = a["W_grid"], a["g_grid"]
    ipr_m, pt, com = a["ipr"], a["p_trap"], a["com"]

    def rec(i, j):
        return {"W": float(W[i]), "g": float(g[j]), "ipr": float(ipr_m[i, j]), "p_trap": float(pt[i, j]), "com": float(com[i, j])}

    pick = {}
    pick["skin"] = rec(*np.unravel_index(np.argmax(com), com.shape))
    mask = np.abs(com[:, 0]) < 3.0
    pick["anderson"] = rec(int(np.argmax(np.where(mask, ipr_m[:, 0], -np.inf))) if mask.any() else len(W) - 1, 0)
    pick["trapped"] = rec(0, int(np.argmax(pt[0])))
    pick["h"] = H_REGIMES
    save_json("fig4_regimes", pick)
    return pick

def validate(regime: str, seed: int = 0, time_budget_s: float = 480.0) -> bool:
    name = f"fig4_profiles_{regime}"
    if already_done(name):
        return True
    reg = load_json("fig4_regimes")[regime]
    lat = Lattice(h=H_REGIMES, g=reg["g"], eps=disorder(41, reg["W"], 1000))
    res = run_training_resumable(name, lat, TrainConfig(seed=seed), time_budget_s=time_budget_s)
    if res is None:
        return False
    psi_pinn = predict_psi(res, lat, Z_EVAL)
    psi_ref = solve_psi(lat, Z_EVAL)
    err = rel_l2(psi_pinn, psi_ref)
    meta = {"regime": regime, "h": H_REGIMES, "W": reg["W"], "g": reg["g"], "seed": seed, "err_z10": float(err[-1]), "err_max": float(err.max()),
            "ipr_ref": float(ipr(psi_ref[-1])), "ipr_pinn": float(ipr(psi_pinn[-1])), "p_trap_ref": float(p_trap(lat, psi_ref[-1])),
            "p_trap_pinn": float(p_trap(lat, psi_pinn[-1])), "com_ref": float(com_shift(lat, psi_ref[-1])), "com_pinn": float(com_shift(lat, psi_pinn[-1])),
            "wall_time": res.wall_time}
    save_result(name, {"z_eval": Z_EVAL, "rel_l2": err, "psi_ref_z10": psi_ref[-1], "psi_pinn_z10": psi_pinn[-1],
                       "I_ref": np.abs(psi_ref) ** 2, "I_pinn": np.abs(psi_pinn) ** 2}, meta)
    print(f"{name}: W={reg['W']} g={reg['g']} err(z=10)={err[-1]:.3e}", flush=True)
    return True

def _point_observables(h: float, W: float, g: float) -> dict:
    name = f"fig4_point_h{h:g}_W{W:g}_g{g:g}"
    try:
        return load_json(name)
    except FileNotFoundError:
        pass
    o = sweep_observables(41, 21, h, [W], [g], N_REAL, z_max=10.0, n_steps=4000, seed=1000)
    rec = {"h": h, "W": W, "g": g, "ipr": float(o["ipr"][0, 0]), "p_trap": float(o["p_trap"][0, 0]), "com": float(o["com"][0, 0])}
    save_json(name, rec)
    return rec

def summarize_fig4() -> dict:
    out = {}
    for h in H_MAP:
        a, _ = load_result(f"fig4_map_h{h:g}")
        W, g = a["W_grid"], a["g_grid"]
        pt, ipr_m, com = a["p_trap"], a["ipr"], a["com"]
        iW1, iW8 = int(np.argmin(np.abs(W - 1.0))), len(W) - 1
        above = np.where(pt[0] > 0.5)[0]
        out[f"h{h:g}"] = {
            "g_threshold_W0": float(g[above[0]]) if above.size else None,
            "g_threshold_W0_prev": float(g[above[0] - 1]) if above.size and above[0] > 0 else None,
            "W_half_g0": float(W[np.where(pt[:, 0] > 0.5)[0][0]]) if (pt[:, 0] > 0.5).any() else None,
            "W_com_below5_g0": float(W[np.where(com[:, 0] < 5.0)[0][0]]) if (com[:, 0] < 5.0).any() else None,
            "com_W0_gsmall": float(com[0, 1]), "g_small": float(g[1]), "com_W0_gmax": float(com[0, -1]),
            "p_trap_W0_gmax": float(pt[0, -1]), "p_trap_W0_max": float(pt[0].max()), "p_trap_max": float(pt.max()),
            "com_W0_g0": float(com[0, 0]), "com_W8_g0": float(com[iW8, 0]),
            "com_W1_g2_grid": float(com[iW1, np.argmin(np.abs(g - 2.0))]),
            "com_W1_g2": _point_observables(h, 1.0, 2.0)["com"],
            "ipr_W0_g0": float(ipr_m[0, 0]), "ipr_W8_g0": float(ipr_m[iW8, 0]),
            "argmax_ipr_W0": {"g": float(g[np.argmax(ipr_m[0])]), "ipr": float(ipr_m[0].max())},
        }
    try:
        out["regimes"] = load_json("fig4_regimes")
    except FileNotFoundError:
        pass
    val = {}
    for r in REGIMES:
        try:
            _, m = load_result(f"fig4_profiles_{r}")
            val[r] = {k: m[k] for k in ("W", "g", "err_z10", "ipr_ref", "ipr_pinn", "p_trap_ref", "p_trap_pinn", "com_ref", "com_pinn")}
        except FileNotFoundError:
            pass
    out["validation"] = val
    out["max_val_err"] = max([v["err_z10"] for v in val.values()], default=None)
    save_json("fig4_summary", out)
    return out

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--map", action="store_true")
    p.add_argument("--h", type=float, default=0.2)
    p.add_argument("--pick-regimes", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--regime", choices=REGIMES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=480.0)
    a = p.parse_args(argv)
    if a.map:
        run_map(a.h)
    if a.pick_regimes:
        print(pick_regimes())
    if a.validate:
        for r in ([a.regime] if a.regime else REGIMES):
            while not validate(r, time_budget_s=a.time_budget):
                pass
    if a.all:
        for i, r in enumerate(REGIMES):
            if i % a.n_workers == a.worker:
                while not validate(r, time_budget_s=a.time_budget):
                    pass
    if a.summarize:
        print(summarize_fig4())

if __name__ == "__main__":
    main()
