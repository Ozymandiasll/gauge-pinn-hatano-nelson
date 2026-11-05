# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from experiments.common import RESULTS_DIR, _jsonable, load_json, load_result
from experiments.exp_fig1c import KEYS, H_VALUES, summarize_fig1c
from experiments.exp_fig3 import K_VALUES, LIT_MASK, MODES, SIGMAS, baseline_name, pinn_name, summarize_fig3
from gipinn.plotting import COL_W, DOUBLE_W, DIV_CMAP, INK2, MARKERS, SEQ_CMAP, SERIES, despine, light_grid, panel_label, set_style

FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"
LABELS = {"vanilla": r"$\alpha=0$", "gipinn_a0.25": r"$\alpha=1/4$", "gipinn_a0.5": r"$\alpha=1/2$",
          "gipinn_a0.75": r"$\alpha=3/4$", "gipinn_phi": r"$\alpha=1$", "vanilla_rebalanced": r"$\alpha=0$, rebal."}
FIGDATA_DIR = Path(__file__).resolve().parent.parent / "figdata"

class ResultsSource:
    def __init__(self, results_dir=None):
        self.d = Path(results_dir) if results_dir is not None else RESULTS_DIR

    def summarize_fig1c(self):
        return summarize_fig1c(self.d)

    def summarize_fig3(self):
        return summarize_fig3(self.d)

    def load_result(self, name):
        return load_result(name, self.d)

    def load_json(self, name):
        return load_json(name, self.d)

    def glob(self, pattern):
        return [p.stem for p in self.d.glob(pattern)]

class Recorder(ResultsSource):
    def __init__(self, results_dir=None):
        super().__init__(results_dir)
        self.calls, self.arrays = {}, {}

    def _keep(self, key, value):
        self.calls[key] = value
        return value

    def summarize_fig1c(self):
        return self._keep("summarize_fig1c", super().summarize_fig1c())

    def summarize_fig3(self):
        return self._keep("summarize_fig3", super().summarize_fig3())

    def load_json(self, name):
        return self._keep(f"load_json:{name}", super().load_json(name))

    def glob(self, pattern):
        return self._keep(f"glob:{pattern}", super().glob(pattern))

    def load_result(self, name):
        arrays, meta = super().load_result(name)
        self.calls[f"load_result:{name}"] = {"meta": meta, "fields": sorted(arrays)}
        for k, v in arrays.items():
            self.arrays[f"{name}::{k}"] = v
        return arrays, meta

    def save(self, out_dir=FIGDATA_DIR):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "figdata.npz", **self.arrays)
        with open(out_dir / "figdata.json", "w") as f:
            json.dump(_jsonable(self.calls), f)
        return out_dir

class FigData:
    def __init__(self, d=FIGDATA_DIR):
        d = Path(d)
        with open(d / "figdata.json") as f:
            self.calls = json.load(f)
        with np.load(d / "figdata.npz") as z:
            self.arrays = {k: z[k] for k in z.files}

    def _get(self, key):
        if key not in self.calls:
            raise FileNotFoundError(key)
        return self.calls[key]

    def summarize_fig1c(self):
        return self._get("summarize_fig1c")

    def summarize_fig3(self):
        return self._get("summarize_fig3")

    def load_json(self, name):
        return self._get(f"load_json:{name}")

    def glob(self, pattern):
        return self.calls.get(f"glob:{pattern}", [])

    def load_result(self, name):
        rec = self._get(f"load_result:{name}")
        return {k: self.arrays[f"{name}::{k}"] for k in rec["fields"]}, rec["meta"]

def _source(results_dir):
    if isinstance(results_dir, (ResultsSource, FigData)):
        return results_dir
    if results_dir is None and (FIGDATA_DIR / "figdata.json").exists():
        return FigData()
    return ResultsSource(results_dir)

def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return out

def _schematic_lattice(ax):
    ax.set_xlim(-0.8, 8.8); ax.set_ylim(-1.35, 1.75); ax.set_aspect("equal"); ax.axis("off")
    for n in range(9):
        ax.add_patch(Circle((n, 0), 0.3, facecolor="#f4f7fb", edgecolor=INK2, linewidth=0.8))
        if n < 8:
            ax.add_patch(FancyArrowPatch((n + 0.34, 0.16), (n + 0.66, 0.16), arrowstyle="-|>", mutation_scale=6, color=SERIES[0], linewidth=1.4))
            ax.add_patch(FancyArrowPatch((n + 0.66, -0.16), (n + 0.34, -0.16), arrowstyle="-|>", mutation_scale=6, color=SERIES[1], linewidth=0.7))
    ax.text(4, 0.0, r"$n_0$", ha="center", va="center", fontsize=6.5)
    ax.text(1.5, 0.62, r"$e^{+h}$", color=SERIES[0], ha="center", va="bottom", fontsize=7)
    ax.text(1.5, -0.62, r"$e^{-h}$", color=SERIES[1], ha="center", va="top", fontsize=7)
    ax.text(6.0, -0.62, r"$n$", ha="center", va="top", fontsize=6.5, color=INK2)
    ax.annotate(r"$\varepsilon_n,\; g|\psi_n|^2$", xy=(6.0, 0.3), xytext=(6.6, 0.95), fontsize=7, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(8.3, 1.3), xytext=(-0.3, 1.3), arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.8))
    ax.text(4, 1.42, "skin accumulation", ha="center", va="bottom", fontsize=6.5, color=INK2)

def _schematic_network(ax):
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.0); ax.axis("off")
    boxes = [(0.2, 1.0, r"$z$", "#ffffff"), (1.8, 1.5, "MLP", "#f4f7fb"), (3.9, 1.5, r"$u_n(z)$", "#ffffff"),
             (6.0, 2.5, r"$e^{\alpha h(n-n_0)}u_n/\sqrt{g}$", "#fff4ec"), (9.0, 1.0, r"$\psi_n(z)$", "#ffffff")]
    for x, w, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 1.1, boxstyle="round,pad=0.06", facecolor=c, edgecolor=INK2, linewidth=0.7))
        ax.text(x + w / 2, 2.05, t, ha="center", va="center", fontsize=6.5)
    for x1, x2 in ((1.2, 1.75), (3.3, 3.85), (5.4, 5.95), (8.5, 8.95)):
        ax.annotate("", xy=(x2, 2.05), xytext=(x1, 2.05), arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=0.7))
    ax.text(0.2, 3.4, r"hard IC $u(z_a)=u_a$;  trainable $h,\,g,\,\varepsilon_n$", fontsize=6.5, color=SERIES[2], va="center")
    ax.text(0.2, 0.95, r"loss: causal residual of Eq. (3)", fontsize=6.5, color=SERIES[0], va="center")
    ax.text(0.2, 0.3, r"$+\;|\psi_n(z_k)|^2$ vs measured intensity (inverse problem)", fontsize=6.5, color=SERIES[1], va="center")

def fig1(results_dir=None, out=None):
    set_style()
    src = _source(results_dir)
    s = src.summarize_fig1c()
    fig = plt.figure(figsize=(COL_W, 3.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 2.2], hspace=0.25, left=0.17, right=0.98, top=0.98, bottom=0.11)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])
    _schematic_lattice(ax_a); panel_label(ax_a, "(a)", x=-0.2, y=0.82)
    _schematic_network(ax_b); panel_label(ax_b, "(b)", x=-0.2, y=0.82)
    hs = [float(h) for h in H_VALUES]
    for i, key in enumerate(KEYS):
        vals = s["err_z10"].get(key, {})
        if not vals:
            continue
        m = np.array([vals.get(f"{h:g}", (np.nan, np.nan, 0))[0] for h in hs])
        sd = np.array([vals.get(f"{h:g}", (np.nan, np.nan, 0))[1] for h in hs])
        ls = "--" if key == "vanilla_rebalanced" else "-"
        ax_c.errorbar(hs, m, yerr=sd, color=SERIES[i], marker=MARKERS[i], linestyle=ls, capsize=1.5, label=LABELS[key], linewidth=1.0)
    ax_c.set_yscale("log"); ax_c.set_xlabel(r"non-reciprocity $h$"); ax_c.set_ylabel(r"rel. $L_2$ error at $z=10$")
    ax_c.set_xticks(hs); light_grid(ax_c); despine(ax_c)
    lo, hi = ax_c.get_ylim()
    ax_c.set_ylim(lo, hi * 60)
    ax_c.legend(ncol=3, loc="upper left", handlelength=1.4, columnspacing=0.8, fontsize=6)
    panel_label(ax_c, "(c)", x=-0.2, y=0.97)
    return _save(fig, Path(out) if out else FIGS_DIR / "fig1.pdf")

def fig2(results_dir=None, out=None):
    set_style()
    src = _source(results_dir)
    ref, rmeta = src.load_result("fig2_reference")
    from gipinn.train import TrainConfig
    seeds = sorted(src.glob(f"fig2_gipinn_a{TrainConfig().alpha:g}_s*.json")) or sorted(src.glob("fig2_gipinn_s*.json"))
    runs = [src.load_result(p) for p in seeds]
    a0, m0 = runs[0]
    z, N = ref["z_grid"], ref["psi_ref"].shape[1]
    I_ref, I_pinn = np.abs(ref["psi_ref"]) ** 2, np.abs(a0["psi_pinn"]) ** 2
    from matplotlib.colors import LogNorm
    vmax = max(I_ref.max(), 1e-12)
    norm = LogNorm(vmin=vmax * 1e-6, vmax=vmax)
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 2.6))
    ext = [0.5, N + 0.5, z[0], z[-1]]
    im = axs[0, 0].imshow(np.maximum(I_ref, vmax * 1e-6), origin="lower", aspect="auto", extent=ext, cmap=SEQ_CMAP, norm=norm)
    axs[0, 1].imshow(np.maximum(I_pinn, vmax * 1e-6), origin="lower", aspect="auto", extent=ext, cmap=SEQ_CMAP, norm=norm)
    for ax in axs[0]:
        ax.set_xticks([1, 11, 21, 31, 41])
    axs[0, 0].set_title("reference (DOP853)", fontsize=7); axs[0, 1].set_title("GI-PINN", fontsize=7)
    for ax in axs[0]:
        ax.set_xlabel(r"site $n$")
    axs[0, 0].set_ylabel(r"$z$"); axs[0, 1].set_yticklabels([])
    errs = np.array([r[0]["rel_l2"] for r in runs])
    axs[1, 0].plot(z, errs.mean(axis=0), color=SERIES[0])
    if len(runs) > 1:
        axs[1, 0].fill_between(z, errs.min(axis=0), errs.max(axis=0), color=SERIES[0], alpha=0.25, linewidth=0)
    for za, zb in a0["windows"]:
        axs[1, 0].axvline(zb, color="#d9d8d4", linewidth=0.4, zorder=0)
    axs[1, 0].set_yscale("log"); axs[1, 0].set_xlabel(r"$z$"); axs[1, 0].set_ylabel(r"rel. $L_2$ error"); despine(axs[1, 0])
    axs[1, 1].plot(z, ref["ipr_ref"], color=INK2, linewidth=1.6, label="IPR (ref.)")
    axs[1, 1].plot(z, a0["ipr_pinn"], color=SERIES[0], linestyle="--", label="IPR (PINN)")
    axs[1, 1].plot(z, a0["P_phi_pinn"], color=SERIES[2], linewidth=2.0, label=r"$P_\phi$ (PINN)")
    axs[1, 1].plot(z, -a0["H_phi_pinn"] / max(abs(ref["H_phi_ref"][0]), 1e-12), color=SERIES[1], linestyle=":", linewidth=1.4,
                   label=r"$-H_\phi/|H_\phi(0)|$")
    axs[1, 1].set_ylim(0, 2.3); axs[1, 1].set_yticks([0, 0.5, 1.0]); axs[1, 1].set_xlabel(r"$z$")
    axs[1, 1].legend(loc="upper right", ncol=1, handlelength=1.3, fontsize=5.8, labelspacing=0.2); despine(axs[1, 1])
    for ax, lab in zip(axs.flat, ("(a)", "(b)", "(c)", "(d)")):
        panel_label(ax, lab, x=-0.25 if lab in ("(a)", "(c)") else -0.08)
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.15, right=0.88, top=0.92, bottom=0.14)
    pos = axs[0, 1].get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"$|\psi_n|^2$", fontsize=7)
    return _save(fig, Path(out) if out else FIGS_DIR / "fig2.pdf")

def fig3(results_dir=None, out=None):
    set_style()
    src = _source(results_dir)
    s = src.summarize_fig3()
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 2.7))
    sig = [float(x) for x in SIGMAS]

    ax = axs[0, 0]
    from matplotlib.lines import Line2D
    for j, K in enumerate(K_VALUES):
        for nm, ls, mk in (("err_g", "-", "s"), ("err_h", "--", "o")):
            vals = s[nm].get("abs", {}).get(str(K), {})
            if not vals:
                continue
            m = np.array([vals.get(f"{x:g}", (np.nan,))[0] for x in sig])
            ax.plot(sig, m, color=SERIES[j], marker=mk, linestyle=ls, markersize=3, linewidth=0.9)
    fix = s["err_g"].get("abs_fixeps", {}).get("10", {})
    if fix:
        m = np.array([fix.get(f"{x:g}", (np.nan,))[0] for x in sig])
        ax.plot(sig, m, color=SERIES[1], marker="x", linestyle=":", markersize=3.5, linewidth=0.9)
    ax.axhline(0.05, color="#d9d8d4", linewidth=0.6); ax.set_yscale("log"); ax.set_xlabel(r"noise $\sigma$")
    ax.set_ylabel("rel. error of $h$, $g$")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, 0.3) * 40)
    handles = [Line2D([], [], color=SERIES[j], linewidth=1.2, label=f"K={K}") for j, K in enumerate(K_VALUES)]
    handles += [Line2D([], [], color=INK2, linestyle="-", marker="s", markersize=3, label="$g$"),
                Line2D([], [], color=INK2, linestyle="--", marker="o", markersize=3, label="$h$"),
                Line2D([], [], color=INK2, linestyle=":", marker="x", markersize=3.5, label=r"$g$, $\varepsilon$ known")]
    ax.legend(handles=handles, ncol=2, handlelength=1.3, fontsize=5.5, loc="upper left", columnspacing=0.6); despine(ax)

    ax = axs[0, 1]
    try:
        a, m = src.load_result(pinn_name("abs", 10, 0.05, 0))
        hh, gg = a["hist_h"] / m["h_true"], a["hist_g"] / m["g_true"]
        steps = 10 * np.arange(len(hh) - 1)
        x_end = steps[-1] * 1.18
        ax.plot(steps, hh[:-1], color=SERIES[0], label=r"$h/h_{\rm true}$")
        ax.plot(steps, gg[:-1], color=SERIES[1], label=r"$g/g_{\rm true}$")
        ax.plot([x_end], [hh[-1]], color=SERIES[0], marker="o", linestyle="none", markersize=3.5, clip_on=False)
        ax.plot([x_end], [gg[-1]], color=SERIES[1], marker="s", linestyle="none", markersize=3.5, clip_on=False)
        ax.axvline(steps[-1] * 1.06, color="#d9d8d4", linewidth=0.6)
        ax.text(x_end, 1.17, "L-BFGS", ha="center", fontsize=5.5, color=INK2)
        ax.axhline(1.0, color=INK2, linewidth=0.5)
    except FileNotFoundError:
        pass
    ax.set_ylim(0, 1.38); ax.set_xlabel("Adam step"); ax.set_ylabel("estimate / truth")
    ax.legend(loc="lower right", handlelength=1.3, fontsize=6); despine(ax)

    ax = axs[1, 0]
    try:
        a, m = src.load_result(pinn_name("abs", 10, 0.02, 0))
        n = np.arange(1, 42)
        et, ee = a["eps_true"], a["eps_est"]
        ee = ee - ee[LIT_MASK].mean() + et[LIT_MASK].mean()
        ax.plot(n, et, color=INK2, linewidth=1.0, label=r"true $\varepsilon_n$")
        ax.plot(n[LIT_MASK], ee[LIT_MASK], color=SERIES[0], marker="o", linestyle="none", markersize=3, label="recovered (lit)")
        ax.plot(n[~LIT_MASK], ee[~LIT_MASK], color=SERIES[0], marker="o", linestyle="none", markersize=2, alpha=0.3, label="unlit (free)")
        ax.axvspan(21 - 4.5, 21 + 4.5, color="#f4f7fb", zorder=0)
        ax.set_ylim(min(et.min(), ee[LIT_MASK].min()) - 0.95, max(et.max(), ee[LIT_MASK].max()) + 0.1)
    except FileNotFoundError:
        pass
    ax.set_xlabel(r"site $n$"); ax.set_ylabel(r"$\varepsilon_n$")
    ax.legend(loc="lower left", ncol=1, handlelength=1.2, fontsize=5.5, labelspacing=0.2); despine(ax)

    ax = axs[1, 1]
    Ks = [str(K) for K in K_VALUES]
    x = np.arange(len(Ks))
    pinn = [s["err_g"].get("abs", {}).get(K, {}).get("0.05", (np.nan, 0.0)) for K in Ks]
    base = [s["baseline"].get("abs", {}).get(K, {}).get("0.05", {}).get("err_g", (np.nan, 0.0)) for K in Ks]
    pinn_g, pinn_s = np.array([v[0] for v in pinn], dtype=float), np.array([v[1] for v in pinn], dtype=float)
    base_g, base_s = np.array([v[0] for v in base], dtype=float), np.array([v[1] for v in base], dtype=float)
    ax.bar(x - 0.18, pinn_g, 0.34, yerr=pinn_s, color=SERIES[0], label="GI-PINN", error_kw=dict(linewidth=0.6, capsize=1.5))
    ax.bar(x + 0.18, base_g, 0.34, yerr=base_s, color=SERIES[1], label="RK4 least squares", error_kw=dict(linewidth=0.6, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in Ks]); ax.set_yscale("log"); ax.set_ylabel(r"rel. error of $g$")
    top = np.nanmax(np.concatenate([pinn_g + pinn_s, base_g + base_s, [1e-3]])) * 8
    ax.set_ylim(top=top)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.0); despine(ax)
    for ax, lab in zip(axs.flat, ("(a)", "(b)", "(c)", "(d)")):
        panel_label(ax, lab, x=-0.28 if lab in ("(a)", "(c)") else -0.12)
    fig.subplots_adjust(hspace=0.55, wspace=0.5, left=0.15, right=0.98, top=0.95, bottom=0.13)
    return _save(fig, Path(out) if out else FIGS_DIR / "fig3.pdf")

def _clabel(ax, cs, fmt, manual):
    try:
        ax.clabel(cs, fmt=fmt, fontsize=5, inline=True, manual=manual)
    except (TypeError, IndexError, ValueError):
        ax.clabel(cs, fmt=fmt, fontsize=5, inline=True)

def fig4(results_dir=None, out=None):
    set_style()
    src = _source(results_dir)
    maps = {h: src.load_result(f"fig4_map_h{h:g}")[0] for h in (0.0, 0.2)}
    regimes = src.load_json("fig4_regimes")
    fig = plt.figure(figsize=(DOUBLE_W, 2.55))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.06, 0.32, 1, 1], wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    gs2 = gs[0, 4:6].subgridspec(1, 2, wspace=0.62)
    axes += [fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])]
    for ax, h in zip(axes[:2], (0.0, 0.2)):
        m = maps[h]
        W, g = m["W_grid"], m["g_grid"]
        im = ax.imshow(m["p_trap"].T, origin="lower", aspect="auto", extent=[W[0], W[-1], g[0], g[-1]], cmap=SEQ_CMAP, vmin=0, vmax=1)
        cs = ax.contour(W, g, m["ipr"].T, levels=[0.2, 0.4], colors=[SERIES[1]], linewidths=0.7)
        _clabel(ax, cs, "IPR %.1f", [(1.0, 2.0), (3.0, 4.5)] if h == 0 else [(6.5, 1.0), (7.5, 4.0)])
        if h > 0:
            cs2 = ax.contour(W, g, m["com"].T, levels=[2, 5], colors=[SERIES[2]], linewidths=0.7, linestyles="--")
            _clabel(ax, cs2, r"$\Delta n$=%.0f", [(5.0, 2.0), (4.0, 5.3)])
        ax.set_xlabel(r"disorder $W$"); ax.set_title(rf"$h={h:g}$", fontsize=7)
    axes[0].set_ylabel(r"Kerr coefficient $g$"); axes[1].set_yticklabels([])
    fig.colorbar(im, cax=cax); cax.set_ylabel(r"$P_{\rm trap}$", fontsize=7, labelpad=2)
    for name, mk, col in (("skin", "^", SERIES[1]), ("anderson", "s", SERIES[2]), ("trapped", "o", SERIES[3])):
        r = regimes[name]
        axes[1].plot(r["W"], r["g"], marker=mk, color=col, markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                     linestyle="none", clip_on=False, zorder=6)

    ax = axes[2]
    off = 0
    for name, mk, col in (("skin", "^", SERIES[1]), ("anderson", "s", SERIES[2]), ("trapped", "o", SERIES[3])):
        try:
            a, m = src.load_result(f"fig4_profiles_{name}")
        except FileNotFoundError:
            continue
        n = np.arange(1, 42)
        Ir, Ip = np.abs(a["psi_ref_z10"]) ** 2, np.abs(a["psi_pinn_z10"]) ** 2
        sc = Ir.max() if Ir.max() > 0 else 1.0
        ax.plot(n, Ir / sc + off, color=INK2, linewidth=1.0)
        ax.plot(n[::2], Ip[::2] / sc + off, color=col, marker=mk, linestyle="none", markersize=2.5)
        ax.text(1.5, off + 0.62, f"{name.capitalize() if name == 'anderson' else name} (W={m['W']:g}, g={m['g']:g})", fontsize=5.5, color=col)
        off += 1.1
    ax.set_xlabel(r"site $n$"); ax.set_ylabel(r"$|\psi_n(z\!=\!10)|^2$ (norm., offset)"); ax.set_yticks([]); ax.set_ylim(-0.1, off + 0.05); despine(ax)

    ax = axes[3]
    try:
        s = src.summarize_fig3()
        pts = []
        for K in K_VALUES:
            for sig in SIGMAS:
                for p in src.glob(f"fig3_pinn_abs_K{K}_sig{sig:g}_s*.json"):
                    _, m = src.load_result(p)
                    if m.get("learn_eps", True):
                        pts.append((m["g_est"], m["h_est"], K))
        pts = np.array(pts)
        for j, K in enumerate(K_VALUES):
            sel = pts[:, 2] == K
            ax.plot(pts[sel, 0], pts[sel, 1], marker=MARKERS[j], linestyle="none", color=SERIES[j], markersize=3, alpha=0.8, label=f"K={K}")
        ax.plot([2.0], [0.2], marker="*", color=INK2, markersize=8, linestyle="none", label="truth")
    except Exception:
        pass
    ax.set_xlabel(r"inferred $g$"); ax.set_ylabel(r"inferred $h$")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi - lo) * 1.45)
    ax.legend(fontsize=5.5, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3); despine(ax)
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
        panel_label(ax, lab, x=-0.08 if lab == "(b)" else -0.3, y=1.03)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.17)
    return _save(fig, Path(out) if out else FIGS_DIR / "fig4.pdf")

def make_all(results_dir=None, figs_dir=None):
    figs_dir = Path(figs_dir) if figs_dir is not None else FIGS_DIR
    return [f(results_dir, figs_dir / f"fig{i}.pdf") for i, f in enumerate((fig1, fig2, fig3, fig4), start=1)]

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--fig", type=int)
    p.add_argument("--results")
    p.add_argument("--figs")
    p.add_argument("--export", action="store_true")
    a = p.parse_args(argv)
    if a.export:
        rec = Recorder(a.results)
        make_all(rec, a.figs)
        print(rec.save())
    elif a.all:
        for path in make_all(a.results, a.figs):
            print(path)
    elif a.fig:
        print({1: fig1, 2: fig2, 3: fig3, 4: fig4}[a.fig](a.results, (Path(a.figs) / f"fig{a.fig}.pdf") if a.figs else None))

if __name__ == "__main__":
    main()
