"""
Draws every figure and table in the paper.

Figures 4-6 and both tables are read from the CSVs written by
run_experiments.py and timing_control.py -- no simulation happens for
them here. Figures 1-3 show single mesh snapshots, which are cheap and
deterministic, so they are re-simulated with parameters read from the
same CSVs (the matched budget parameter for target 128).

Output: figures/fig*.pdf (+ .png previews) and results/tables.tex.
Usage:  python make_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from amr_nonuniform import run_adaptive, cell_centers, cell_widths, gradient_indicator_nonuniform

NU = 0.01
COL, PAGE = 3.5, 7.16           # IEEE single- and double-column widths (inches)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-3 of the validated reference palette, fixed per marker
# (never re-assigned by rank); uniform is the neutral baseline
COLOR = {"hysteresis": "#2a78d6", "topk": "#eb6834", "gumbel_topk": "#1baf7a", "uniform": INK2}
SHAPE = {"hysteresis": "s", "topk": "^", "gumbel_topk": "D", "uniform": "o"}
LABEL = {"hysteresis": "Threshold (hysteresis)", "topk": "Deterministic top-$K$",
         "gumbel_topk": "Gumbel top-$K$ ($T=0.1$)", "uniform": "Uniform grid"}
SCHEME = {"muscl2": "second order (MUSCL, SSP-RK2)", "rusanov1": "first order (Rusanov, Euler)"}
MARKERS = ["hysteresis", "topk", "gumbel_topk"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)


def load_sweep(scheme):
    return pd.read_csv(f"results/sweep_{scheme}_nu{NU}.csv")


def summarize(df):
    """Mean and std per (method, target); deterministic methods have std 0."""
    g = df.groupby(["method", "target"])
    out = g[["cells", "l2", "l2_shock", "l2_smooth", "cell_updates", "turnover_mean", "n_regrids"]].mean()
    out["l2_std"] = g["l2"].std().fillna(0.0)
    out["turnover_sd"] = g["turnover_mean"].std().fillna(0.0)
    out["n_runs"] = g.size()
    out["param"] = g["param"].first()
    return out.reset_index()


def adaptive_run(marker, param, scheme="muscl2", seed=0):
    kw = {"frac_hi": float(param)} if marker == "hysteresis" else {"K": int(float(param))}
    if marker == "gumbel_topk":
        kw["temperature"] = 0.1
    snaps, _ = run_adaptive(n0=32, L=1.0, nu=NU, t_final=0.3, marker=marker, marker_kwargs=kw,
                            frac_lo=0.05, regrid_dt=0.015, min_contrast=3.5, seed=seed, scheme=scheme)
    return snaps


def fig_solution_and_mesh(s):
    """Fig 1: final solution and indicator. Fig 2: mesh history. Both for threshold marking at target 128."""
    p = s[(s.method == "hysteresis") & (s.target == 128)].param.iloc[0]
    snaps = adaptive_run("hysteresis", p)
    t, e, u = snaps[-1]
    x = cell_centers(e)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.8), sharex=True)
    a1.plot(x, u, color=COLOR["hysteresis"], marker=".", ms=2.5, lw=1.0)
    a1.set_ylabel("$u(x, T)$")
    a2.plot(x, gradient_indicator_nonuniform(u, e), color=INK2, lw=1.0)
    a2.set_ylabel(r"$\eta_i = h_i\,|\partial_x u|_i$")
    a2.set_xlabel("$x$")
    save(fig, "fig1_solution_indicator")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE, 2.2), gridspec_kw={"width_ratios": [2, 1]})
    for ts, es, _ in snaps:
        xc = cell_centers(es)
        a1.scatter(xc, np.full_like(xc, ts), s=3, color=COLOR["hysteresis"], lw=0)
    a1.set_xlabel("$x$")
    a1.set_ylabel("$t$ at regrid")
    a1.set_title("Cell centers after each regrid event")
    a2.step([sn[0] for sn in snaps], [len(sn[2]) for sn in snaps], where="post", color=COLOR["hysteresis"])
    a2.set_xlabel("$t$")
    a2.set_ylabel("cells")
    a2.set_title("Cell count")
    save(fig, "fig2_mesh_history")


def fig_final_meshes(s):
    """Fig 3: final cell width vs x, the three markers at target 128."""
    fig, axes = plt.subplots(3, 1, figsize=(COL, 4.0), sharex=True, sharey=True, layout="constrained")
    for ax, m in zip(axes, MARKERS):
        p = s[(s.method == m) & (s.target == 128)].param.iloc[0]
        _, e, u = adaptive_run(m, p)[-1]
        ax.scatter(cell_centers(e), cell_widths(e), s=9, color=COLOR[m], marker=SHAPE[m], lw=0)
        ax.set_ylim(1 / 400, 1 / 22)
        ax.set_yscale("log", base=2)
        ax.set_yticks([1 / 32, 1 / 64, 1 / 128, 1 / 256], ["1/32", "1/64", "1/128", "1/256"])
        ax.set_title(f"{LABEL[m]}: {len(u)} cells", loc="left")
        ax.set_ylabel("$h_i$")
    axes[-1].set_xlabel("$x$")
    save(fig, "fig3_final_meshes")


def fig_error_vs_cost(sums):
    """Fig 4: L2 error vs final cells and vs cell updates, both schemes."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 4.6))
    for row, scheme in enumerate(["muscl2", "rusanov1"]):
        s = sums[scheme]
        for col, cost in enumerate(["cells", "cell_updates"]):
            ax = axes[row, col]
            for m in ["uniform"] + MARKERS:
                d = s[s.method == m].sort_values(cost)
                ax.errorbar(d[cost], d.l2, yerr=d.l2_std if m == "gumbel_topk" else None,
                            color=COLOR[m], marker=SHAPE[m], ms=4, lw=1.4, capsize=2,
                            ls="--" if m == "uniform" else "-", label=LABEL[m])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("final cell count" if cost == "cells" else "cell updates (work)")
            ax.set_ylabel("$L^2$ error")
            ax.set_title(f"{SCHEME[scheme]}, cost = {'cells' if cost == 'cells' else 'work'}", loc="left")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig4_error_vs_cost")


def fig_turnover(s):
    """Fig 5: mean grid turnover per run vs final cells (second-order scheme)."""
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for m in MARKERS:
        d = s[s.method == m].sort_values("cells")
        ax.errorbar(d.cells, d.turnover_mean, yerr=d.turnover_sd if m == "gumbel_topk" else None,
                    color=COLOR[m], marker=SHAPE[m], ms=4, capsize=2, label=LABEL[m])
    ax.set_xscale("log")
    ax.set_xlabel("final cell count")
    ax.set_ylabel("mean Jaccard turnover")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig5_turnover")


def fig_timing(tc):
    """Fig 6: final error by early-phase mesh arm (second-order scheme)."""
    arms = ["A base", "E away", "D front", "C all"]
    arm_label = {"A base": "A: base mesh (gated)", "E away": "E: refine away from front",
                 "D front": "D: refine around front", "C all": "C: refine everywhere"}
    # categorical slots 1-4 in fixed order (validated adjacent-pair palette);
    # the values are also in Table II, which serves as the table view
    arm_color = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    width = 0.2
    for j, (arm, c) in enumerate(zip(arms, arm_color)):
        g = tc[tc.arm == arm].groupby("marker").l2
        mean, sd = g.mean().reindex(MARKERS), g.std().fillna(0).reindex(MARKERS)
        xs = np.arange(len(MARKERS)) + (j - 1.5) * width
        ax.bar(xs, mean, width * 0.9, color=c, label=arm_label[arm],
               yerr=sd.where(sd > 0), capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(MARKERS)), ["Threshold", "Top-$K$", "Gumbel top-$K$"])
    ax.set_ylabel("final $L^2$ error")
    ax.grid(axis="x", visible=False)
    # legend fills column-first; reorder so it reads row-wise in bar order A, E / D, C
    h, lab = ax.get_legend_handles_labels()
    ax.legend([h[i] for i in (0, 2, 1, 3)], [lab[i] for i in (0, 2, 1, 3)],
              frameon=False, fontsize=6, loc="upper left", ncol=2)
    ax.set_ylim(0, tc[tc.arm.isin(arms)].groupby(["marker", "arm"]).l2.mean().max() * 1.45)
    save(fig, "fig6_timing_control")


def tables(sums, tc):
    lines = []
    lines.append("% Table I: budget-matched comparison (generated by make_figures.py)")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & Method & Cells & $L^2$ error & Shock & Smooth & Work \\")
    lines.append(r"\midrule")
    for scheme in ["muscl2", "rusanov1"]:
        s = sums[scheme]
        for target in [64, 128, 192]:
            for m in MARKERS + ["uniform"]:
                d = s[(s.method == m) & (s.target == target)]
                if d.empty:
                    continue
                r = d.iloc[0]
                err = f"{r.l2:.2e}" + (f" $\\pm$ {r.l2_std:.1e}" if m == "gumbel_topk" else "")
                lines.append(f"{'2nd' if scheme == 'muscl2' else '1st'} & {LABEL[m]} & {r.cells:.0f} & {err} & "
                             f"{r.l2_shock:.2e} & {r.l2_smooth:.2e} & {r.cell_updates:.2e} \\\\")
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append("% Table II: early-phase control, second-order scheme (generated by make_figures.py)")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Marker & Arm & Error at $t_g$ & Final cells & Final $L^2$ & Work \\")
    lines.append(r"\midrule")
    for m in MARKERS:
        for arm in ["A base", "E away", "D front", "C all", "B ungated"]:
            d = tc[(tc.marker == m) & (tc.arm == arm)]
            err_tg = f"{d.err_tg.mean():.2e}" if d.err_tg.notna().any() else "--"
            lines.append(f"{LABEL[m]} & {arm} & {err_tg} & {d.cells.mean():.0f} & "
                         f"{d.l2.mean():.2e} & {d.cell_updates.mean():.2e} \\\\")
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines += indicator_cost_table() + work_matched_table()
    with open("results/tables.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def turnover_cell(d):
    """Turnover mean with the number of event pairs it rests on; '--' where undefined."""
    if d.turnover_mean.isna().all():
        return "--"
    return f"{d.turnover_mean.mean():.2f} ({d.n_turnover_pairs.mean():.0f})"


def indicator_cost_table():
    """Table III: accuracy AND cost of the gradient vs curvature indicator, both depths."""
    df = pd.concat([pd.read_csv(f"results/ablation_muscl2_nu{NU}_{ind}_L{lv}_flo0.05.csv")
                    for ind in ["gradient", "curvature"] for lv in [3, 5]])
    lines = ["", "% Table III: indicator ablation, accuracy and cost (generated by make_figures.py)",
             "% turnover column: mean Jaccard turnover (mean number of event pairs it rests on); -- = undefined",
             r"\begin{tabular}{rllrrrrrrr}", r"\toprule",
             r"Levels & Indicator & Method & Cells & $L^2$ error & Work & Steps & Refined/merged & Regrids & Turnover \\",
             r"\midrule"]
    for lv in [3, 5]:
        for target in sorted(df[df.levels == lv].target.unique()):
            for ind in ["gradient", "curvature"]:
                for m in MARKERS:
                    d = df[(df.levels == lv) & (df.indicator == ind) & (df.method == m) & (df.target == target)]
                    err = f"{d.l2.mean():.2e}" + (f" $\\pm$ {d.l2.std():.1e}" if m == "gumbel_topk" else "")
                    lines.append(f"{lv} & {ind} & {LABEL[m]} & {d.cells.mean():.0f} & {err} & "
                                 f"{d.cell_updates.mean():.2e} & {d.n_steps.mean():.0f} & "
                                 f"{d.refined.mean():.0f}/{d.coarsened.mean():.0f} & {d.n_regrids.mean():.1f} & "
                                 f"{turnover_cell(d)} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def work_matched_table():
    """Table IV: AMR vs the two ACTUAL uniform runs bracketing its work, and the ratio under each rule."""
    wm = pd.read_csv(f"results/work_matched_muscl2_nu{NU}.csv")
    keep = (wm.source == "indicator ablation") | ((wm.source == "main sweep") & wm.target.isin([128, 192, 256]))
    wm = wm[keep].sort_values(["levels", "indicator", "target", "method"])
    lines = ["", "% Table IV: work-matched comparison with uniform grids (generated by make_figures.py)",
             "% ratio = uniform error / AMR error; >1 favours AMR. cons. = uniform run with MORE work than AMR",
             r"\begin{tabular}{rllrrrrrrrrr}", r"\toprule",
             r"Lv & Ind. & Method & Cells & Work & AMR $L^2$ & Uniform below ($n$, $L^2$) & Uniform above ($n$, $L^2$) "
             r"& log-log & linear & nearest & cons. \\", r"\midrule"]
    for r in wm.itertuples():
        win = f" [{r.seeds_win:.0%} seeds]" if isinstance(r.seeds_win, float) and not np.isnan(r.seeds_win) else ""
        lines.append(f"{r.levels} & {r.indicator} & {LABEL[r.method]} & {r.cells:.0f} & {r.work:.2e} & "
                     f"{r.amr_l2:.2e}{win} & "
                     f"{r.uni_lo_n}, {r.uni_lo_l2:.2e} & {r.uni_hi_n}, {r.uni_hi_l2:.2e} & "
                     f"{r.ratio_loglog:.2f} & {r.ratio_linear:.2f} & {r.ratio_nearest:.2f} & "
                     f"\\textbf{{{r.ratio_conservative:.2f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


if __name__ == "__main__":
    sweeps = {sc: load_sweep(sc) for sc in ["muscl2", "rusanov1"]}
    sums = {sc: summarize(df) for sc, df in sweeps.items()}
    tc = pd.read_csv(f"results/timing_control_muscl2_nu{NU}.csv")
    fig_solution_and_mesh(sums["muscl2"])
    fig_final_meshes(sums["muscl2"])
    fig_error_vs_cost(sums)
    fig_turnover(sums["muscl2"])
    fig_timing(tc)
    tables(sums, tc)
    for sc, df in sweeps.items():
        print(f"{sc}: {len(df)} runs, max mass drift {pd.to_numeric(df.mass_drift, errors='coerce').max():.1e}")
    print("wrote figures/ and results/tables.tex")
