"""
Writes the compact tables used in paper/main.tex, directly from the CSVs in
results/, so no number in a paper table is typed by hand.

Output: paper/tables/tab_*.tex (tabular environments only; captions and
labels live in main.tex). Usage: python make_paper_tables.py
"""

import os
import numpy as np
import pandas as pd

NU = 0.01
NAME = {"hysteresis": "Threshold", "topk": "Top-$K$", "gumbel_topk": "Gumbel top-$K$", "uniform": "Uniform"}
MARKERS = ["hysteresis", "topk", "gumbel_topk"]
OUT = "paper/tables"


def sci(x, digits=2):
    """3.45e-03 -> 3.45\\times10^{-3} in math mode."""
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def err(d, gumbel):
    s = f"{d.l2.mean() * 1e3:.2f}"
    return s + (f" $\\pm$ {d.l2.std() * 1e3:.2f}" if gumbel else "")


def turnover(d):
    if "n_turnover_pairs" not in d or d.turnover_mean.isna().all():
        return "--"
    return f"{d.turnover_mean.mean():.2f} ({d.n_turnover_pairs.mean():.0f})"


def write(name, lines):
    with open(f"{OUT}/{name}.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def tab_sweep():
    """Budget-matched comparison, both schemes, targets 64/128/192."""
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Target & Method & Cells & $L^2$ error ($\times10^{-3}$) & Work ($\times10^{3}$) & Turnover \\",
             r"\midrule"]
    for scheme, title in [("muscl2", "Second-order scheme (MUSCL, SSP-RK2)"),
                          ("rusanov1", "First-order scheme (Rusanov, forward Euler)")]:
        df = pd.read_csv(f"results/sweep_{scheme}_nu{NU}.csv")
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for target in [64, 128, 192]:
            for i, m in enumerate(MARKERS + ["uniform"]):
                d = df[(df.method == m) & (df.target == target)]
                lines.append(f"{target if i == 0 else ''} & {NAME[m]} & {d.cells.mean():.0f} & "
                             f"{err(d, m == 'gumbel_topk')} & {d.cell_updates.mean() / 1e3:.1f} & "
                             f"{turnover(d) if m != 'uniform' else '--'} \\\\")
            lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("tab_sweep", lines)


def tab_timing():
    """Early-phase control: final error per arm and marker, share of the A->C gap recovered."""
    tc = pd.read_csv(f"results/timing_control_muscl2_nu{NU}.csv")
    mean = tc.groupby(["marker", "arm"]).l2.mean()
    arms = [("A base", "A: base mesh (gated)"), ("E away", "E: fine away from front"),
            ("D front", "D: fine around front"), ("C all", "C: fine everywhere")]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Early-phase mesh & Error at $t_g$ & Threshold & Top-$K$ & Gumbel \\",
             r" & ($\times10^{-3}$) & \multicolumn{3}{c}{final $L^2$ error ($\times10^{-3}$) [gap recovered]} \\",
             r"\midrule"]
    for arm, label in arms:
        err_tg = tc[tc.arm == arm].err_tg.mean()
        cells = []
        for m in MARKERS:
            a, c, v = mean[(m, "A base")], mean[(m, "C all")], mean[(m, arm)]
            share = "" if arm in ("A base", "C all") else f" [{(a - v) / (a - c):.0%}]".replace("%", r"\%")
            cells.append(f"{v * 1e3:.2f}{share}")
        lines.append(f"{label} & {err_tg * 1e3:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_timing", lines)


def tab_indicator():
    """Gradient vs curvature indicator, 3 levels: accuracy with its cost."""
    df = pd.concat([pd.read_csv(f"results/ablation_muscl2_nu{NU}_{ind}_L3_flo0.05.csv")
                    for ind in ["gradient", "curvature"]])
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Target & Indicator / method & Cells & $L^2$ ($\times10^{-3}$) & Work ($\times10^{3}$) & Steps "
             r"& Refined/merged & Regrids & Turnover \\", r"\midrule"]
    for target in [48, 64, 80]:
        first = True
        for ind in ["gradient", "curvature"]:
            for m in MARKERS:
                d = df[(df.indicator == ind) & (df.method == m) & (df.target == target)]
                lines.append(f"{target if first else ''} & {ind}, {NAME[m]} & {d.cells.mean():.0f} & "
                             f"{err(d, m == 'gumbel_topk')} & {d.cell_updates.mean() / 1e3:.1f} & "
                             f"{d.n_steps.mean():.0f} & {d.refined.mean():.0f}/{d.coarsened.mean():.0f} & "
                             f"{d.n_regrids.mean():.1f} & {turnover(d)} \\\\")
                first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("tab_indicator", lines)


def tab_work():
    """Work-matched comparison against the two actual bracketing uniform runs (3 levels)."""
    wm = pd.read_csv(f"results/work_matched_muscl2_nu{NU}.csv")
    wm = wm[(wm.source == "indicator ablation") & (wm.levels == 3)]
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Target & Indicator / method & Work ($\times10^{3}$) & AMR $L^2$ & \multicolumn{2}{c}{Uniform below} "
             r"& \multicolumn{2}{c}{Uniform above} & Ratio \\",
             r" & & & ($\times10^{-3}$) & $n$ & $L^2$ & $n$ & $L^2$ & log-log / cons. \\", r"\midrule"]
    for target in [48, 64, 80]:
        first = True
        for ind in ["gradient", "curvature"]:
            for m in MARKERS:
                r = wm[(wm.indicator == ind) & (wm.method == m) & (wm.target == target)].iloc[0]
                win = f" [{float(r.seeds_win):.0%}]".replace("%", r"\%") if m == "gumbel_topk" else ""
                lines.append(f"{target if first else ''} & {ind}, {NAME[m]} & {r.work / 1e3:.1f} & "
                             f"{r.amr_l2 * 1e3:.2f}{win} & {r.uni_lo_n} & {r.uni_lo_l2 * 1e3:.2f} & "
                             f"{r.uni_hi_n} & {r.uni_hi_l2 * 1e3:.2f} & "
                             f"{r.ratio_loglog:.2f} / \\textbf{{{r.ratio_conservative:.2f}}} \\\\")
                first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("tab_work", lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tab_sweep()
    tab_timing()
    tab_indicator()
    tab_work()
    print("wrote", sorted(os.listdir(OUT)))
