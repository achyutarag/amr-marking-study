"""
Coarsening-threshold ablation.

Finding that motivates it: with frac_lo = 0.05, about 82% of the cells in
a ~128-cell AMR mesh lie in smooth regions, refined below the base width.
The indicator eta = h |u_x| on the smooth ramp is ~23% of its maximum
when the contrast gate first opens (t = 0.105), so the first regrid
events refine much of the domain; by t = 0.3 those cells sit at ~10% of
the maximum, above frac_lo = 0.05, so they are never coarsened back.

This script varies ONLY frac_lo (shared by all markers), re-matches every
marker's budget parameter to the same target final cell counts, and
records how concentrated the mesh is, alongside error and cost:
    front_frac    fraction of final cells within |x - x_front| < 0.05
    smooth_ref    number of final cells outside that window that are
                  finer than the base mesh
    coarsened     total pairs merged over the run (refine/coarsen churn)
    n_steps       timesteps taken; with one global dt set by the smallest
                  cell, this is what makes deep refinement expensive
    min_width     smallest cell width reached during the run
Hysteresis keeps frac_hi >= frac_lo + 0.02 (see budget_match.match_param).

Usage: python coarsen_ablation.py [--scheme muscl2] [--frac_lo 0.05 0.10 0.15 0.20]
                                  [--indicator gradient|curvature] [--levels 3] [--targets ...]
Writes results/ablation_<scheme>_nu0.01_<indicator>_L<levels>_flo<frac_lo values>.csv

With --indicator curvature and a single frac_lo this is the indicator
ablation: same markers, budgets, gate and coarsening rule, but refinement
and coarsening are driven by h^2|u_xx| instead of h|u_x|.
"""

import argparse
import csv
import numpy as np
from amr_nonuniform import cell_widths, cell_centers
from pareto_frontier import run_uniform
from budget_match import run, match_param, errors, L, T_FINAL, SHOCK_HALF_WIDTH

N_SEEDS = 10
TARGETS = [64, 96, 128, 192]
NU = 0.01

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="muscl2", choices=["rusanov1", "muscl2"])
    ap.add_argument("--frac_lo", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--indicator", default="gradient", choices=["gradient", "curvature"])
    ap.add_argument("--levels", type=int, default=3, help="max refinement levels below the base mesh")
    ap.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    args = ap.parse_args()

    edges_ref, u_ref = run_uniform(4096, L, NU, T_FINAL, scheme=args.scheme)
    x_ref = cell_centers(edges_ref)
    dx_ref = L / 4096
    x_shock = x_ref[np.argmax(np.abs(np.gradient(u_ref)))]
    shock_mask = (np.abs((x_ref - x_shock + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH).astype(float)
    base = L / 32

    fields = ["scheme", "indicator", "levels", "frac_lo", "method", "target", "param", "seed", "cells", "front_frac", "smooth_ref",
              "l2", "l2_shock", "l2_smooth", "cell_updates", "n_steps", "min_width", "refined", "coarsened", "turnover_mean",
              "turnover_std", "n_turnover_pairs", "n_regrids"]
    # every setting that changes the runs is in the file name, so no two
    # configurations can write to the same CSV
    flo = "-".join(f"{f:g}" for f in args.frac_lo)
    path = f"results/ablation_{args.scheme}_nu{NU}_{args.indicator}_L{args.levels}_flo{flo}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for frac_lo in args.frac_lo:
            common = dict(n0=32, L=L, nu=NU, t_final=T_FINAL, frac_lo=frac_lo,
                          regrid_dt=T_FINAL / 20, min_contrast=3.5, scheme=args.scheme,
                          indicator=args.indicator, max_level_width=(L / 32) / 2**args.levels)
            for target in args.targets:
                for marker in ["hysteresis", "topk", "gumbel_topk"]:
                    p = match_param(marker, target, common)
                    for seed in (range(N_SEEDS) if marker == "gumbel_topk" else [0]):
                        snaps, turnover, st = run(marker, p, common, seed)
                        e, u = snaps[-1][1], snaps[-1][2]
                        x = cell_centers(e)
                        in_front = np.abs((x - x_shock + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH
                        tot, sh, sm = errors(e, u, x_ref, u_ref, dx_ref, shock_mask)
                        w.writerow(dict(scheme=args.scheme, indicator=args.indicator, levels=args.levels, frac_lo=frac_lo, method=marker, target=target,
                                        param=p, seed=seed, cells=len(u), front_frac=in_front.mean(),
                                        smooth_ref=int(np.sum(~in_front & (cell_widths(e) < base * (1 - 1e-9)))),
                                        l2=tot, l2_shock=sh, l2_smooth=sm, cell_updates=st["cell_updates"],
                                        n_steps=st["n_steps"], min_width=st["min_width"],
                                        refined=sum(st["n_refined"]), coarsened=sum(st["n_coarsened"]),
                                        turnover_mean=np.mean(turnover) if turnover else "",
                                        turnover_std=np.std(turnover) if turnover else "",
                                        n_turnover_pairs=len(turnover),
                                        n_regrids=len(st["n_refined"])))
                    fh.flush()
                    print(f"frac_lo={frac_lo} target={target} {marker} param={p} done", flush=True)
