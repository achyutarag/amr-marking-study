"""
Runs the budget-matched sweep used for the paper's figures and tables and
writes one CSV row per run, so every number in the paper can be traced to
a row in results/. Figures are drawn separately by make_figures.py from
these CSVs (no simulation happens at plotting time).

For each scheme and each target final cell count, every marker's budget
parameter is tuned to hit the target (budget_match.match_param), then:
    - hysteresis and deterministic top-K: one run (they are deterministic)
    - Gumbel top-K: N_SEEDS runs with seeds 0..N_SEEDS-1
plus a uniform-grid run at a range of resolutions.

Each row reports both cost measures (weakness #3):
    cells         final cell count (degrees of freedom at t_final)
    cell_updates  sum over timesteps of the cell count (the work, since dt is
                  global and there is no subcycling)
and the turnover statistics (weakness #5): mean and standard deviation of
the Jaccard turnover over the run's regrid events, and the number of events.

Usage: python run_experiments.py --scheme muscl2   (writes results/sweep_muscl2_nu0.01.csv)
"""

import argparse
import csv
import os
import numpy as np
from amr_nonuniform import cell_widths, cell_centers
from pareto_frontier import run_uniform
from budget_match import run, match_param, errors, L, T_FINAL, SHOCK_HALF_WIDTH

N_SEEDS = 20
TARGETS = [48, 64, 96, 128, 160, 192, 256]
UNIFORM_N = [32, 48, 64, 96, 128, 160, 192, 256, 384, 512]
FIELDS = ["scheme", "nu", "method", "target", "param", "seed", "cells", "l2", "l2_shock", "l2_smooth",
          "cell_updates", "n_steps", "turnover_mean", "turnover_std", "n_turnover_pairs", "n_regrids", "mass_drift"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="muscl2", choices=["rusanov1", "muscl2"])
    ap.add_argument("--nu", type=float, default=0.01)
    args = ap.parse_args()
    common = dict(n0=32, L=L, nu=args.nu, t_final=T_FINAL, frac_lo=0.05,
                  regrid_dt=T_FINAL / 20, min_contrast=3.5, scheme=args.scheme)

    edges_ref, u_ref = run_uniform(4096, L, args.nu, T_FINAL, scheme=args.scheme)
    x_ref = cell_centers(edges_ref)
    dx_ref = L / 4096
    x_shock = x_ref[np.argmax(np.abs(np.gradient(u_ref)))]
    shock_mask = (np.abs((x_ref - x_shock + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH).astype(float)

    os.makedirs("results", exist_ok=True)
    path = f"results/sweep_{args.scheme}_nu{args.nu}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        base = dict(scheme=args.scheme, nu=args.nu)

        for n in UNIFORM_N:
            e, u, n_steps = run_uniform(n, L, args.nu, T_FINAL, scheme=args.scheme, return_steps=True)
            tot, sh, sm = errors(e, u, x_ref, u_ref, dx_ref, shock_mask)
            w.writerow(dict(base, method="uniform", target=n, param="", seed="", cells=n, l2=tot,
                            l2_shock=sh, l2_smooth=sm, cell_updates=n * n_steps, n_steps=n_steps,
                            turnover_mean="", turnover_std="", n_turnover_pairs=0, n_regrids=0, mass_drift=""))
        fh.flush()
        print(f"uniform done -> {path}", flush=True)

        for target in TARGETS:
            for marker in ["hysteresis", "topk", "gumbel_topk"]:
                p = match_param(marker, target, common)
                for seed in (range(N_SEEDS) if marker == "gumbel_topk" else [0]):
                    snaps, turnover, stats = run(marker, p, common, seed)
                    e, u = snaps[-1][1], snaps[-1][2]
                    m0 = np.sum(snaps[0][2] * cell_widths(snaps[0][1]))
                    tot, sh, sm = errors(e, u, x_ref, u_ref, dx_ref, shock_mask)
                    w.writerow(dict(base, method=marker, target=target, param=p, seed=seed, cells=len(u),
                                    l2=tot, l2_shock=sh, l2_smooth=sm, cell_updates=stats["cell_updates"],
                                    n_steps=stats["n_steps"],
                                    turnover_mean=np.mean(turnover) if turnover else "",
                                    turnover_std=np.std(turnover) if turnover else "",
                                    n_turnover_pairs=len(turnover),
                                    n_regrids=len(stats["n_refined"]),
                                    mass_drift=abs(np.sum(u * cell_widths(e)) - m0)))
                fh.flush()
                print(f"target {target} {marker} param={p} done", flush=True)
