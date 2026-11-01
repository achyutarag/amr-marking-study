"""
Work-matched comparison of AMR against uniform grids, without relying on
interpolation alone.

Step 1 (--uniform): run the uniform grid at every n = 32, 36, ..., 320 so
that every AMR work budget is bracketed by ACTUAL uniform runs whose work
(cell updates) differs from it by only a few percent. Writes
results/uniform_dense_<scheme>_nu0.01.csv.

Step 2 (default): for every AMR configuration (method x target x
indicator x levels; Gumbel averaged over seeds), find the two uniform
runs whose work brackets the AMR run's mean work, and report:
    uni_lo_n, uni_lo_work, uni_lo_l2   actual uniform run just BELOW the AMR work
    uni_hi_n, uni_hi_work, uni_hi_l2   actual uniform run just ABOVE the AMR work
    ratio_* = uniform error / AMR error under four comparison rules
        loglog        log(error) linear in log(work) between the brackets
        linear        error linear in work between the brackets
        nearest       the bracketing run closest in work
        conservative  the run ABOVE (uniform is given at least as much work
                      as AMR; since uniform error decreases with n this is
                      the comparison least favourable to AMR)
    ratio > 1 means AMR is more accurate at matched work. A claim of an AMR
    advantage is only made where ratio_conservative > 1. For Gumbel top-K
    seeds_win is the fraction of seeds whose error is below the uniform
    error at THAT seed's own work (log-log rule).

Usage:
    python work_matched.py --uniform         (once; ~minutes)
    python work_matched.py                   (writes results/work_matched_<scheme>_nu0.01.csv)
"""

import argparse
import glob
import numpy as np
import pandas as pd
from amr_nonuniform import cell_centers
from pareto_frontier import run_uniform
from budget_match import errors, L, T_FINAL, SHOCK_HALF_WIDTH

NU = 0.01
DENSE_N = list(range(32, 321, 4))


def uniform_sweep(scheme):
    er, ur = run_uniform(4096, L, NU, T_FINAL, scheme=scheme)
    xr = cell_centers(er)
    xs = xr[np.argmax(np.abs(np.gradient(ur)))]
    mask = (np.abs((xr - xs + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH).astype(float)
    rows = []
    for n in DENSE_N:
        e, u, n_steps = run_uniform(n, L, NU, T_FINAL, scheme=scheme, return_steps=True)
        tot, sh, sm = errors(e, u, xr, ur, L / 4096, mask)
        rows.append(dict(n=n, n_steps=n_steps, work=n * n_steps, l2=tot, l2_shock=sh, l2_smooth=sm))
        print(f"uniform n={n} work={n * n_steps} l2={tot:.4e}", flush=True)
    pd.DataFrame(rows).to_csv(f"results/uniform_dense_{scheme}_nu{NU}.csv", index=False)


def uniform_at(work, uni):
    """Bracketing uniform runs and the four comparison values at a given work."""
    uni = uni.sort_values("work")
    if work < uni.work.iloc[0] or work > uni.work.iloc[-1]:
        return None
    hi = uni[uni.work >= work].iloc[0]
    lo = uni[uni.work <= work].iloc[-1]
    if hi.work == lo.work:
        loglog = linear = hi.l2
    else:
        f_log = (np.log(work) - np.log(lo.work)) / (np.log(hi.work) - np.log(lo.work))
        loglog = float(np.exp(np.log(lo.l2) + f_log * (np.log(hi.l2) - np.log(lo.l2))))
        f_lin = (work - lo.work) / (hi.work - lo.work)
        linear = float(lo.l2 + f_lin * (hi.l2 - lo.l2))
    nearest = lo.l2 if (work - lo.work) <= (hi.work - work) else hi.l2
    return lo, hi, dict(loglog=loglog, linear=linear, nearest=nearest, conservative=hi.l2)


def load_amr(scheme):
    """All AMR runs for this scheme: main sweep (gradient, 3 levels) + indicator ablation."""
    frames = []
    main = pd.read_csv(f"results/sweep_{scheme}_nu{NU}.csv")
    main = main[main.method != "uniform"].assign(indicator="gradient", levels=3, source="main sweep")
    frames.append(main)
    for f in glob.glob(f"results/ablation_{scheme}_nu{NU}_*_flo0.05.csv"):
        frames.append(pd.read_csv(f).assign(source="indicator ablation"))
    cols = ["source", "indicator", "levels", "method", "target", "seed", "cells", "l2", "cell_updates"]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


def table(scheme):
    uni = pd.read_csv(f"results/uniform_dense_{scheme}_nu{NU}.csv")
    amr = load_amr(scheme)
    rows = []
    for key, g in amr.groupby(["source", "indicator", "levels", "method", "target"]):
        work = g.cell_updates.mean()
        b = uniform_at(work, uni)
        if b is None:
            continue
        lo, hi, ref = b
        seed_wins = [r.l2 < uniform_at(r.cell_updates, uni)[2]["loglog"]
                     for r in g.itertuples() if uniform_at(r.cell_updates, uni) is not None]
        row = dict(zip(["source", "indicator", "levels", "method", "target"], key),
                   n_runs=len(g), cells=g.cells.mean(), work=work, amr_l2=g.l2.mean(),
                   amr_l2_std=g.l2.std() if len(g) > 1 else 0.0,
                   uni_lo_n=int(lo.n), uni_lo_work=int(lo.work), uni_lo_l2=lo.l2,
                   uni_hi_n=int(hi.n), uni_hi_work=int(hi.work), uni_hi_l2=hi.l2,
                   **{f"uni_{k}": v for k, v in ref.items()},
                   **{f"ratio_{k}": v / g.l2.mean() for k, v in ref.items()},
                   seeds_win=np.mean(seed_wins) if len(g) > 1 else "")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"results/work_matched_{scheme}_nu{NU}.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="muscl2", choices=["rusanov1", "muscl2"])
    ap.add_argument("--uniform", action="store_true")
    args = ap.parse_args()
    if args.uniform:
        uniform_sweep(args.scheme)
    else:
        out = table(args.scheme)
        pd.set_option("display.width", 250)
        show = ["indicator", "levels", "method", "target", "cells", "work", "amr_l2", "uni_lo_n", "uni_lo_l2",
                "uni_hi_n", "uni_hi_l2", "ratio_loglog", "ratio_linear", "ratio_nearest", "ratio_conservative",
                "seeds_win"]
        print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
