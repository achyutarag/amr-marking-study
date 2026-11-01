"""
Budget-matched comparison of the three marking strategies.

The earlier comparisons matched budgets only loosely: the demo derived K
from the hysteresis run and ended at 86 / 51 / 60 final cells, and the
Pareto sweep used independent K and frac_hi grids, so each method's
points landed at different cell counts. Here each method's budget
parameter (frac_hi for hysteresis, K for the top-K markers) is tuned
separately to hit a common TARGET final cell count, and every run is
reported with both cost measures:
    - final cell count (degrees of freedom at t_final), and
    - cell updates (sum of cell count over all timesteps), which is the
      actual work, since the solver uses one global dt set by the
      smallest cell.
The L2 error is also split into a window of half-width 0.05 around the
front ("shock") and the rest of the domain ("smooth"), to separate
error made at the feature from error made where the indicator does
not refine.

Final cell count is not exactly monotone in K or frac_hi (coarsening
interacts with refinement), so the search scans a coarse grid and then
refines locally around the closest point rather than bisecting.

Usage:
    python budget_match.py                         # original setting (rusanov1, nu=0.01)
    python budget_match.py --scheme muscl2         # second-order scheme
    python budget_match.py --scheme muscl2 --nu 0.03
The reference solution always uses the same scheme and nu as the runs.
"""

import argparse
import numpy as np
from amr_nonuniform import run_adaptive, cell_widths, cell_centers
from pareto_frontier import run_uniform

L, T_FINAL = 1.0, 0.3
TEMPERATURE = 0.1
N_SEEDS = 10
SHOCK_HALF_WIDTH = 0.05


def run(marker, param, common, seed=0):
    if marker == "hysteresis":
        kw = {"frac_hi": param}
    elif marker == "topk":
        kw = {"K": int(param)}
    else:
        kw = {"K": int(param), "temperature": TEMPERATURE}
    stats = {}
    snaps, turnover = run_adaptive(marker=marker, marker_kwargs=kw, seed=seed, stats=stats, **common)
    return snaps, turnover, stats


def match_param(marker, target, common):
    """Return the budget parameter whose (seed-0) final cell count is closest to target."""
    final_cells = lambda p: len(run(marker, p, common)[0][-1][2])
    if marker == "hysteresis":
        # the refine threshold must stay above the coarsen threshold, or a
        # cell can be refined and merged back in the same regrid event
        lo = common.get("frac_lo", 0.05) + 0.02
        grid = [p for p in np.linspace(0.05, 0.8, 16) if p >= lo]
        local = lambda p: [q for q in np.linspace(p - 0.05, p + 0.05, 11) if q >= lo]
    else:
        grid = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
        local = lambda p: list(range(max(1, int(p) - 4), int(p) + 5))
    best = min(grid, key=lambda p: abs(final_cells(p) - target))
    return min(local(best), key=lambda p: abs(final_cells(p) - target))


def errors(edges, u, x_ref, u_ref, dx_ref, shock_mask):
    """(total, shock-window, smooth-region) L2 error against the reference."""
    sq = (np.interp(x_ref, cell_centers(edges), u, period=L) - u_ref) ** 2
    return tuple(np.sqrt(dx_ref * np.sum(sq * m)) for m in (1.0, shock_mask, 1.0 - shock_mask))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="rusanov1", choices=["rusanov1", "muscl2"])
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--targets", type=int, nargs="+", default=[64, 96, 128, 192])
    args = ap.parse_args()
    common = dict(n0=32, L=L, nu=args.nu, t_final=T_FINAL, frac_lo=0.05,
                  regrid_dt=T_FINAL / 20, min_contrast=3.5, scheme=args.scheme)

    edges_ref, u_ref = run_uniform(4096, L, args.nu, T_FINAL, scheme=args.scheme)
    x_ref = cell_centers(edges_ref)
    dx_ref = L / 4096
    x_shock = x_ref[np.argmax(np.abs(np.gradient(u_ref)))]
    shock_mask = (np.abs((x_ref - x_shock + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH).astype(float)

    print(f"scheme={args.scheme} nu={args.nu}  front at x={x_shock:.4f}; "
          f"numerical/physical viscosity on base cells ~ {1.25 * (L / 32) / 2 / args.nu:.2f} (rusanov1 only)")
    print(f"{'method':12s} {'target':>6s} {'param':>7s} {'cells':>11s} {'L2 error':>21s} "
          f"{'shock':>9s} {'smooth':>9s} {'cell updates':>13s} {'turnover':>13s}  refined/event (seed 0)")
    for target in args.targets:
        for marker in ["hysteresis", "topk", "gumbel_topk"]:
            p = match_param(marker, target, common)
            seeds = range(N_SEEDS) if marker == "gumbel_topk" else [0]
            cells, errs, work, turn = [], [], [], []
            for s in seeds:
                snaps, turnover, stats = run(marker, p, common, s)
                edges, u = snaps[-1][1], snaps[-1][2]
                assert abs(np.sum(u * cell_widths(edges)) - np.sum(snaps[0][2] * cell_widths(snaps[0][1]))) < 1e-12
                cells.append(len(u))
                errs.append(errors(edges, u, x_ref, u_ref, dx_ref, shock_mask))
                work.append(stats["cell_updates"])
                turn.append(np.mean(turnover) if turnover else np.nan)
                if s == 0:
                    refined = stats["n_refined"]
            errs = np.array(errs)
            p_str = f"{p:.3f}" if marker == "hysteresis" else f"K={int(p)}"
            print(f"{marker:12s} {target:6d} {p_str:>7s} {np.mean(cells):6.1f}±{np.std(cells):4.1f} "
                  f"{errs[:, 0].mean():.4e}±{errs[:, 0].std():.1e} {errs[:, 1].mean():9.3e} "
                  f"{errs[:, 2].mean():9.3e} {np.mean(work):13.0f} "
                  f"{np.nanmean(turn):.3f}±{np.nanstd(turn):.3f}  {refined}")
        e, u, n_steps = run_uniform(target, L, args.nu, T_FINAL, scheme=args.scheme, return_steps=True)
        tot, sh, sm = errors(e, u, x_ref, u_ref, dx_ref, shock_mask)
        print(f"{'uniform':12s} {target:6d} {'':>7s} {target:6d}      {tot:.4e}         "
              f"{sh:9.3e} {sm:9.3e} {target * n_steps:13d}")
        print(flush=True)
