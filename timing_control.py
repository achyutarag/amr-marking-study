"""
Control experiment for the paper's Section 6 claim (weakness #2).

Claim under test: removing the contrast gate helps because refinement
then follows the shock during its FORMATION, and numerical diffusion
committed on the coarse mesh during formation cannot be undone later.

Competing explanation: early refinement helps simply because it reduces
numerical diffusion in the SMOOTH regions, wherever it is spent.

Design. With the gate on, no regrid passes the contrast test before
t_g (the first gated regrid time), so the gated run is exactly
"base mesh on [0, t_g], then gated AMR on [t_g, T]". Every arm below
keeps that structure and changes ONLY the mesh used on [0, t_g]:

    A  base       base mesh everywhere (= the gated run)
    C  all        finest level (1/256) on the whole domain
    D  front      finest level on [0.375, 0.75], which contains the
                  forming front for all t < t_g (it moves 0.50 -> 0.55)
    E  away       finest level on [0.875, 1) U [0, 0.25], the SAME width
                  as D, far from the front
    B  ungated    for reference: min_contrast = 0, AMR from t = 0

At t_g, arms A/C/D/E are conservatively averaged back onto the base
mesh, and all of them then run the identical gated AMR (same marker,
same K or frac_hi, same regrid schedule) to T. So any difference in
final error comes only from the solution state handed over at t_g,
i.e. from where resolution was spent during formation.

Reading the outcome:
    D ~ C  and  E ~ A   -> resolving the forming front is what matters
                           (supports the paper's mechanism)
    E ~ C  or  D ~ E    -> smooth-region diffusion matters at least as
                           much (paper's mechanism not isolated)
The error at t_g (restricted to the base mesh) is also reported, so it
is visible how much error each arm carries into the AMR phase.

Usage: python timing_control.py [--scheme muscl2] [--nu 0.01]
"""

import argparse
import csv
import numpy as np
import amr_nonuniform as A
from pareto_frontier import run_uniform

L, T_FINAL, N0, LEVELS = 1.0, 0.3, 32, 3
BASE = L / N0
REGRID_DT = T_FINAL / 20
FRAC_LO, MIN_CONTRAST = 0.05, 3.5
SHOCK_HALF_WIDTH = 0.05
REGIONS = {
    "A base": lambda x: np.zeros_like(x, dtype=bool),
    "C all": lambda x: np.ones_like(x, dtype=bool),
    "D front": lambda x: (x > 0.375) & (x < 0.75),
    "E away": lambda x: (x > 0.875) | (x < 0.25),
}


def ic(x):
    return np.sin(2 * np.pi * x / L) + 0.5


def early_mesh(region):
    """Base mesh with every cell whose center lies in `region` refined LEVELS times."""
    edges = A.uniform_edges(N0, L)
    for _ in range(LEVELS):
        edges, _ = A.refine(edges, np.zeros(len(edges) - 1), region(A.cell_centers(edges)))
    return edges


def restrict_to_base(edges, u):
    """Conservative average of a dyadic-refined solution onto the base mesh."""
    idx = np.minimum((A.cell_centers(edges) / BASE).astype(int), N0 - 1)
    w = A.cell_widths(edges)
    return A.uniform_edges(N0, L), np.bincount(idx, weights=w * u, minlength=N0) / BASE


def evolve(edges, u, t, t_end, scheme, nu, regrid=None, stats=None):
    """
    Advance to t_end. With `regrid` = (marker, kw, rng, min_contrast), a
    regrid is attempted at every multiple of REGRID_DT, exactly as in
    run_adaptive; without it the mesh is static. If t itself is a regrid
    time (the hand-over at t_g), that regrid is attempted first, since
    in the gated run it is the event at which the gate first opens.
    """
    next_regrid = (np.floor(t / REGRID_DT + 1e-9) + 1) * REGRID_DT
    if regrid is not None and t > 0 and abs(t / REGRID_DT - round(t / REGRID_DT)) < 1e-9:
        marker, kw, rng, min_contrast = regrid
        if A.indicator_contrast(u, edges) >= min_contrast:
            edges, u, _ = A.regrid(edges, u, marker, FRAC_LO, BASE / 2**LEVELS, kw, rng, BASE,
                                   "linear" if scheme == "muscl2" else "constant")
    while t < t_end - 1e-14:
        dt = A.stable_dt_nonuniform(u, edges, nu)
        dt = min(dt, t_end - t, next_regrid - t if next_regrid > t else dt)
        u = A.step_burgers_nonuniform(u, edges, dt, nu, scheme)
        t += dt
        if stats is not None:
            stats["cell_updates"] += len(u)
        if t >= next_regrid - 1e-12:
            next_regrid += REGRID_DT
            if regrid is not None:
                marker, kw, rng, min_contrast = regrid
                if A.indicator_contrast(u, edges) >= min_contrast:
                    edges, u, _ = A.regrid(edges, u, marker, FRAC_LO, BASE / 2**LEVELS, kw, rng, BASE,
                                           "linear" if scheme == "muscl2" else "constant")
    return edges, u, t


def errors(edges, u, x_ref, u_ref, mask):
    sq = (np.interp(x_ref, A.cell_centers(edges), u, period=L) - u_ref) ** 2
    dx = x_ref[1] - x_ref[0]
    return tuple(np.sqrt(dx * np.sum(sq * m)) for m in (1.0, mask, 1.0 - mask))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="muscl2", choices=["rusanov1", "muscl2"])
    ap.add_argument("--nu", type=float, default=0.01)
    args = ap.parse_args()
    nu, scheme = args.nu, args.scheme
    out = []  # one row per (marker, arm, seed), written to results/ at the end

    # gate time: first time the contrast test passes on the base mesh
    e0 = A.uniform_edges(N0, L)
    u0 = ic(A.cell_centers(e0))
    t_g = 0.0
    while True:
        e0, u0, t_g = evolve(e0, u0, t_g, t_g + REGRID_DT, scheme, nu)
        if A.indicator_contrast(u0, e0) >= MIN_CONTRAST:
            break
    t_g = round(t_g / REGRID_DT) * REGRID_DT

    # reference at t_g (restricted to base, for the hand-over error) and at T
    er, ur = run_uniform(4096, L, nu, t_g, scheme=scheme)
    _, ur_g_base = restrict_to_base(er, ur)
    er, ur = run_uniform(4096, L, nu, T_FINAL, scheme=scheme)
    xr = A.cell_centers(er)
    xs = xr[np.argmax(np.abs(np.gradient(ur)))]
    mask = (np.abs((xr - xs + 0.5) % 1.0 - 0.5) < SHOCK_HALF_WIDTH).astype(float)

    print(f"scheme={scheme} nu={nu}  gate opens at t_g={t_g:.3f}  final front at x={xs:.4f}")
    print(f"{'marker':22s} {'arm':10s} {'early cells':>11s} {'err@t_g':>9s} {'final cells':>11s} "
          f"{'L2 error':>21s} {'shock':>9s} {'smooth':>9s} {'cell updates':>12s}")
    markers = [("topk", {"K": 11}, 1), ("hysteresis", {"frac_hi": 0.24}, 1),
               ("gumbel_topk", {"K": 9, "temperature": 0.1}, 10)]
    for marker, kw, n_seeds in markers:
        name = marker + " " + ",".join(f"{k}={v}" for k, v in kw.items() if k != "temperature")
        for arm, region in REGIONS.items():
            rows = []
            for seed in range(n_seeds):
                stats = {"cell_updates": 0}
                e = early_mesh(region)
                u = ic(A.cell_centers(e))
                n_early = len(u)
                e, u, t = evolve(e, u, 0.0, t_g, scheme, nu, stats=stats)
                e, u = restrict_to_base(e, u)
                err_g = np.sqrt(BASE * np.sum((u - ur_g_base) ** 2))
                e, u, t = evolve(e, u, t, T_FINAL, scheme, nu,
                                 (marker, kw, np.random.default_rng(seed), MIN_CONTRAST), stats)
                rows.append((err_g, len(u), *errors(e, u, xr, ur, mask), stats["cell_updates"]))
                out.append(dict(marker=marker, arm=arm, seed=seed, early_cells=n_early, err_tg=err_g,
                                cells=len(u), l2=rows[-1][2], l2_shock=rows[-1][3], l2_smooth=rows[-1][4],
                                cell_updates=stats["cell_updates"]))
            r = np.array(rows)
            print(f"{name:22s} {arm:10s} {n_early:11d} {r[:, 0].mean():9.3e} {r[:, 1].mean():11.1f} "
                  f"{r[:, 2].mean():.4e}±{r[:, 2].std():.1e} {r[:, 3].mean():9.3e} {r[:, 4].mean():9.3e} "
                  f"{r[:, 5].mean():12.0f}")
        # B: ungated, through the library entry point
        rows = []
        for seed in range(n_seeds):
            st = {}
            snaps, _ = A.run_adaptive(n0=N0, L=L, nu=nu, t_final=T_FINAL, marker=marker, marker_kwargs=kw,
                                      frac_lo=FRAC_LO, regrid_dt=REGRID_DT, min_contrast=0.0, seed=seed,
                                      stats=st, scheme=scheme)
            e, u = snaps[-1][1], snaps[-1][2]
            rows.append((len(u), *errors(e, u, xr, ur, mask), st["cell_updates"]))
            out.append(dict(marker=marker, arm="B ungated", seed=seed, early_cells="", err_tg="",
                            cells=len(u), l2=rows[-1][1], l2_shock=rows[-1][2], l2_smooth=rows[-1][3],
                            cell_updates=st["cell_updates"]))
        r = np.array(rows)
        print(f"{name:22s} {'B ungated':10s} {'-':>11s} {'-':>9s} {r[:, 0].mean():11.1f} "
              f"{r[:, 1].mean():.4e}±{r[:, 1].std():.1e} {r[:, 2].mean():9.3e} {r[:, 3].mean():9.3e} "
              f"{r[:, 4].mean():12.0f}")
        print(flush=True)

    with open(f"results/timing_control_{scheme}_nu{nu}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
