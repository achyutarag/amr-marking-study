"""
Error-vs-cell-count Pareto frontier for the AMR marking-strategy project.

Runs each marking strategy (hysteresis, deterministic top-K, Gumbel
top-K) across a sweep of budget settings, plus a plain uniform grid at
several resolutions as the classical baseline, and measures L2 error
against a high-resolution reference solution at matched final cell
count. This is the actual headline comparison the research question is
about -- turnover and mesh-structure plots are supporting evidence for
*why* a method behaves the way it does, but this is the *does it
actually work* result.

Reference solution: uniform grid at n=4096 using the same conservative
solver (step_burgers_nonuniform), since that's now verified conservative
to machine precision -- no need for a separate baseline solver.
"""

import numpy as np
from amr_nonuniform import (
    uniform_edges, cell_centers, cell_widths,
    step_burgers_nonuniform, stable_dt_nonuniform,
    run_adaptive,
)


def initial_condition(x, L):
    return np.sin(2 * np.pi * x / L) + 0.5


def run_uniform(n: int, L: float, nu: float, t_final: float, cfl: float = 0.4,
                scheme: str = "rusanov1", return_steps: bool = False):
    edges = uniform_edges(n, L)
    u = initial_condition(cell_centers(edges), L)
    t, n_steps = 0.0, 0
    while t < t_final:
        dt = stable_dt_nonuniform(u, edges, nu, cfl)
        dt = min(dt, t_final - t)
        u = step_burgers_nonuniform(u, edges, dt, nu, scheme)
        t += dt
        n_steps += 1
    return (edges, u, n_steps) if return_steps else (edges, u)


def l2_error_against_reference(edges: np.ndarray, u: np.ndarray,
                                x_ref: np.ndarray, u_ref: np.ndarray, dx_ref: float) -> float:
    """
    Interpolate the (possibly non-uniform) solution onto the fine
    reference grid's cell centers and compute the L2 norm of the
    difference, scaled by the reference cell width (a discrete integral).
    """
    centers = cell_centers(edges)
    # np.interp needs an ascending x array with period; wrap manually for periodicity.
    u_on_ref = np.interp(x_ref, centers, u, period=x_ref[-1] + dx_ref)
    return np.sqrt(dx_ref * np.sum((u_on_ref - u_ref) ** 2))


if __name__ == "__main__":
    L, nu, t_final = 1.0, 0.01, 0.3
    n0 = 32
    frac_lo = 0.05
    regrid_dt = t_final / 20
    min_contrast = 3.5

    # --- Reference solution ---
    n_ref = 4096
    edges_ref, u_ref = run_uniform(n_ref, L, nu, t_final)
    x_ref = cell_centers(edges_ref)
    dx_ref = L / n_ref

    results = {"uniform": [], "hysteresis": [], "topk": [], "gumbel_topk": []}

    # --- Uniform-grid baseline sweep ---
    for n in [32, 48, 64, 96, 128, 192]:
        edges, u = run_uniform(n, L, nu, t_final)
        err = l2_error_against_reference(edges, u, x_ref, u_ref, dx_ref)
        results["uniform"].append((n, err))

    # --- Hysteresis sweep (vary frac_hi: lower = more permissive = more cells) ---
    for frac_hi in [0.6, 0.4, 0.3, 0.2, 0.1]:
        snaps, _ = run_adaptive(n0=n0, L=L, nu=nu, t_final=t_final, marker="hysteresis",
                                 marker_kwargs={"frac_hi": frac_hi}, frac_lo=frac_lo,
                                 regrid_dt=regrid_dt, min_contrast=min_contrast)
        edges, u = snaps[-1][1], snaps[-1][2]
        err = l2_error_against_reference(edges, u, x_ref, u_ref, dx_ref)
        results["hysteresis"].append((len(u), err))

    # --- Deterministic top-K sweep ---
    for K in [5, 10, 15, 25, 40]:
        snaps, _ = run_adaptive(n0=n0, L=L, nu=nu, t_final=t_final, marker="topk",
                                 marker_kwargs={"K": K}, frac_lo=frac_lo,
                                 regrid_dt=regrid_dt, min_contrast=min_contrast)
        edges, u = snaps[-1][1], snaps[-1][2]
        err = l2_error_against_reference(edges, u, x_ref, u_ref, dx_ref)
        results["topk"].append((len(u), err))

    # --- Gumbel top-K sweep (fixed temperature, vary K; multiple seeds per K) ---
    for K in [5, 10, 15, 25, 40]:
        errs, counts = [], []
        for seed in range(3):
            snaps, _ = run_adaptive(n0=n0, L=L, nu=nu, t_final=t_final, marker="gumbel_topk",
                                     marker_kwargs={"K": K, "temperature": 0.1}, frac_lo=frac_lo,
                                     regrid_dt=regrid_dt, min_contrast=min_contrast, seed=seed)
            edges, u = snaps[-1][1], snaps[-1][2]
            errs.append(l2_error_against_reference(edges, u, x_ref, u_ref, dx_ref))
            counts.append(len(u))
        results["gumbel_topk"].append((np.mean(counts), np.mean(errs), np.std(errs)))

    print(f"{'method':14s} {'cells':>8s} {'L2 error':>12s}")
    for method in ["uniform", "hysteresis", "topk"]:
        for cells, err in results[method]:
            print(f"{method:14s} {cells:8d} {err:12.6e}")
    for cells, err, std in results["gumbel_topk"]:
        print(f"{'gumbel_topk':14s} {cells:8.1f} {err:12.6e}  (std over 3 seeds: {std:.2e})")

    import pickle
    with open("/home/claude/pareto_results.pkl", "wb") as f:
        pickle.dump(results, f)
