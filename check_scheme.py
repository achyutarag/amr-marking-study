"""
Verification of the second-order option (scheme="muscl2") in amr_nonuniform.py.
Run: python check_scheme.py

  1. scheme="rusanov1" reproduces the ORIGINAL solver to round-off
     (the refactor into burgers_rhs changed nothing).
  2. Observed order of accuracy on a smooth solution (t = 0.05, before
     the front steepens), by self-convergence on uniform grids.
  3. Order on a NON-uniform (refined-patch) grid, so the non-uniform
     reconstruction is checked too.
  4. Mass conservation through adaptive runs with linear prolongation.
  5. Reference-solution error for each scheme at T = 0.3 (n=4096 vs 8192).
"""

import importlib.util
import sys
import numpy as np
import amr_nonuniform as A
from pareto_frontier import run_uniform

L, NU = 1.0, 0.01
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def l2_cellavg(edges, u, edges_f, u_f):
    """L2 difference after averaging the fine solution onto the coarse cells (nested grids)."""
    idx = np.searchsorted(edges, A.cell_centers(edges_f), side="right") - 1
    w = A.cell_widths(edges_f)
    avg = np.bincount(idx, weights=w * u_f, minlength=len(u)) / np.bincount(idx, weights=w, minlength=len(u))
    return np.sqrt(np.sum(A.cell_widths(edges) * (u - avg) ** 2))


def run_on(edges, scheme, t_final):
    u = np.sin(2 * np.pi * A.cell_centers(edges)) + 0.5
    t = 0.0
    while t < t_final:
        dt = min(A.stable_dt_nonuniform(u, edges, NU), t_final - t)
        u = A.step_burgers_nonuniform(u, edges, dt, NU, scheme)
        t += dt
    return u


# 1. rusanov1 unchanged vs the original file
spec = importlib.util.spec_from_file_location("orig", "original/amr_nonuniform.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
e = A.uniform_edges(128, L)
u_new = np.sin(2 * np.pi * A.cell_centers(e)) + 0.5
u_old = u_new.copy()
for _ in range(200):
    dt = A.stable_dt_nonuniform(u_new, e, NU)
    u_new = A.step_burgers_nonuniform(u_new, e, dt, NU, "rusanov1")
    u_old = orig.step_burgers_nonuniform(u_old, e, dt, NU)
check("rusanov1 matches original solver", np.max(np.abs(u_new - u_old)) < 1e-12,
      f"max diff={np.max(np.abs(u_new - u_old)):.1e}")

# 2./3. order of accuracy on smooth data
T_SMOOTH = 0.05
for grid in ["uniform", "non-uniform"]:
    for scheme, expected in [("rusanov1", 1.0), ("muscl2", 2.0)]:
        def make(n):
            e = A.uniform_edges(n, L)
            if grid == "non-uniform":   # refine the middle third once: a 2:1 width jump on each side
                mid = (A.cell_centers(e) > 1 / 3) & (A.cell_centers(e) < 2 / 3)
                e, _ = A.refine(e, np.zeros(n), mid)
            return e
        ef = make(4096)
        uf = run_on(ef, scheme, T_SMOOTH)
        ns = [64, 128, 256, 512]
        errs = [l2_cellavg(make(n), run_on(make(n), scheme, T_SMOOTH), ef, uf) for n in ns]
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        check(f"{scheme} order on {grid} grid ~{expected:.0f}", abs(rates[-1] - expected) < 0.35,
              "errors " + " ".join(f"{x:.2e}" for x in errs) + "  rates " + " ".join(f"{r:.2f}" for r in rates))

# 4. conservation through AMR with linear prolongation
for marker, kw in [("hysteresis", {"frac_hi": 0.2}), ("topk", {"K": 20}),
                   ("gumbel_topk", {"K": 20, "temperature": 0.1})]:
    snaps, _ = A.run_adaptive(n0=32, L=L, nu=NU, t_final=0.3, marker=marker, marker_kwargs=kw,
                              frac_lo=0.05, regrid_dt=0.015, min_contrast=3.5, scheme="muscl2")
    m = [np.sum(u * A.cell_widths(e)) for _, e, u in snaps]
    drift = max(abs(x - m[0]) for x in m)
    check(f"muscl2 + linear prolongation, {marker}: mass conserved", drift < 1e-12, f"max drift={drift:.1e}")

# 5. reference-solution error at T = 0.3
for scheme in ["rusanov1", "muscl2"]:
    e4, u4 = run_uniform(4096, L, NU, 0.3, scheme=scheme)
    e8, u8 = run_uniform(8192, L, NU, 0.3, scheme=scheme)
    print(f"INFO  {scheme}: reference error ||u_4096 - u_8192||_2 = {l2_cellavg(e4, u4, e8, u8):.2e}")

sys.exit(1 if failures else 0)
