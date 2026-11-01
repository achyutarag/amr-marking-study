"""
Invariant checks for the budget-control fixes in amr_nonuniform.py.
Run: python check_mesh.py   (prints PASS/FAIL per check; exits nonzero on failure)

  1. Top-K markers refine exactly min(K, #eligible) cells per regrid event.
  2. No cell is ever coarser than the base mesh (width <= L/n0).
  3. The mesh stays dyadic: every width is (L/n0) / 2**level and every
     cell sits on its level's lattice.
  4. Mass is conserved to round-off through refine/coarsen.
"""

import sys
import numpy as np
from amr_nonuniform import (run_adaptive, regrid, cell_widths, cell_levels, uniform_edges,
                            cell_centers)

L, n0 = 1.0, 32
base = L / n0
max_level_width = base / 8
COMMON = dict(n0=n0, L=L, nu=0.01, t_final=0.3, frac_lo=0.05, regrid_dt=0.3 / 20, min_contrast=3.5)
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def is_dyadic(edges):
    lv = cell_levels(edges, base)
    w = base / 2.0**lv
    return (np.allclose(cell_widths(edges), w, rtol=1e-9, atol=0)
            and np.allclose(edges[:-1] / w, np.rint(edges[:-1] / w), atol=1e-6))


# 1. exact-K refinement, checked event by event against the eligible count
for marker, kw in [("topk", {"K": 25}), ("gumbel_topk", {"K": 25, "temperature": 0.1})]:
    rng = np.random.default_rng(0)
    edges = uniform_edges(n0, L)
    # steepen the profile artificially so several events hit the depth cap
    u = np.tanh((cell_centers(edges) - 0.5) / 0.01)
    ok, log = True, []
    for _ in range(8):
        n_eligible = int(np.sum(cell_widths(edges) > max_level_width * (1 + 1e-9)))
        edges, u, mask = regrid(edges, u, marker, 0.05, max_level_width, kw, rng, base)
        log.append((int(mask.sum()), min(25, n_eligible)))
        ok &= mask.sum() == min(25, n_eligible)
    check(f"{marker}: refines exactly min(K, eligible) per event", ok, str(log[:4]))

for marker, kw in [("hysteresis", {"frac_hi": 0.1}), ("topk", {"K": 40}),
                   ("gumbel_topk", {"K": 40, "temperature": 0.1})]:
    stats = {}
    snaps, _ = run_adaptive(marker=marker, marker_kwargs=kw, stats=stats, **COMMON)
    # 2. width bound
    check(f"{marker}: max width <= base", stats["max_width"] <= base * (1 + 1e-9),
          f"max={stats['max_width']:.5f} base={base:.5f}")
    # 3. dyadic at every snapshot
    check(f"{marker}: mesh dyadic at every snapshot", all(is_dyadic(e) for _, e, _ in snaps))
    # 4. conservation
    m = [np.sum(u * cell_widths(e)) for _, e, u in snaps]
    drift = max(abs(x - m[0]) for x in m)
    check(f"{marker}: mass conserved", drift < 1e-12, f"max drift={drift:.1e}")
    print(f"      refined per event: {stats['n_refined']}")

sys.exit(1 if failures else 0)
