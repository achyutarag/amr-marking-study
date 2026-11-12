"""
Reproduce the SERS paper's headline comparison (Table II) from scratch.

Run from the repository root:  python sers/headline.py   (a few minutes)
Writes results/sers_headline.txt.

One adaptive run (second-order scheme, curvature indicator, 5 refinement
levels, deterministic top-K with K=13) is compared with
  - the uniform grid with about the same number of cells (n=96), and
  - the two uniform grids whose work brackets the adaptive run (n=280, 284).
Error is the L2 difference from a 4096-cell uniform reference at T=0.3.
The same numbers appear in results/ablation_muscl2_nu0.01_curvature_L5_flo0.05.csv
(row: topk, target 96) and results/uniform_dense_muscl2_nu0.01.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amr_nonuniform import run_adaptive, cell_centers
from pareto_frontier import run_uniform, l2_error_against_reference

L, NU, T, N0 = 1.0, 0.01, 0.3, 32
N_REF = 4096

edges_ref, u_ref = run_uniform(N_REF, L, NU, T, scheme="muscl2")
x_ref = cell_centers(edges_ref)


def err(edges, u):
    return l2_error_against_reference(edges, u, x_ref, u_ref, L / N_REF)


rows = []
stats = {}
snaps, _ = run_adaptive(N0, L, NU, T, "topk", {"K": 13}, frac_lo=0.05, regrid_dt=T / 20,
                        max_level_width=(L / N0) / 2**5, scheme="muscl2", indicator="curvature",
                        stats=stats)
_, edges, u = snaps[-1]
rows.append(("adaptive, 5 levels", len(u), stats["n_steps"], stats["cell_updates"], err(edges, u)))
for n in [96, 280, 284]:
    e, v, steps = run_uniform(n, L, NU, T, scheme="muscl2", return_steps=True)
    rows.append((f"uniform, n={n}", n, steps, n * steps, err(e, v)))

lines = [f"{'run':20s} {'cells':>6s} {'steps':>6s} {'cell updates':>13s} {'L2 error':>10s}"]
lines += [f"{r[0]:20s} {r[1]:6d} {r[2]:6d} {r[3]:13d} {r[4]:10.3e}" for r in rows]
out = "\n".join(lines)
print(out)
os.makedirs("results", exist_ok=True)
with open("results/sers_headline.txt", "w") as f:
    f.write(out + "\n")
