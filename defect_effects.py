"""Measure each code defect's effect: original/ vs current code, same settings.

Run from the repo root: python defect_effects.py   (about a minute)
Output is quoted in paper_notes/defect_log.md.
"""
import importlib.util, sys
import numpy as np
import pandas as pd

repo = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, repo)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
O = load(f"{repo}/original/amr_nonuniform.py", "orig")
C = load(f"{repo}/amr_nonuniform.py", "cur")

L, n0, nu, T = 1.0, 32, 0.01, 0.3
base = L / n0
cap = base / 8

# D1: top-K realized refinements per event, original regrid vs current
for name, M, extra in [("original", O, ()), ("current", C, (np.random.default_rng(0), base))]:
    edges = M.uniform_edges(n0, L)
    u = np.tanh((M.cell_centers(edges) - 0.5) / 0.01)
    counts = []
    for _ in range(8):
        if name == "original":
            edges, u, mask = M.regrid(edges, u, "topk", 0.05, cap, {"K": 25}, None)
        else:
            edges, u, mask = M.regrid(edges, u, "topk", 0.05, cap, {"K": 25}, *extra)
        counts.append(int(mask.sum()))
    print(f"D1 top-K K=25 refinements per event [{name}]: {counts}")

# D2: coarsening below base / non-dyadic widths, original driver on the real problem
def widths_report(snaps):
    w = np.concatenate([np.diff(e) for _, e, _ in snaps])
    lv = np.log2(base / w)
    nondyadic = np.mean(np.abs(lv - np.rint(lv)) > 1e-6)
    return w.max() / base, nondyadic

for marker, kw in [("hysteresis", {"frac_hi": 0.1}), ("topk", {"K": 10}),
                   ("gumbel_topk", {"K": 10, "temperature": 0.1})]:
    so, _ = O.run_adaptive(n0, L, nu, T, marker, kw, 0.05)
    sc, _ = C.run_adaptive(n0, L, nu, T, marker, kw, 0.05)
    mo, no = widths_report(so); mc, nc = widths_report(sc)
    print(f"D2 {marker:11s} max width/base original {mo:.2f}, current {mc:.2f}; "
          f"non-dyadic cell share original {no:.1%}, current {nc:.1%}")

# D3: turnover superseded vs current, same rows
for s in ["muscl2", "rusanov1"]:
    old = pd.read_csv(f"{repo}/results/superseded/sweep_{s}_nu0.01.csv")
    new = pd.read_csv(f"{repo}/results/sweep_{s}_nu0.01.csv")
    k = ["method", "target", "seed"]
    m = old.merge(new, on=k, suffixes=("_old", "_new"))
    m = m[m.method != "uniform"]
    d = (m.turnover_mean_old - m.turnover_mean_new)
    print(f"D3 {s}: rows {len(m)}, turnover changed in {(d.abs() > 1e-9).sum()}, "
          f"old range {m.turnover_mean_old.min():.2f}-{m.turnover_mean_old.max():.2f}, "
          f"new range {m.turnover_mean_new.min():.2f}-{m.turnover_mean_new.max():.2f}, "
          f"new undefined {m.turnover_mean_new.isna().sum()}")
    by = m.groupby("method")[["turnover_mean_old", "turnover_mean_new"]].mean().round(3)
    print(by.to_string())
