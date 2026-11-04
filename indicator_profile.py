"""
Measures how large the gradient indicator is on the smooth ramp relative to
its maximum, which explains why gradient-based AMR refines smooth regions.

    1. At the gate time t_g = 0.105, on uniform grids n = 32..256: the ratio
       eta(x = 0.2) / max(eta), and median(eta) / max(eta).
    2. At t = T, on the final mesh of the threshold-marking run matched to
       128 cells (second-order scheme): median over cells outside the shock
       window of eta / max(eta).

Writes results/indicator_profile_muscl2_nu0.01.csv. Usage: python indicator_profile.py
"""

import numpy as np
import pandas as pd
import amr_nonuniform as A
from pareto_frontier import run_uniform

NU, T_G, T = 0.01, 0.105, 0.3

if __name__ == "__main__":
    rows = []
    for n in [32, 64, 128, 256]:
        e, u = run_uniform(n, 1.0, NU, T_G, scheme="muscl2")
        eta = A.gradient_indicator_nonuniform(u, e)
        x = A.cell_centers(e)
        rows.append(dict(case=f"uniform n={n} at t_g", ramp_ratio=eta[np.argmin(np.abs(x - 0.2))] / eta.max(),
                         median_ratio=np.median(eta) / eta.max()))

    sweep = pd.read_csv("results/sweep_muscl2_nu0.01.csv")
    frac_hi = float(sweep[(sweep.method == "hysteresis") & (sweep.target == 128)].param.iloc[0])
    snaps, _ = A.run_adaptive(n0=32, L=1.0, nu=NU, t_final=T, marker="hysteresis",
                              marker_kwargs={"frac_hi": frac_hi}, frac_lo=0.05, regrid_dt=T / 20,
                              min_contrast=3.5, scheme="muscl2")
    _, e, u = snaps[-1]
    eta = A.gradient_indicator_nonuniform(u, e)
    x = A.cell_centers(e)
    smooth = np.abs(x - x[np.argmax(eta)]) >= 0.05
    rows.append(dict(case=f"threshold AMR at T ({len(u)} cells), smooth cells",
                     ramp_ratio=eta[np.argmin(np.abs(x - 0.2))] / eta.max(),
                     median_ratio=np.median(eta[smooth]) / eta.max()))
    out = pd.DataFrame(rows)
    out.to_csv("results/indicator_profile_muscl2_nu0.01.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
