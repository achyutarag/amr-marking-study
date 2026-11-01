# AMR marking strategies for viscous Burgers (ICAMCDS 2026)

Code and results for a controlled comparison of AMR marking strategies
(threshold, deterministic top-K, Gumbel top-K) on the 1D viscous Burgers
equation, and of the factors that confound such comparisons: scheme order,
refinement timing, the indicator, and the cost measure.

`original/` holds the three scripts exactly as received; everything else
builds on copies of them.

## Code

| File | Role |
|---|---|
| `amr_nonuniform.py` | Solver (`rusanov1`, `muscl2`), indicators (`gradient`, `curvature`), markers, refine/coarsen, `run_adaptive` |
| `pareto_frontier.py` | Uniform-grid runner and L2 error helper |
| `budget_match.py` | Tunes each marker's budget parameter to a target final cell count |
| `run_experiments.py` | Main budget-matched sweep -> `results/sweep_<scheme>_nu0.01.csv` |
| `timing_control.py` | Early-phase control (weakness #2) -> `results/timing_control_*` |
| `coarsen_ablation.py` | Coarsening-threshold and indicator ablations -> `results/ablation_*.csv` |
| `work_matched.py` | Dense uniform sweep and bracketed work-matched comparison -> `results/uniform_dense_*`, `results/work_matched_*` |
| `make_figures.py` | All figures (`figures/`) and LaTeX tables (`results/tables.tex`), drawn from the CSVs |
| `check_mesh.py`, `check_scheme.py` | Invariant and convergence checks; both must print only PASS |

## Reproducing

Requires Python 3, NumPy, pandas, matplotlib. From this folder:

```
python check_mesh.py
python check_scheme.py
python run_experiments.py --scheme muscl2
python run_experiments.py --scheme rusanov1
python timing_control.py --scheme muscl2
python coarsen_ablation.py --frac_lo 0.05 0.10 0.15 0.20
python coarsen_ablation.py --frac_lo 0.05 --indicator gradient --targets 48 64 80
python coarsen_ablation.py --frac_lo 0.05 --indicator curvature --targets 48 64 80
python coarsen_ablation.py --frac_lo 0.05 --indicator gradient --levels 5 --targets 64 96 128
python coarsen_ablation.py --frac_lo 0.05 --indicator curvature --levels 5 --targets 64 96 128
python work_matched.py --uniform
python work_matched.py
python make_figures.py
```

All runs are deterministic (Gumbel top-K uses seeds 0..N-1). `check_scheme.py`
and the sweeps take minutes to tens of minutes each; they are independent and
can run in parallel.

`results/superseded/` holds outputs from before two corrections; see its README.
`paper_notes/methods_text.md` holds draft paper text, with the source file for
every number.
