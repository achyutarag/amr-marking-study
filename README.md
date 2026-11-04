# AMR marking strategies for viscous Burgers 

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

## The key ideas in code

Short, simplified excerpts so you don't have to read the whole solver. Each
names the function it comes from; the real versions handle uneven
(non-uniform) cells, periodic boundaries and edge cases.

**Three ways to pick cells to split** (`mark_hysteresis`,
`mark_deterministic_topk`, `mark_gumbel_topk` in `amr_nonuniform.py`).
Gumbel top-K is ordinary top-K after adding random Gumbel noise to the
scores; the temperature `T` sets how random it is.

```python
# Threshold: above a fraction of the max
split = score > frac * score.max()

# Top-K: the K highest scores
split = top_k(score, K)

# Random (Gumbel) top-K: add noise first
g = -np.log(-np.log(rng.uniform(size=n)))
split = top_k(score / T + g, K)
```

**Two scores: steepness vs. bend** (`gradient_indicator_nonuniform`,
`curvature_indicator_nonuniform`). Both are finite-difference derivatives.
On a straight ramp the second difference is zero, so the bend score does not
waste small cells there. Shown for equal cells of width `h`.

```python
# Steepness: h * |u'|
steep = h * abs((u[i+1] - u[i-1]) / (2*h))

# Bend: h^2 * |u''|
bend = h**2 * abs((u[i+1] - 2*u[i] + u[i-1]) / h**2)
```

**The more accurate method** (`minmod`, `limited_slopes`). The first-order
scheme treats each cell as flat; the second-order scheme fits a slope in each
cell, taking the smaller of the two neighbouring slopes, or zero at a peak, so
no new wiggles appear.

```python
def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.0)

slope = minmod(right_diff, left_diff)
```

**When refinement starts** (`indicator_contrast`, used in `run_adaptive`).
Regridding waits until the steepest cell stands out from the typical cell.
This gate always uses the steepness score, whichever score drives marking, so
indicator comparisons share the same start time.

```python
s = gradient_indicator(u)
if s.max() / np.median(s) >= 3.5:
    regrid()
```

**Why cell count is not cost** (`stable_dt_nonuniform`). There is one global
time step, set by the smallest cell. Once cells are small the diffusion limit
`h**2` dominates, so halving the smallest cell can quarter the step for every
cell. Results therefore compare methods at equal *work* (total cell updates).

```python
h = widths.min()
dt = 0.4 * min(h / abs(u).max(), h**2 / (2 * nu))
```

**Checks that must always hold** (`check_mesh.py`, simplified). These caught
bugs that produced plausible-looking numbers.

```python
assert n_split == min(K, n_eligible)    # top-K really splits K cells
assert widths.max() <= base_width       # never coarser than the start
assert abs(mass - mass0) < 1e-12        # conservation to round-off
```

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
