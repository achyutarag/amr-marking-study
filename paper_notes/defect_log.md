# Defect and confound log (for the SERS paper)

Every entry names its evidence. "Measured" numbers come from
`python defect_effects.py` (original/ vs current code, same settings) or from
the result files named. Entries marked **[session only]** have no file
record; they come from the working session and must be confirmed or dropped.

Taxonomy: **D** implementation defect, **M** metric-definition defect,
**C** experimental-design confound, **T** tooling error.

## Summary table

| ID | Type | What was wrong | Output looked plausible? | Detected by | Effect | Changed a conclusion? |
|---|---|---|---|---|---|---|
| D1 | D | Top-K picked the K best cells of *all* cells, then dropped ones already at the depth cap | Yes | Invariant: refinements per event = min(K, eligible) | K=25 test: 25,25,25,21,20,21,21,20 refinements (current: 25 every event) | Budget parameter K did not control the budget |
| D2 | D | Coarsening merged any adjacent low-score pair, not sibling pairs | Yes | Invariants: width ≤ base, mesh dyadic | Hysteresis run: largest cell 1.5× the base cell; 10.6% of cells non-dyadic (top-K 2.3%, Gumbel 1.2%); current 1.00× and 0% | Coarse regions could be *less* resolved than a uniform 32-cell grid |
| M1 | M | Turnover averaged over all event pairs, including empty→empty (0) and first appearance (1) | Yes | Reasoning about the definition, then rerun | Changed in all 154 adaptive rows per scheme; second-order mean, threshold 0.21→0.15, top-K 0.29→0.23, Gumbel 0.30→0.25 | Accuracy unchanged; turnover levels and some orderings changed |
| M2 | D | Budget search could set refine threshold below coarsen threshold | No output error visible | Parameter audit | One run (first order, threshold, target 256, frac_hi 0.01) | No; one configuration |
| C1 | C | First-order scheme's numerical viscosity ≈ 0.02 on base cells vs physical ν = 0.01 | Yes | Analysis, then second-order rerun | Scheme factor 5.7–7.0× at matched budgets; first-order AMR lost to uniform in 21/21 work-matched configs | Yes: hid marker effects |
| C2 | C | Markers compared at different final cell counts | Yes | Budget matching | Residual mismatch remains (target 128: Gumbel 112 vs threshold 126 cells) | Yes: raw comparisons favored whoever used more cells |
| C3 | C | Cost measured as final cell count; global Δt ∝ h_min² makes work depend on the finest cell | Yes | Work tracking + dense uniform sweep | Curvature used 1.8–2.7× gradient's work at ~80 cells; 5-level AMR lost on work (ratios 0.21–0.83) | Yes: 5-level wins at matched cells became losses at matched work |
| C4 | C | Refinement gate opened after the front began forming (t_g = 0.105) | Yes | Timing control experiment | Early resolution recovered 61–82% of the error gap | Yes: timing > marker |
| C5 | C | Gradient indicator spends fine cells on smooth steep ramps | Yes | Indicator profiling | 103 of 126 refined cells in smooth regions at ~128 cells | Yes: curvature indicator −12–46% error |
| T1 | T | Syntax error in `coarsen_ablation.py` | No (script failed loudly) | Run failed | Nothing ran, nothing overwritten | No **[session only]** |
| T2 | T | Output filename collision between ablation runs | Would have silently overwritten CSVs | Noticed before rerun | Fixed before any overwrite; filenames now unique | No **[session only]** |

## Evidence per entry

- **D1.** Original: `original/amr_nonuniform.py`, `mark_deterministic_topk` and the
  eligibility mask applied after selection in `regrid`. Fix: `mark_deterministic_topk`
  / `mark_gumbel_topk` take `eligible`. Check: `check_mesh.py` check 1. The
  docstring's example "25, 25, 25, 6, 7" is the author's note from a real run and
  is **not** reproduced here; use the measured 21/20 numbers from the synthetic test.
- **D2.** Original `regrid` merges `coarsen_candidate & roll(coarsen_candidate)`.
  Fix: `& sibling_pairs(...)`. Checks: `check_mesh.py` checks 2–3. Measured on
  the real problem (ν=0.01, T=0.3, 32 base cells, 3 levels; hysteresis
  frac_hi 0.1, top-K/Gumbel K=10). Accuracy impact of D2 alone was **not
  measured** separately.
- **M1.** `results/superseded/README.md` item 1 and `run_adaptive` comment;
  numbers from `results/superseded/sweep_*` vs `results/sweep_*`.
- **M2.** `results/superseded/README.md` item 2; guard in `budget_match.py`.
- **C1.** `burgers_rhs` docstring (0.5 · 1.25 · 1/32 ≈ 0.02); `number_audit.md`
  rows "Scheme factor" and "First-order, work-matched".
- **C2.** `number_audit.md` row "Budget mismatch at target 128".
- **C3.** `number_audit.md` rows on curvature work, five-level ratios; `stable_dt_nonuniform`.
- **C4.** `results/timing_control_muscl2_nu0.01.csv`; `number_audit.md`.
- **C5.** `number_audit.md` row "Smooth cells refined at ~128 cells".

## Earlier fixes by the original author (not ours)

Comments in `original/amr_nonuniform.py` describe defects fixed before we
received the code. We did not observe or verify them, so cite them only as
the author's account:
- Gumbel scores not normalized, so temperature had no fixed meaning (caught
  by comparing regrid frequency across markers).
- A non-conservative upwind discretization replaced by flux form.
- Regrid cadence by step count (would give finer runs more regrids) replaced
  by fixed simulation-time intervals.
- A global Lax-Friedrichs coefficient replaced by a per-face one.

## Still to fill in (you)

- **AI involvement per entry:** who proposed each check, what the assistant
  got wrong, what you verified by hand. T1 and T2 were errors made during
  AI-assisted editing; say so if accurate.
- **Order of discovery:** git history starts after these fixes, so the paper
  should not claim a timeline it cannot show.
- Confirm or drop T1, T2.
