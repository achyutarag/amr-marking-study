# Superseded results — do not cite

These files were produced before two corrections and are kept only so the
history of the analysis can be checked.

1. **Turnover definition.** Turnover was computed over every pair of
   consecutive regrid events, including pairs where neither event had any
   finest-level cells (distance 0 by convention) and the first appearance of
   finest-level cells (distance 1 by definition). Both biased the mean. The
   current definition (`amr_nonuniform.run_adaptive`) uses only pairs where
   both events have finest-level cells. **Every turnover value in this folder
   is affected.**
2. **Hysteresis threshold guard.** The budget search could choose a refine
   threshold below the coarsening threshold (`frac_hi < frac_lo`). This
   happened in one run: first-order scheme, hysteresis, target 256
   (`frac_hi = 0.01`). The current search requires `frac_hi >= frac_lo + 0.02`.

Accuracy and cost columns (`cells`, `l2`, `l2_shock`, `l2_smooth`,
`cell_updates`) in the CSVs here are identical to the current files, except
for the one run in item 2. This was checked row by row after the rerun.

The `bm_*_nu0.03.txt` files are the only record of the nu = 0.03 runs. Their
accuracy values are valid, but their turnover column is not. The nu = 0.03
setting was judged unsuitable as a diffusion control (the front is too wide
to be a localized feature) and is not used in the paper.
