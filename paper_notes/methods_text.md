# Paper text: methodological clarifications

Draft wording for the methods and results sections. Every number cited here
must come from a file in `results/`; the file is named next to each claim.

## Indicator ablation: what is held fixed (fix 2)

**For Section III (Methods), indicator ablation:**

> We compare two marking indicators, the gradient indicator
> $\eta_i^{\nabla} = h_i\,|\partial_x u|_i$ and the curvature indicator
> $\eta_i^{\kappa} = h_i^2\,|\partial_{xx} u|_i$, while holding the gradient-based
> contrast gate fixed. In both arms, refinement is activated only once
> $\max_i \eta_i^{\nabla} / \operatorname{median}_i \eta_i^{\nabla} \ge 3.5$; the
> indicator under study then determines which cells are refined and which are
> coarsened. The comparison therefore isolates the effect of the marking
> indicator *conditional on the same refinement-activation mechanism*; it is not
> a comparison of two complete, independently designed AMR strategies.

**Why the gate is held fixed (same section, next paragraph):**

> The gate is held fixed because refinement timing is itself a strong
> determinant of final accuracy (Section V-C): in our early-phase control,
> resolving the forming front before the gate opens recovered 61-82% of the
> error difference between the gated run and a run resolved everywhere from
> $t = 0$. A curvature-based gate would open at a different time, and the
> resulting difference in error could not be attributed to the marking
> indicator. With the gate fixed, the first regrid occurs at $t_g = 0.105$ in
> both arms.
>
> Holding the gate *rule* fixed does not make the subsequent regrid schedule
> identical. The gate is re-evaluated at every regrid time on each arm's own
> solution and mesh, and the two indicators produce different meshes, so after
> $t_g$ the gate can close at different times in the two arms. In a
> representative pair of runs (deterministic top-$K$, about 64 final cells), the
> gradient arm regrids at 8 of the 14 scheduled regrid times from $t_g$ to $T$
> (inclusive) and the curvature arm at 4. The curvature arm therefore achieves its accuracy with
> fewer regrid events, not more; we report the number of regrid events for
> every run alongside its error.

Sources: `timing_control.py`, `results/timing_control_muscl2_nu0.01.csv`
(61-82% figure), `amr_nonuniform.indicator_contrast` (gate always uses the
gradient indicator), regrid counts in `results/ablation_*_flo0.05.csv`
(`n_regrids`). The representative 8-vs-4 schedule is not stored in a CSV; it
comes from rerunning top-K (K=5 gradient, K=9 curvature, 3 levels) and listing
the snapshot times returned by `run_adaptive`.

**Limitation this creates:** a gate that, once open, stays open ("latching")
would equalize the regrid schedule. We did not run that variant.

## Computational cost of the curvature indicator (fix 3)

**For Section IV (Results), after the indicator accuracy results:**

> The accuracy gain of the curvature indicator comes with a cost that depends
> on the budget. At about 48 and 64 final cells (3 levels), curvature-based
> marking used similar or less work than gradient-based marking
> (1.4-2.7 x 10^4 vs 1.6-2.3 x 10^4 cell updates). At about 80 cells it used
> 1.8-2.7x the work of gradient marking (4.7-6.2 x 10^4 vs 2.3-2.6 x 10^4). Here
> the curvature indicator concentrates refinement so narrowly that the target
> budget exceeds what the front can absorb at the depth cap: cells refined at
> one event fall below the coarsening threshold and are merged at the next
> (228-378 refinements and 182-330 merges per run, against 56-63 and 8-9 for
> gradient marking). The time-averaged cell count (89-107) then exceeds the
> final count (75-80), so final cell count understates the cost.

> **Depth cap and global time step.** All runs advance with a single global
> time step, $\Delta t = 0.4\,\min(h_{\min}/\max|u|,\; h_{\min}^2/(2\nu))$, set by
> the smallest cell. For $\nu = 0.01$ and $\max|u| \approx 1.35$ the diffusive
> limit is the smaller one whenever $h_{\min} < 2\nu/\max|u| \approx 1/68$, so on every
> refined level $\Delta t \propto h_{\min}^2$ and each additional level multiplies the number of
> time steps by up to four. With 3 levels ($h_{\min} = 1/256$) the AMR runs took
> 300-580 steps; with 5 levels, runs whose cells reached $h_{\min} = 1/1024$ took
> 1,519-4,526 steps, against 984 for a uniform 256-cell grid. Deeper
> refinement therefore raises the cost of every cell in the mesh, not only
> the refined ones. This is why the 5-level runs lose to uniform refinement at
> matched work (Table IV) even where they win at matched cell count. Local time
> stepping (subcycling) would remove this penalty; we did not implement it.

Sources: `results/ablation_muscl2_nu0.01_{gradient,curvature}_L{3,5}_flo0.05.csv`
(columns `cell_updates`, `n_steps`, `min_width`, `refined`, `coarsened`);
time-averaged cells = `cell_updates / n_steps`; uniform 256 steps from
`results/uniform_dense_muscl2_nu0.01.csv`.

## Work-matched comparison against uniform refinement (fix 1)

**For Section IV, replacing any interpolated-only claim:**

> To compare AMR and uniform refinement at equal work, we ran uniform grids at
> every $n = 32, 36, \ldots, 320$. Each AMR configuration's mean work therefore
> lies between two actual uniform runs whose work differs by 3-7%. Table IV
> reports both neighbouring uniform runs and the uniform error at the AMR work
> under four rules: log-log linear interpolation between the neighbours,
> linear interpolation, the nearest neighbour, and a conservative rule that
> compares against the neighbour with *more* work than the AMR run. Because the
> neighbours are close, the four rules agree to within about 5%. We claim an
> AMR advantage only where it holds under the conservative rule.


**For Section VI (Limitations):**

> Because the gate is shared, our results do not establish how a fully
> curvature-based strategy (curvature-based gate and marking) would perform;
> such a strategy would change both where and when refinement occurs.

## Turnover: definition and what it supports (fix 3, continued)

**For Section III (Metrics):**

> Grid turnover is the Jaccard distance between the sets of reference points
> covered by finest-level cells at two consecutive regrid events. It is
> computed only for pairs of events in which *both* sets are non-empty: a pair
> of empty sets carries no information about mesh motion, and the first
> appearance of finest-level cells (empty to non-empty) always has distance 1
> regardless of how the mesh then behaves. A run's turnover is the mean over
> its valid pairs; we report the number of valid pairs alongside it, and treat
> turnover as undefined for runs with none.

**For Section IV (Results):**

> With the second-order scheme, threshold marking produced the most stable
> meshes at moderate and large budgets (mean turnover 0.07-0.11 at 126-218
> final cells, against 0.16-0.32 for deterministic top-$K$ and 0.22-0.28 for
> Gumbel top-$K$). Gumbel and deterministic top-$K$ show no consistent
> ordering: Gumbel is lower at about 195 cells and higher at about 130 and 218
> (seed standard deviation 0.015-0.023 at these budgets). For the curvature indicator, turnover
> is reliably measured only at about 80 cells (10-11 valid pairs per run),
> where it is higher than for the gradient indicator (0.25-0.37 vs
> 0.18-0.26). At smaller budgets curvature runs have only 1-4 valid pairs, and
> with 5 refinement levels most runs have 0-3, so we do not report turnover
> there.

Sources: `results/sweep_muscl2_nu0.01.csv` and
`results/ablation_muscl2_nu0.01_*_flo0.05.csv` (columns `turnover_mean`,
`n_turnover_pairs`). Earlier turnover values, computed with the empty-set
pairs included, are kept in `results/superseded/` for the record; they
must not be cited.
