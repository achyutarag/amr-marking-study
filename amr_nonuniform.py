"""
Non-uniform grid harness for the AMR marking-strategy project.

Extends the uniform-grid Burgers baseline to an actual adaptive mesh:
cells can have different widths, and a MARK step (for now: hysteresis)
drives real REFINE/COARSEN operations between timesteps.

Grid representation:
    A 1D mesh is stored as `edges`, a sorted array of n+1 breakpoints
    over [0, L]. This gives n cells; cell i spans [edges[i], edges[i+1]].
    Cells can have arbitrary (non-power-of-2) widths -- this is a plain
    non-uniform grid, not a dyadic tree. That's a deliberate simplification:
    a full 2:1-balanced tree is the "proper" AMR data structure, but it's
    more infrastructure than a first working version needs. Swap it in
    later once the marking-strategy comparison itself is validated.

Solver on a non-uniform grid:
    - Advective term: non-uniform one-sided (upwind) differences.
    - Diffusive term: the standard 3-point non-uniform second-derivative
      formula (reduces to the usual central-difference formula when
      spacing is uniform).
    - Periodic boundary conditions throughout.

Mesh operations:
    - refine(edges, u, marked_indices): splits each marked cell in half.
      Solution value is duplicated to both children (piecewise-constant
      injection -- simple and conservative for a first version).
    - coarsen(edges, u, marked_pairs): merges each marked adjacent pair
      back into one cell. Solution value is the width-weighted average
      (conservative: preserves the integral of u over the merged region).

Marking (this file wires up ONE strategy -- hysteresis -- end to end as
a working demonstration; deterministic top-K and Gumbel-top-K plug into
the exact same regrid() call and are implemented in mark_strategies.py).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def uniform_edges(n_cells: int, L: float) -> np.ndarray:
    return np.linspace(0.0, L, n_cells + 1)


def cell_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def cell_widths(edges: np.ndarray) -> np.ndarray:
    return np.diff(edges)


# ---------------------------------------------------------------------------
# Non-uniform finite-difference solver step
# ---------------------------------------------------------------------------

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def limited_slopes(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Minmod-limited slope per cell on a non-uniform periodic grid: the
    smaller (in magnitude) of the one-sided center-to-center differences,
    or zero at a local extremum. Used both by the MUSCL reconstruction
    and by linear prolongation in refine().
    """
    dx = cell_widths(edges)
    h_right = 0.5 * (dx + np.roll(dx, -1))
    h_left = 0.5 * (dx + np.roll(dx, 1))
    return minmod((np.roll(u, -1) - u) / h_right, (u - np.roll(u, 1)) / h_left)


def burgers_rhs(u: np.ndarray, edges: np.ndarray, nu: float, scheme: str = "rusanov1") -> np.ndarray:
    """
    Semi-discrete right-hand side du/dt of the conservative finite-volume
    scheme (flux differences divided by cell width).

    scheme = "rusanov1": piecewise-constant face states (first order).
        The Rusanov flux then adds numerical viscosity of roughly
        alpha * dx / 2 -- about 0.02 on the 1/32 base cells, i.e. ~2x the
        physical nu = 0.01. That is large enough that coarse regions
        effectively solve Burgers with a different viscosity, which
        confounds any comparison of marking strategies.
    scheme = "muscl2": minmod-limited piecewise-linear reconstruction of
        the face states (MUSCL), same Rusanov flux on the reconstructed
        states. Numerical viscosity is O(dx^2) in smooth regions and the
        limiter keeps it TVD at the shock. Conservation is unchanged:
        still one flux per face, telescoping.
    """
    dx = cell_widths(edges)
    u_right = np.roll(u, -1)
    dx_right = np.roll(dx, -1)
    h_face = 0.5 * (dx + dx_right)  # distance between cell i and i+1 centers

    if scheme == "rusanov1":
        uL, uR = u, u_right
    elif scheme == "muscl2":
        s = limited_slopes(u, edges)
        uL = u + 0.5 * dx * s                         # left state at face i+1/2
        uR = u_right - 0.5 * dx_right * np.roll(s, -1)  # right state at face i+1/2
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    # Advective numerical flux at the face between cell i and cell i+1.
    # LOCAL Lax-Friedrichs: viscosity coefficient is computed PER FACE from
    # just the two neighboring face states, not a single global max over
    # the whole domain. This matters a lot for AMR specifically: with a
    # global alpha, numerical diffusion at a face is set by the fastest
    # wave speed ANYWHERE in the domain, so refining a cell locally doesn't
    # actually reduce the numerical diffusion smearing the shock there.
    alpha = np.maximum(np.abs(uL), np.abs(uR)) + 1e-12
    F_adv = 0.25 * (uL**2 + uR**2) - 0.5 * alpha * (uR - uL)

    # Diffusive numerical flux at the same face.
    F_diff = nu * (u_right - u) / h_face

    F = F_adv - F_diff
    return -(F - np.roll(F, 1)) / dx


def step_burgers_nonuniform(u: np.ndarray, edges: np.ndarray, dt: float, nu: float,
                            scheme: str = "rusanov1") -> np.ndarray:
    """
    Advance u by one explicit timestep on a non-uniform periodic grid.
    "rusanov1" uses forward Euler (the original scheme, unchanged);
    "muscl2" uses the two-stage SSP-RK2 (Heun) method, which keeps the
    TVD property of the spatial scheme under the same CFL limit.

    Both are CONSERVATIVE finite-volume schemes:
        - Advective flux: local Lax-Friedrichs (Rusanov) numerical flux for
          f(u) = u^2/2. This is conservative by construction on ANY grid
          (uniform or not) because it's a telescoping sum of edge fluxes
          -- each interior flux is added to one cell and subtracted from
          its neighbor, so nothing is created or destroyed except at
          domain boundaries (periodic here, so nothing there either).
        - Diffusive flux: nu * (u_{i+1} - u_i) / h_face, also applied in
          flux form so it's conservative too.
    This replaces the earlier non-conservative upwind scheme, which was
    discretizing u*u_x directly -- stable, but not guaranteed to
    conserve mass or converge to the correct shock speed, since Burgers'
    equation is a conservation law and needs a flux-difference scheme to
    respect that structure (this is a standard requirement, not specific
    to this project -- see e.g. LeVeque, "Finite Volume Methods for
    Hyperbolic Problems").
    """
    if scheme == "rusanov1":
        return u + dt * burgers_rhs(u, edges, nu, scheme)
    u1 = u + dt * burgers_rhs(u, edges, nu, scheme)
    return 0.5 * u + 0.5 * (u1 + dt * burgers_rhs(u1, edges, nu, scheme))


def stable_dt_nonuniform(u: np.ndarray, edges: np.ndarray, nu: float, cfl: float = 0.4) -> float:
    dx = cell_widths(edges)
    dx_min = dx.min()
    u_max = np.max(np.abs(u)) + 1e-12
    dt_advect = dx_min / u_max
    dt_diffuse = dx_min**2 / (2 * nu) if nu > 0 else np.inf
    return cfl * min(dt_advect, dt_diffuse)


def gradient_indicator_nonuniform(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    dx = cell_widths(edges)
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    h_left = np.roll(dx, 1)
    h_right = dx
    slope = (u_right - u_left) / (h_left + h_right)
    return dx * np.abs(slope)


def curvature_indicator_nonuniform(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Per-cell indicator h_i^2 |u_xx|_i, with the standard 3-point second
    derivative on a non-uniform grid (center-to-center spacings).

    Why this exists: for a second-order scheme the local error scales
    with curvature, not slope. h |u_x| is large on a straight ramp, where
    a piecewise-linear reconstruction is essentially exact, so it steers
    refinement into smooth regions (see coarsen_ablation.py). h^2 |u_xx|
    vanishes on a linear profile and is largest on the shoulders of a
    steep front.
    """
    dx = cell_widths(edges)
    h_right = 0.5 * (dx + np.roll(dx, -1))
    h_left = 0.5 * (dx + np.roll(dx, 1))
    u_xx = 2.0 * ((np.roll(u, -1) - u) / h_right - (u - np.roll(u, 1)) / h_left) / (h_left + h_right)
    return dx**2 * np.abs(u_xx)


INDICATORS = {"gradient": gradient_indicator_nonuniform, "curvature": curvature_indicator_nonuniform}


# ---------------------------------------------------------------------------
# Mesh operations: refine / coarsen
# ---------------------------------------------------------------------------

def refine(edges: np.ndarray, u: np.ndarray, marked: np.ndarray, prolong: str = "constant"):
    """
    Split each marked cell into two equal halves. `marked` is a boolean
    array, one entry per current cell.
    prolong = "constant": the value is duplicated to both children
        (piecewise-constant injection, first order).
    prolong = "linear": children take u_i -/+ s_i * dx_i / 4, where s_i is
        the minmod-limited slope. Still exactly conservative (the two
        children average to u_i) and introduces no new extrema; it is
        the prolongation consistent with a second-order scheme, so a
        freshly refined cell does not start from a staircase.
    """
    if prolong == "linear":
        offset = 0.25 * cell_widths(edges) * limited_slopes(u, edges)
    elif prolong == "constant":
        offset = np.zeros_like(u)
    else:
        raise ValueError(f"unknown prolongation: {prolong}")
    new_edges = [edges[0]]
    new_u = []
    for i in range(len(u)):
        left, right = edges[i], edges[i + 1]
        if marked[i]:
            mid = 0.5 * (left + right)
            new_edges.append(mid)
            new_edges.append(right)
            new_u.append(u[i] - offset[i])
            new_u.append(u[i] + offset[i])
        else:
            new_edges.append(right)
            new_u.append(u[i])
    return np.array(new_edges), np.array(new_u)


def coarsen(edges: np.ndarray, u: np.ndarray, marked_pairs: np.ndarray):
    """
    Merge each marked adjacent pair of cells back into one. `marked_pairs`
    is a boolean array, one entry per current cell, where True at index i
    means "merge cell i with cell i+1". Pairs are processed greedily and
    non-overlapping (if i is marked, i+1 is skipped as a merge start).
    Solution value on the merged cell is the width-weighted average
    (conservative: preserves integral of u).
    """
    n = len(u)
    dx = cell_widths(edges)
    new_edges = [edges[0]]
    new_u = []
    i = 0
    while i < n:
        if marked_pairs[i] and i + 1 < n:
            left, right = edges[i], edges[i + 2]
            w1, w2 = dx[i], dx[i + 1]
            merged_val = (w1 * u[i] + w2 * u[i + 1]) / (w1 + w2)
            new_edges.append(right)
            new_u.append(merged_val)
            i += 2
        else:
            new_edges.append(edges[i + 1])
            new_u.append(u[i])
            i += 1
    return np.array(new_edges), np.array(new_u)


# ---------------------------------------------------------------------------
# Marking: hysteresis (wired up end-to-end here as the working demonstration)
# ---------------------------------------------------------------------------

def mark_hysteresis(indicator: np.ndarray, frac_hi: float, frac_lo: float) -> np.ndarray:
    """
    Returns a boolean array: True where the cell should be refined.
    Thresholds are RELATIVE to the indicator's current max, not fixed
    absolute numbers -- a fixed absolute threshold has no consistent
    meaning across problems, resolutions, or even across time within one
    run, since the indicator's natural scale depends on the solution's
    own slope magnitude at the current grid resolution.
    """
    tau_hi = frac_hi * indicator.max()
    return indicator > tau_hi


def unmark_hysteresis(indicator: np.ndarray, frac_lo: float) -> np.ndarray:
    """Returns a boolean array: True where the cell is a coarsen candidate."""
    tau_lo = frac_lo * indicator.max()
    return indicator < tau_lo


def mark_deterministic_topk(indicator: np.ndarray, K: int, eligible: np.ndarray = None) -> np.ndarray:
    """
    Refine exactly the K ELIGIBLE cells with the largest indicator values.
    Selection is restricted to eligible cells (those not already at the
    depth cap) BEFORE taking the top K. The earlier version selected the
    top K over all cells and masked ineligible ones afterwards, so once
    the shock cells hit the cap the realized number of refinements per
    event collapsed well below K (e.g. 25, 25, 25, 6, 7, ... for K=25),
    and K no longer controlled the budget.
    """
    if eligible is None:
        eligible = np.ones_like(indicator, dtype=bool)
    idx = np.flatnonzero(eligible)
    K = min(K, len(idx))
    mask = np.zeros_like(indicator, dtype=bool)
    if K > 0:
        mask[idx[np.argpartition(indicator[idx], -K)[-K:]]] = True
    return mask


def mark_gumbel_topk(indicator: np.ndarray, K: int, temperature: float, rng: np.random.Generator,
                     eligible: np.ndarray = None) -> np.ndarray:
    """
    Refine exactly K cells, sampled without replacement with probability
    proportional to softmax(indicator / temperature), via the Gumbel-top-K
    trick: add i.i.d. Gumbel(0,1) noise to the scaled scores and take the
    top K by the PERTURBED score.

    IMPORTANT: the indicator is normalized by its own max BEFORE dividing
    by temperature. Without this, temperature has no consistent meaning:
    Gumbel(0,1) noise has a fixed spread (std ~1.28) regardless of the
    problem, so if the raw indicator's magnitude happens to be small
    (as it often is), even a "low" temperature fails to make the noise
    negligible relative to the scores, and the selection ends up close
    to uniformly random instead of close to deterministic top-K. This
    was a live bug in an earlier version of this function -- caught by
    comparing regrid frequency against the other two markers, which
    should NOT have diverged this much if temperature=0.1 were truly
    behaving near-deterministically.

    As in mark_deterministic_topk, sampling is restricted to eligible
    cells so that exactly min(K, #eligible) cells are refined. Noise is
    drawn for every cell (not just eligible ones) so the random stream
    consumed per event does not depend on the mesh state.
    """
    if eligible is None:
        eligible = np.ones_like(indicator, dtype=bool)
    eps = 1e-12
    normalized = indicator / (indicator.max() + eps)
    scores = normalized / (temperature + eps)
    u = rng.uniform(size=len(indicator))
    gumbel_noise = -np.log(-np.log(u + eps) + eps)
    perturbed = np.where(eligible, scores + gumbel_noise, -np.inf)
    idx = np.flatnonzero(eligible)
    K = min(K, len(idx))
    mask = np.zeros_like(indicator, dtype=bool)
    if K > 0:
        mask[idx[np.argpartition(perturbed[idx], -K)[-K:]]] = True
    return mask


def cell_levels(edges: np.ndarray, base_width: float) -> np.ndarray:
    """Refinement level of each cell: width = base_width / 2**level."""
    return np.rint(np.log2(base_width / cell_widths(edges))).astype(int)


def sibling_pairs(edges: np.ndarray, base_width: float) -> np.ndarray:
    """
    Boolean array, one entry per cell: True at i if cells i and i+1 are
    the two children of a single parent (same level >= 1, and cell i is
    the LEFT child, i.e. its index on its own level's uniform lattice is
    even). Because the mesh only ever splits cells in half and merges
    siblings back, every cell lies on this dyadic lattice, so sibling
    relationships can be recovered from the edges alone without storing
    a tree.
    """
    level = cell_levels(edges, base_width)
    width = base_width / 2.0**level
    lattice_idx = np.rint(edges[:-1] / width).astype(int)
    same_level = np.zeros(len(level), dtype=bool)
    same_level[:-1] = level[:-1] == level[1:]
    return same_level & (level >= 1) & (lattice_idx % 2 == 0)


def regrid(edges: np.ndarray, u: np.ndarray, marker: str, frac_lo: float,
           max_level_width: float, marker_kwargs: dict = None, rng: np.random.Generator = None,
           base_width: float = None, prolong: str = "constant", indicator: str = "gradient"):
    """
    One MARK + REFINE/COARSEN pass. `indicator` ("gradient" or "curvature",
    see INDICATORS) is the error estimate used for both refinement and
    coarsening; `marker` selects the refine-selection
    rule ('hysteresis', 'topk', or 'gumbel_topk') -- all three share the
    exact same coarsen policy (frac_lo relative threshold), so the ONLY
    thing that differs between methods is which cells get refined.
    Returns (edges, u, refine_mask) -- the mask is returned (aligned to
    the PRE-refine cell indexing) so callers can track it for turnover
    measurement.
    """
    marker_kwargs = marker_kwargs or {}
    estimate = INDICATORS[indicator]
    eta = estimate(u, edges)
    dx = cell_widths(edges)
    if base_width is None:
        base_width = dx.max()
    # Eligibility is applied INSIDE the K-based markers (see
    # mark_deterministic_topk) so they refine exactly K eligible cells.
    # Hysteresis is a threshold rule with no budget, so masking after
    # selection is equivalent for it.
    eligible = dx > max_level_width * (1 + 1e-9)

    if marker == "hysteresis":
        refine_mask = mark_hysteresis(eta, marker_kwargs["frac_hi"], frac_lo)
    elif marker == "topk":
        refine_mask = mark_deterministic_topk(eta, marker_kwargs["K"], eligible)
    elif marker == "gumbel_topk":
        refine_mask = mark_gumbel_topk(eta, marker_kwargs["K"], marker_kwargs["temperature"], rng,
                                       eligible)
    else:
        raise ValueError(f"unknown marker: {marker}")

    refine_mask = refine_mask & eligible
    edges, u = refine(edges, u, refine_mask, prolong)

    # Recompute after refine before considering coarsening, so we don't
    # immediately re-merge cells we just split. Coarsen policy is shared
    # across all three markers -- isolates the refine rule as the only
    # experimental variable.
    #
    # Only true siblings are merged. The earlier rule merged ANY adjacent
    # pair of low-indicator cells, which produced non-nested cells of
    # mixed width (e.g. 3/64) and cells coarser than the base mesh -- so
    # the coarse regions could end up LESS resolved than a uniform n0
    # grid, adding numerical diffusion that had nothing to do with the
    # marking rule. Restricting to siblings keeps the mesh dyadic and
    # bounds every cell's width by base_width.
    indicator2 = estimate(u, edges)
    coarsen_candidate = unmark_hysteresis(indicator2, frac_lo)
    both_candidates = coarsen_candidate & np.roll(coarsen_candidate, -1) & sibling_pairs(edges, base_width)
    edges, u = coarsen(edges, u, both_candidates)

    return edges, u, refine_mask


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def indicator_contrast(u: np.ndarray, edges: np.ndarray) -> float:
    """
    Ratio of the worst cell's indicator to the typical (median) cell's
    indicator. Low early in a smooth transient (no feature stands out
    yet); grows once a shock/sharp feature actually forms and dominates.
    Always computed from the GRADIENT indicator, whichever indicator drives
    marking, so the gate opens at the same time in an indicator ablation
    and only the refine/coarsen decisions differ (refinement timing is a
    separate confound, see timing_control.py).
    """
    ind = gradient_indicator_nonuniform(u, edges)
    return ind.max() / (np.median(ind) + 1e-12)


def refined_occupancy(edges: np.ndarray, ref_points: np.ndarray, refined_width_cutoff: float) -> np.ndarray:
    """
    Project 'is this location currently refined?' onto a FIXED set of
    reference points. This gives every snapshot a common representation
    (independent of how many cells it has or how they're indexed), so
    turnover between snapshots can be measured by direct comparison
    instead of trying to match up cell indices that shift every time the
    mesh changes.
    """
    cell_idx = np.searchsorted(edges, ref_points, side="right") - 1
    cell_idx = np.clip(cell_idx, 0, len(edges) - 2)
    widths = cell_widths(edges)
    return widths[cell_idx] <= refined_width_cutoff


def jaccard_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 0.0
    intersection = np.sum(mask_a & mask_b)
    return 1.0 - intersection / union


def run_adaptive(n0: int, L: float, nu: float, t_final: float,
                  marker: str, marker_kwargs: dict, frac_lo: float,
                  regrid_dt: float = None, max_level_width: float = None,
                  cfl: float = 0.4, min_contrast: float = 3.5, seed: int = 0,
                  stats: dict = None, scheme: str = "rusanov1", prolong: str = None,
                  indicator: str = "gradient"):
    """
    Run Burgers' equation with a chosen marking strategy driving adaptive
    refinement. Regridding happens at fixed SIMULATION-TIME intervals
    (regrid_dt), not a fixed step count -- a step-count cadence would let
    methods that refine more aggressively (and therefore take smaller,
    CFL-limited timesteps) get more regrid opportunities than a method
    that stays coarser, confounding any comparison between methods.
    Returns:
        snapshots: list of (t, edges, u)
        turnover:  list of Jaccard turnover values between consecutive
                   regrid events, measured on a fixed reference grid, over
                   finest-level occupancy; only pairs of events that both
                   have finest-level cells are included (empty list if the
                   run never has two such events, i.e. turnover undefined)
    If a `stats` dict is passed it is filled with cost/budget diagnostics:
    n_steps, cell_updates (sum of cell count over all timesteps -- the
    work measure, since dt is global), n_refined and n_coarsened (pairs
    merged) per regrid event, and the min/max cell width ever reached.
    `scheme` selects the solver ("rusanov1" or "muscl2", see burgers_rhs);
    `prolong` selects how refined cells are initialized ("constant" or
    "linear", see refine) and defaults to the one consistent with the
    scheme's order.
    """
    if prolong is None:
        prolong = "linear" if scheme == "muscl2" else "constant"
    if max_level_width is None:
        max_level_width = (L / n0) / 8  # allow up to 3 refinement levels
    if regrid_dt is None:
        regrid_dt = t_final / 20  # ~20 regrid opportunities over the run

    rng = np.random.default_rng(seed)
    ref_points = np.linspace(0, L, 2048, endpoint=False)

    def initial_condition(x):
        return np.sin(2 * np.pi * x / L) + 0.5

    edges = uniform_edges(n0, L)
    u = initial_condition(cell_centers(edges))

    t = 0.0
    next_regrid_t = regrid_dt
    snapshots = [(t, edges.copy(), u.copy())]
    occupancy_history = []
    turnover = []
    base_width = L / n0
    if stats is None:
        stats = {}
    stats.update(n_steps=0, cell_updates=0, n_refined=[], n_coarsened=[], min_width=base_width, max_width=base_width)

    while t < t_final:
        dt = stable_dt_nonuniform(u, edges, nu, cfl)
        dt = min(dt, t_final - t, next_regrid_t - t if next_regrid_t > t else dt)
        u = step_burgers_nonuniform(u, edges, dt, nu, scheme)
        t += dt
        stats["n_steps"] += 1
        stats["cell_updates"] += len(u)

        if t >= next_regrid_t - 1e-12:
            next_regrid_t += regrid_dt
            if indicator_contrast(u, edges) >= min_contrast:
                n_before = len(u)
                edges, u, refine_mask = regrid(edges, u, marker, frac_lo, max_level_width,
                                                marker_kwargs, rng, base_width, prolong, indicator)
                stats["n_refined"].append(int(refine_mask.sum()))
                # each merge removes one cell: before + refined - merged = after
                stats["n_coarsened"].append(n_before + int(refine_mask.sum()) - len(u))
                widths = cell_widths(edges)
                stats["min_width"] = min(stats["min_width"], widths.min())
                stats["max_width"] = max(stats["max_width"], widths.max())
                snapshots.append((t, edges.copy(), u.copy()))

                occ = refined_occupancy(edges, ref_points, max_level_width * 1.5)
                # Turnover is only defined between two events that BOTH have
                # cells at the finest level. Two empty sets would give a
                # distance of 0 (no change measured, because there was
                # nothing to measure) and empty -> non-empty always gives 1
                # (first appearance, not churn); counting either biased the
                # mean, and made runs that never reach the finest level
                # report a spurious turnover of exactly 0.
                if occupancy_history and occ.any() and occupancy_history[-1].any():
                    turnover.append(jaccard_distance(occ, occupancy_history[-1]))
                occupancy_history.append(occ)

    snapshots.append((t, edges.copy(), u.copy()))
    return snapshots, turnover


if __name__ == "__main__":
    L = 1.0
    nu = 0.01
    t_final = 0.3
    n0 = 32
    frac_lo = 0.05
    regrid_dt = t_final / 20
    min_contrast = 3.5

    snap_hyst, turn_hyst = run_adaptive(
        n0=n0, L=L, nu=nu, t_final=t_final, marker="hysteresis",
        marker_kwargs={"frac_hi": 0.3}, frac_lo=frac_lo,
        regrid_dt=regrid_dt, min_contrast=min_contrast,
    )
    n_final_hyst = len(snap_hyst[-1][2])
    n_regrids_hyst = sum(1 for i in range(1, len(snap_hyst)) if len(snap_hyst[i][2]) != len(snap_hyst[i-1][2]))
    K = max(1, (n_final_hyst - n0) // max(1, n_regrids_hyst))

    snap_topk, turn_topk = run_adaptive(
        n0=n0, L=L, nu=nu, t_final=t_final, marker="topk",
        marker_kwargs={"K": K}, frac_lo=frac_lo,
        regrid_dt=regrid_dt, min_contrast=min_contrast,
    )

    snap_gumbel, turn_gumbel = run_adaptive(
        n0=n0, L=L, nu=nu, t_final=t_final, marker="gumbel_topk",
        marker_kwargs={"K": K, "temperature": 0.1}, frac_lo=frac_lo,
        regrid_dt=regrid_dt, min_contrast=min_contrast, seed=0,
    )

    print(f"Matched budget: K={K} cells refined per regrid event (derived from hysteresis run)")
    print()
    for name, snaps, turn in [("hysteresis", snap_hyst, turn_hyst),
                               ("deterministic top-K", snap_topk, turn_topk),
                               ("gumbel top-K (T=0.1)", snap_gumbel, turn_gumbel)]:
        final_n = len(snaps[-1][2])
        mass0 = np.sum(snaps[0][2] * cell_widths(snaps[0][1]))
        mass1 = np.sum(snaps[-1][2] * cell_widths(snaps[-1][1]))
        mean_turnover = np.mean(turn) if turn else float("nan")
        print(f"{name:24s}  final_cells={final_n:4d}  mass_drift={abs(mass1-mass0):.2e}  "
              f"mean_turnover={mean_turnover:.4f}  n_regrids={len(turn)}")
