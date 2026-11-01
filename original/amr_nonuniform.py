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

def step_burgers_nonuniform(u: np.ndarray, edges: np.ndarray, dt: float, nu: float) -> np.ndarray:
    """
    Advance u by one explicit timestep on a non-uniform periodic grid,
    using a CONSERVATIVE finite-volume scheme:
        - Advective flux: global Lax-Friedrichs numerical flux for
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
    dx = cell_widths(edges)
    u_right = np.roll(u, -1)
    dx_right = np.roll(dx, -1)
    h_face = 0.5 * (dx + dx_right)  # distance between cell i and i+1 centers

    # Advective numerical flux at the face between cell i and cell i+1.
    # LOCAL Lax-Friedrichs: viscosity coefficient is computed PER FACE from
    # just the two neighboring cell values, not a single global max over
    # the whole domain. This matters a lot for AMR specifically: with a
    # global alpha, numerical diffusion at a face is set by the fastest
    # wave speed ANYWHERE in the domain, so refining a cell locally doesn't
    # actually reduce the numerical diffusion smearing the shock there --
    # the shock stays exactly as smeared as it would be on a coarse grid.
    # Local alpha lets refinement actually sharpen the shock, which is the
    # entire point of doing AMR in the first place.
    f = 0.5 * u**2
    f_right = np.roll(f, -1)
    alpha = np.maximum(np.abs(u), np.abs(u_right)) + 1e-12
    F_adv = 0.5 * (f + f_right) - 0.5 * alpha * (u_right - u)

    # Diffusive numerical flux at the same face.
    F_diff = nu * (u_right - u) / h_face

    F_adv_left = np.roll(F_adv, 1)   # flux at the LEFT face of cell i
    F_diff_left = np.roll(F_diff, 1)

    return u + dt * (
        -(F_adv - F_adv_left) / dx
        + (F_diff - F_diff_left) / dx
    )


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


# ---------------------------------------------------------------------------
# Mesh operations: refine / coarsen
# ---------------------------------------------------------------------------

def refine(edges: np.ndarray, u: np.ndarray, marked: np.ndarray):
    """
    Split each marked cell into two equal halves. Solution value is
    duplicated to both children (piecewise-constant injection).
    `marked` is a boolean array, one entry per current cell.
    """
    new_edges = [edges[0]]
    new_u = []
    for i in range(len(u)):
        left, right = edges[i], edges[i + 1]
        if marked[i]:
            mid = 0.5 * (left + right)
            new_edges.append(mid)
            new_edges.append(right)
            new_u.append(u[i])
            new_u.append(u[i])
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


def mark_deterministic_topk(indicator: np.ndarray, K: int) -> np.ndarray:
    """Refine exactly the K cells with the largest indicator values."""
    K = min(K, len(indicator))
    mask = np.zeros_like(indicator, dtype=bool)
    top_idx = np.argpartition(indicator, -K)[-K:] if K > 0 else np.array([], dtype=int)
    mask[top_idx] = True
    return mask


def mark_gumbel_topk(indicator: np.ndarray, K: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
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
    """
    K = min(K, len(indicator))
    eps = 1e-12
    normalized = indicator / (indicator.max() + eps)
    scores = normalized / (temperature + eps)
    u = rng.uniform(size=len(indicator))
    gumbel_noise = -np.log(-np.log(u + eps) + eps)
    perturbed = scores + gumbel_noise
    mask = np.zeros_like(indicator, dtype=bool)
    top_idx = np.argpartition(perturbed, -K)[-K:] if K > 0 else np.array([], dtype=int)
    mask[top_idx] = True
    return mask


def regrid(edges: np.ndarray, u: np.ndarray, marker: str, frac_lo: float,
           max_level_width: float, marker_kwargs: dict = None, rng: np.random.Generator = None):
    """
    One MARK + REFINE/COARSEN pass. `marker` selects the refine-selection
    rule ('hysteresis', 'topk', or 'gumbel_topk') -- all three share the
    exact same coarsen policy (frac_lo relative threshold), so the ONLY
    thing that differs between methods is which cells get refined.
    Returns (edges, u, refine_mask) -- the mask is returned (aligned to
    the PRE-refine cell indexing) so callers can track it for turnover
    measurement.
    """
    marker_kwargs = marker_kwargs or {}
    indicator = gradient_indicator_nonuniform(u, edges)
    dx = cell_widths(edges)

    if marker == "hysteresis":
        refine_mask = mark_hysteresis(indicator, marker_kwargs["frac_hi"], frac_lo)
    elif marker == "topk":
        refine_mask = mark_deterministic_topk(indicator, marker_kwargs["K"])
    elif marker == "gumbel_topk":
        refine_mask = mark_gumbel_topk(indicator, marker_kwargs["K"], marker_kwargs["temperature"], rng)
    else:
        raise ValueError(f"unknown marker: {marker}")

    refine_mask = refine_mask & (dx > max_level_width)
    edges, u = refine(edges, u, refine_mask)

    # Recompute after refine before considering coarsening, so we don't
    # immediately re-merge cells we just split. Coarsen policy is shared
    # across all three markers -- isolates the refine rule as the only
    # experimental variable.
    indicator2 = gradient_indicator_nonuniform(u, edges)
    coarsen_candidate = unmark_hysteresis(indicator2, frac_lo)
    both_candidates = coarsen_candidate & np.roll(coarsen_candidate, -1)
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
                  cfl: float = 0.4, min_contrast: float = 3.5, seed: int = 0):
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
                   regrid events, measured on a fixed reference grid
    """
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

    while t < t_final:
        dt = stable_dt_nonuniform(u, edges, nu, cfl)
        dt = min(dt, t_final - t, next_regrid_t - t if next_regrid_t > t else dt)
        u = step_burgers_nonuniform(u, edges, dt, nu)
        t += dt

        if t >= next_regrid_t - 1e-12:
            next_regrid_t += regrid_dt
            if indicator_contrast(u, edges) >= min_contrast:
                edges, u, refine_mask = regrid(edges, u, marker, frac_lo, max_level_width,
                                                marker_kwargs, rng)
                snapshots.append((t, edges.copy(), u.copy()))

                occ = refined_occupancy(edges, ref_points, max_level_width * 1.5)
                if occupancy_history:
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
