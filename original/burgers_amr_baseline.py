"""
Baseline harness for the AMR marking-strategy project.

Solves the 1D viscous Burgers' equation:
    u_t + u u_x = nu * u_xx

on a uniform grid using a simple finite-difference scheme, and computes
a per-cell gradient-based error indicator at each timestep. This is the
"solve + estimate" half of the SOLVE -> ESTIMATE -> MARK -> REFINE loop.

The MARK step (threshold+hysteresis, deterministic bulk top-K, and
temperature-scaled probabilistic top-K) is deliberately NOT implemented
here yet -- this script just gives you a working solver and indicator
to attach those strategies to next.

Numerical method:
    - Advective term: first-order upwind (stable, simple, a bit
      diffusive -- fine for a first pass; you can swap in a
      higher-order scheme later without touching the AMR logic).
    - Diffusive term: standard second-order central difference.
    - Time stepping: explicit Euler, with dt chosen from a CFL-style
      condition that accounts for both advection and diffusion.

A reference solution is also computed on a much finer uniform grid so
you have a ground truth to measure error against once you plot the
error-vs-cell-count Pareto frontier.
"""

import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth initial data that steepens into a shock under Burgers' dynamics."""
    return np.sin(2 * np.pi * x / x[-1]) + 0.5


def step_burgers(u: np.ndarray, dx: float, dt: float, nu: float) -> np.ndarray:
    """
    Advance u by one explicit timestep using upwind advection +
    central-difference diffusion, with periodic boundary conditions.
    """
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)

    # Upwind advective term: use backward difference where u > 0,
    # forward difference where u < 0.
    dudx_back = (u - u_left) / dx
    dudx_fwd = (u_right - u) / dx
    advective = np.where(u > 0, u * dudx_back, u * dudx_fwd)

    # Central-difference diffusive term.
    diffusive = nu * (u_right - 2 * u + u_left) / dx**2

    return u + dt * (-advective + diffusive)


def stable_dt(u: np.ndarray, dx: float, nu: float, cfl: float = 0.4) -> float:
    """
    Explicit-scheme stability limit: combine the advective CFL condition
    with the diffusive stability condition, take the tighter one, and
    apply a safety factor (cfl < 1).
    """
    u_max = np.max(np.abs(u)) + 1e-12
    dt_advect = dx / u_max
    dt_diffuse = dx**2 / (2 * nu) if nu > 0 else np.inf
    return cfl * min(dt_advect, dt_diffuse)


def gradient_indicator(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Per-cell error indicator: local slope magnitude scaled by cell width.
    Large near shocks/steep gradients, near zero in smooth regions.
    Periodic boundary conditions, same convention as the solver.
    """
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    slope = (u_right - u_left) / (2 * dx)
    return dx * np.abs(slope)


def run_solver(n_cells: int, L: float, nu: float, t_final: float, cfl: float = 0.4):
    """
    Run the Burgers' solver on a uniform grid of n_cells cells over
    [0, L] until t_final. Returns the final state, the grid, and the
    error indicator at the final time.
    """
    dx = L / n_cells
    x = (np.arange(n_cells) + 0.5) * dx  # cell centers
    u = initial_condition(x)

    t = 0.0
    while t < t_final:
        dt = stable_dt(u, dx, nu, cfl)
        dt = min(dt, t_final - t)
        u = step_burgers(u, dx, dt, nu)
        t += dt

    indicator = gradient_indicator(u, dx)
    return x, u, indicator


def l2_error(u_coarse_on_fine_grid: np.ndarray, u_reference: np.ndarray, dx_fine: float) -> float:
    """L2 norm of the difference, scaled by cell width (a discrete integral)."""
    return np.sqrt(dx_fine * np.sum((u_coarse_on_fine_grid - u_reference) ** 2))


if __name__ == "__main__":
    # --- Problem setup ---
    L = 1.0
    nu = 0.01       # viscosity: smaller -> sharper shock
    t_final = 0.3

    # --- Reference solution on a fine uniform grid ---
    n_fine = 4096
    x_fine, u_fine, indicator_fine = run_solver(n_fine, L, nu, t_final)

    # --- Baseline coarse run (uniform grid, no AMR yet) ---
    n_coarse = 128
    x_coarse, u_coarse, indicator_coarse = run_solver(n_coarse, L, nu, t_final)

    # Interpolate the coarse solution onto the fine grid to compute error.
    u_coarse_on_fine = np.interp(x_fine, x_coarse, u_coarse, period=L)
    dx_fine = L / n_fine
    err = l2_error(u_coarse_on_fine, u_fine, dx_fine)

    print(f"Reference grid: {n_fine} cells")
    print(f"Coarse grid:    {n_coarse} cells")
    print(f"L2 error (coarse vs. reference): {err:.6e}")
    print(f"Max indicator value (coarse grid): {indicator_coarse.max():.4f}")
    print(f"Mean indicator value (coarse grid): {indicator_coarse.mean():.4f}")
    print()
    print("Indicator is highest near the shock -- check that its argmax")
    print(f"lands near the steepest part of u. argmax location: x = {x_coarse[np.argmax(indicator_coarse)]:.4f}")
