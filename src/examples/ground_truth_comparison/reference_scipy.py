"""SciPy reference solutions for the Rust ground-truth comparison example."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

DT = 0.001
NUM_STEPS = 100
T_FINAL = DT * NUM_STEPS


def solve_reference(fun, y0):
    sol = solve_ivp(
        fun,
        (0.0, T_FINAL),
        np.asarray(y0, dtype=float),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        t_eval=[T_FINAL],
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]


def replicator_reference():
    interaction = np.array(
        [
            [0.0, 0.3, -0.2],
            [-0.1, 0.0, 0.25],
            [0.15, -0.35, 0.0],
        ],
        dtype=float,
    )
    growth = np.array([0.04, -0.02, 0.01], dtype=float)
    y0 = np.array([0.2, 0.5, 0.3], dtype=float)

    def rhs(_t, y):
        interaction_term = interaction @ y
        fitness = growth + interaction_term
        upsilon = float(y @ fitness)
        return y * (fitness - upsilon)

    return solve_reference(rhs, y0)


def glv_no_diffusion_reference():
    interaction = np.array(
        [
            [-0.40, 0.08],
            [-0.05, -0.30],
        ],
        dtype=float,
    )
    growth = np.array([0.20, 0.10], dtype=float)
    initial_space = np.array(
        [
            [0.40, 0.20],
            [0.15, 0.50],
            [0.30, 0.25],
            [0.10, 0.35],
        ],
        dtype=float,
    )

    def rhs(_t, flat):
        space = flat.reshape(initial_space.shape)
        out = np.zeros_like(space)
        for cell in range(space.shape[0]):
            local = space[cell]
            out[cell] = local * (growth + interaction @ local)
        return out.ravel()

    return solve_reference(rhs, initial_space.ravel())


def glv_periodic_diffusion_reference():
    interaction = np.array(
        [
            [-0.30, 0.04],
            [-0.02, -0.25],
        ],
        dtype=float,
    )
    growth = np.array([0.05, 0.02], dtype=float)
    diffusion = np.array([0.01, 0.02], dtype=float)
    initial_space = np.array(
        [
            [0.30, 0.10],
            [0.25, 0.20],
            [0.10, 0.40],
            [0.15, 0.30],
        ],
        dtype=float,
    )
    num_cells = initial_space.shape[0]

    def rhs(_t, flat):
        space = flat.reshape(initial_space.shape)
        out = np.zeros_like(space)
        for cell in range(num_cells):
            local = space[cell]
            reaction = local * (growth + interaction @ local)
            plus = space[(cell + 1) % num_cells]
            minus = space[(cell - 1) % num_cells]
            laplacian = plus + minus - 2.0 * local
            out[cell] = reaction + diffusion * laplacian
        return out.ravel()

    return solve_reference(rhs, initial_space.ravel())


def main():
    output_path = Path(__file__).resolve().parents[2] / "output" / "ground_truth_comparison"
    output_path.mkdir(parents=True, exist_ok=True)
    references = {
        "well_mixed_replicator": replicator_reference().tolist(),
        "spatial_glv_no_diffusion": glv_no_diffusion_reference().tolist(),
        "spatial_glv_periodic_diffusion": glv_periodic_diffusion_reference().tolist(),
    }
    reference_path = output_path / "scipy_reference.json"
    reference_path.write_text(json.dumps(references, indent=2))
    print(reference_path)


if __name__ == "__main__":
    main()
