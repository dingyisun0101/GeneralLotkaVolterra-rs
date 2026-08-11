"""Regenerate deterministic GLV ground truth with independent fine-step RK4.

This maintainer tool uses only Python's standard library. Routine Rust tests
consume the checked-in fixture and do not require Python.
"""

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "ground_truth.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
FINAL_TIME = FIXTURE["generator"]["final_physical_time"]
REFERENCE_STEPS = FIXTURE["generator"]["reference_steps"]


def matvec(matrix, vector):
    return [sum(a * x for a, x in zip(row, vector)) for row in matrix]


def rows(flat, columns):
    return [flat[start : start + columns] for start in range(0, len(flat), columns)]


def integrate(rhs, initial):
    dt = FINAL_TIME / REFERENCE_STEPS
    state = list(initial)
    for _ in range(REFERENCE_STEPS):
        k1 = rhs(state)
        k2 = rhs([x + 0.5 * dt * dx for x, dx in zip(state, k1)])
        k3 = rhs([x + 0.5 * dt * dx for x, dx in zip(state, k2)])
        k4 = rhs([x + dt * dx for x, dx in zip(state, k3)])
        state = [
            x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for x, a, b, c, d in zip(state, k1, k2, k3, k4)
        ]
    return state


def mean_field(case):
    species = len(case["initial_abundance"])
    interaction = rows(case["interaction"], species)

    def rhs(abundance):
        effects = matvec(interaction, abundance)
        fitness = [r + effect for r, effect in zip(case["growth"], effects)]
        mean = sum(x * value for x, value in zip(abundance, fitness))
        return [x * (value - mean) for x, value in zip(abundance, fitness)]

    return integrate(rhs, case["initial_abundance"])


def spatial_glv(case):
    cells, species = case["shape"]
    interaction = rows(case["interaction"], species)

    def rhs(flat):
        state = rows(flat, species)
        output = []
        for cell, local in enumerate(state):
            effects = matvec(interaction, local)
            plus = state[(cell + 1) % cells]
            minus = state[(cell - 1) % cells]
            for index in range(species):
                reaction = local[index] * (case["growth"][index] + effects[index])
                laplacian = plus[index] + minus[index] - 2.0 * local[index]
                output.append(reaction + case["diffusion"][index] * laplacian)
        return output

    return integrate(rhs, case["initial_space"])


cases = FIXTURE["cases"]
print(
    json.dumps(
        {
            "mean_field": mean_field(cases["mean_field"]),
            "spatial_no_diffusion": spatial_glv(cases["spatial_no_diffusion"]),
            "spatial_periodic_diffusion": spatial_glv(cases["spatial_periodic_diffusion"]),
        },
        indent=2,
    )
)
