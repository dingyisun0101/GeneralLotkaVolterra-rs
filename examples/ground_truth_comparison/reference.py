"""Dependency-free high-resolution references for the Rust comparison."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config" / "fixed.json").read_text())
FINAL_TIME = CONFIG["physical_time_increment"] * CONFIG["maximum_iterations"]
REFERENCE_STEPS = 10_000


def matrix(name):
    document = json.loads((ROOT / "inputs" / name).read_text())
    columns = document["columns"]
    values = document["values"]
    return [values[start : start + columns] for start in range(0, len(values), columns)]


def matvec(values, vector):
    return [sum(coefficient * item for coefficient, item in zip(row, vector)) for row in values]


def integrate(rhs, initial):
    step = FINAL_TIME / REFERENCE_STEPS
    state = list(initial)
    for _ in range(REFERENCE_STEPS):
        k1 = rhs(state)
        k2 = rhs([value + 0.5 * step * slope for value, slope in zip(state, k1)])
        k3 = rhs([value + 0.5 * step * slope for value, slope in zip(state, k2)])
        k4 = rhs([value + step * slope for value, slope in zip(state, k3)])
        state = [
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(state, k1, k2, k3, k4)
        ]
    return state


def mean_field():
    case = CONFIG["mean_field"]
    interaction = matrix("mean_field_matrix.json")
    growth = case["growth"]

    def rhs(abundance):
        interaction_term = matvec(interaction, abundance)
        fitness = [base + effect for base, effect in zip(growth, interaction_term)]
        mean_fitness = sum(value * value_fitness for value, value_fitness in zip(abundance, fitness))
        return [value * (value_fitness - mean_fitness) for value, value_fitness in zip(abundance, fitness)]

    return integrate(rhs, case["initial_abundance"])


def spatial(case_name, matrix_name):
    case = CONFIG[case_name]
    interaction = matrix(matrix_name)
    growth = case["growth"]
    diffusion = case["diffusion"]
    cells, species = case["shape"]

    def rhs(flat):
        state = [flat[cell * species : (cell + 1) * species] for cell in range(cells)]
        output = []
        for cell, local in enumerate(state):
            interaction_term = matvec(interaction, local)
            plus = state[(cell + 1) % cells]
            minus = state[(cell - 1) % cells]
            for index in range(species):
                reaction = local[index] * (growth[index] + interaction_term[index])
                laplacian = plus[index] + minus[index] - 2.0 * local[index]
                output.append(reaction + diffusion[index] * laplacian)
        return output

    return integrate(rhs, case["initial_space"])


print(json.dumps({
    "mean_field": mean_field(),
    "spatial_no_diffusion": spatial("spatial_no_diffusion", "spatial_no_diffusion_matrix.json"),
    "spatial_periodic_diffusion": spatial("spatial_periodic_diffusion", "spatial_periodic_diffusion_matrix.json"),
}))
