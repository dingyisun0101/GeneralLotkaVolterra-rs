//! Shared species-last layout, diffusion, and midpoint RK2 facilities.

use physics_in_parallel::prelude::basic::{BoundaryCondition, SquareLatticeConfig, Tensor};

use crate::kernel::core::{KernelCore, KernelStateView};
use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TimeStep};

use super::{KernelAlgorithmError, validate_values};

/// Per-species diffusion paired with PiP's authoritative lattice geometry.
#[derive(Clone, Debug)]
pub struct Diffusion {
    coefficients: Tensor<f64>,
    space: SquareLatticeConfig,
}

impl Diffusion {
    /// Validates diffusion coefficients and spacing independently of a layout.
    pub fn new(
        coefficients: Tensor<f64>,
        space: SquareLatticeConfig,
    ) -> Result<Self, KernelAlgorithmError> {
        if coefficients.rank() != 1 {
            return Err(KernelAlgorithmError::CoefficientRank {
                field: "diffusion",
                actual: coefficients.rank(),
            });
        }
        for (index, value) in coefficients.as_slice().iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(KernelAlgorithmError::InvalidDiffusion { index, value });
            }
        }
        Ok(Self {
            coefficients,
            space,
        })
    }

    /// Creates validated diffusion on a unit-spaced grid.
    pub fn unit_spacing(
        coefficients: Tensor<f64>,
        shape: &[usize],
        boundary: BoundaryCondition,
    ) -> Result<Self, KernelAlgorithmError> {
        let space = SquareLatticeConfig::try_new(shape, boundary, None)?;
        Self::new(coefficients, space)
    }

    /// Borrows per-species diffusion coefficients.
    pub const fn coefficients(&self) -> &Tensor<f64> {
        &self.coefficients
    }

    /// Borrows PiP's complete lattice geometry and finite-difference policy.
    pub const fn space_config(&self) -> &SquareLatticeConfig {
        &self.space
    }

    fn conservative_time_step_limit(&self) -> Option<f64> {
        let maximum_diffusion = self
            .coefficients
            .as_slice()
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        if maximum_diffusion == 0.0 {
            return None;
        }
        let inverse_spacing_sum: f64 = self
            .space
            .spacing()
            .iter()
            .map(|spacing| 1.0 / spacing.powi(2))
            .sum();
        Some(1.0 / (2.0 * maximum_diffusion * inverse_spacing_sum))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SpatialDynamics {
    Replicator,
    Glv,
}

/// Common owned configuration and scratch behind both spatial algorithms.
#[derive(Debug)]
pub(super) struct SpatialRk2 {
    shape: Box<[usize]>,
    species: usize,
    growth: Tensor<f64>,
    diffusion: Diffusion,
    k1: Tensor<f64>,
    temporary: Tensor<f64>,
    output: Tensor<f64>,
}

impl SpatialRk2 {
    pub(super) fn new(
        growth: Tensor<f64>,
        diffusion: Diffusion,
    ) -> Result<Self, KernelAlgorithmError> {
        if growth.rank() != 1 {
            return Err(KernelAlgorithmError::CoefficientRank {
                field: "growth",
                actual: growth.rank(),
            });
        }
        let species = growth.size();
        if species == 0 {
            return Err(KernelAlgorithmError::EmptySpecies);
        }
        if diffusion.coefficients.size() != species {
            return Err(KernelAlgorithmError::DiffusionLength {
                expected: species,
                actual: diffusion.coefficients.size(),
            });
        }
        let cells = diffusion.space.num_sites();
        cells.checked_mul(species).ok_or_else(|| {
            let mut shape = diffusion.space.shape().to_vec();
            shape.push(species);
            KernelAlgorithmError::SpatialShapeOverflow { shape }
        })?;
        let mut shape = diffusion.space.shape().to_vec();
        shape.push(species);
        if let Some((index, value)) = growth
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KernelAlgorithmError::NonFiniteGrowth { index, value });
        }
        Ok(Self {
            shape: shape.clone().into_boxed_slice(),
            species,
            growth,
            diffusion,
            k1: Tensor::zeros(&shape),
            temporary: Tensor::zeros(&shape),
            output: Tensor::zeros(&shape),
        })
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) const fn species(&self) -> usize {
        self.species
    }

    pub(super) const fn growth(&self) -> &Tensor<f64> {
        &self.growth
    }

    pub(super) const fn diffusion(&self) -> &Diffusion {
        &self.diffusion
    }

    pub(super) fn scratch_lengths(&self) -> [usize; 3] {
        [self.k1.size(), self.temporary.size(), self.output.size()]
    }

    pub(super) fn validate(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), KernelAlgorithmError> {
        self.validate_layout(core, state)?;
        validate_values(
            ABUNDANCE_FIELD,
            state.abundance().as_slice().iter().copied(),
        )?;
        validate_values(
            SPACE_FIELD,
            state
                .space()
                .expect("spatial layout was validated")
                .as_slice()
                .iter()
                .copied(),
        )
    }

    fn validate_layout(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), KernelAlgorithmError> {
        if core.species() != self.species {
            return Err(KernelAlgorithmError::CoreSpeciesMismatch {
                expected: self.species,
                actual: core.species(),
            });
        }
        if state.abundance().shape() != [self.species] {
            return Err(KernelAlgorithmError::SpeciesMismatch {
                expected: self.species,
                actual: state.abundance().size(),
            });
        }
        let space = state.space().ok_or(KernelAlgorithmError::SpaceRequired)?;
        if space.shape() != self.shape() {
            return Err(KernelAlgorithmError::SpaceShapeMismatch {
                expected: self.shape().to_vec(),
                actual: space.shape().to_vec(),
            });
        }
        Ok(())
    }

    pub(super) fn validate_time_step(
        &self,
        time_step: TimeStep,
    ) -> Result<(), KernelAlgorithmError> {
        if let Some(maximum) = self.diffusion.conservative_time_step_limit()
            && time_step.get() > maximum
        {
            return Err(KernelAlgorithmError::UnstableTimeStep {
                actual: time_step.get(),
                maximum,
            });
        }
        Ok(())
    }

    pub(super) fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
        dynamics: SpatialDynamics,
    ) -> Result<&'algorithm Tensor<f64>, KernelAlgorithmError> {
        self.validate_layout(core, state)?;
        self.validate_time_step(time_step)?;
        let space = state.space().expect("spatial state was validated");
        rhs(
            core,
            &self.growth,
            &self.diffusion,
            dynamics,
            space,
            &mut self.k1,
            &mut self.output,
        )?;
        let half_dt = 0.5 * time_step.get();
        self.temporary
            .zip_from(space, &self.k1, |value, rate| value + half_dt * rate)
            .expect("RK2 scratch shapes are fixed at construction");
        rhs(
            core,
            &self.growth,
            &self.diffusion,
            dynamics,
            &self.temporary,
            &mut self.k1,
            &mut self.output,
        )?;
        let dt = time_step.get();
        self.output
            .zip_from(space, &self.k1, |value, rate| value + dt * rate)
            .expect("RK2 scratch shapes are fixed at construction");
        Ok(&self.output)
    }

    pub(super) fn residual<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        dynamics: SpatialDynamics,
    ) -> Result<&'algorithm Tensor<f64>, KernelAlgorithmError> {
        self.validate_layout(core, state)?;
        rhs(
            core,
            &self.growth,
            &self.diffusion,
            dynamics,
            state.space().expect("spatial state was validated"),
            &mut self.k1,
            &mut self.output,
        )?;
        Ok(&self.k1)
    }
}

fn rhs(
    core: &KernelCore,
    growth: &Tensor<f64>,
    diffusion: &Diffusion,
    dynamics: SpatialDynamics,
    space: &Tensor<f64>,
    output: &mut Tensor<f64>,
    interaction_output: &mut Tensor<f64>,
) -> Result<(), KernelAlgorithmError> {
    let species_count = growth.size();
    let input = space.as_slice();
    core.apply_interactions(input, interaction_output.as_mut_slice())
        .expect("kernel and spatial species dimensions were validated");
    diffusion
        .space
        .laplacian(input, species_count, output.as_mut_slice())
        .map_err(KernelAlgorithmError::SpaceConfig)?;

    let interactions = interaction_output.as_slice();
    let growth = growth.as_slice();
    let coefficients = diffusion.coefficients.as_slice();
    output.for_each_chunk_mut(species_count, |cell, target| {
        let base = cell * species_count;
        let cell_input = &input[base..base + species_count];
        let cell_interactions = &interactions[base..base + species_count];
        let mean_fitness = match dynamics {
            SpatialDynamics::Glv => 0.0,
            SpatialDynamics::Replicator => cell_input
                .iter()
                .enumerate()
                .map(|(species, value)| value * (growth[species] + cell_interactions[species]))
                .sum(),
        };
        for species in 0..species_count {
            let center = cell_input[species];
            let reaction = match dynamics {
                SpatialDynamics::Glv => center * (growth[species] + cell_interactions[species]),
                SpatialDynamics::Replicator => {
                    center * (growth[species] + cell_interactions[species] - mean_fitness)
                }
            };
            target[species] = reaction + coefficients[species] * target[species];
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::InteractionMatrix;
    use physics_in_parallel::prelude::basic::DenseMatrix;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1.0e-12 * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn batched_interactions_match_per_cell_reference() {
        for species in [1, 2, 5] {
            let matrix_values: Vec<_> = (0..species)
                .flat_map(|row| {
                    (0..species).map(move |column| {
                        ((row * species + column + 1) as f64).sin() / species as f64
                    })
                })
                .collect();
            let matrix = DenseMatrix::from_vec(species, species, matrix_values.clone());
            let interaction =
                InteractionMatrix::from_matrix(matrix).expect("test matrix is square and finite");
            let core = KernelCore::new(interaction);
            let growth = Tensor::from_vec(
                &[species],
                (0..species).map(|index| 0.1 * (index + 1) as f64).collect(),
            );

            for cells in [1, 3, 17] {
                let lattice =
                    SquareLatticeConfig::try_new(&[cells], BoundaryCondition::Periodic, None)
                        .unwrap();
                let diffusion = Diffusion::new(Tensor::zeros(&[species]), lattice).unwrap();
                let shape = [cells, species];
                let space = Tensor::from_vec(
                    &shape,
                    (0..cells * species)
                        .map(|index| 0.25 + (index + 1) as f64 / 37.0)
                        .collect(),
                );

                for dynamics in [SpatialDynamics::Glv, SpatialDynamics::Replicator] {
                    let mut actual = Tensor::zeros(&shape);
                    let mut interaction_scratch = Tensor::zeros(&shape);
                    rhs(
                        &core,
                        &growth,
                        &diffusion,
                        dynamics,
                        &space,
                        &mut actual,
                        &mut interaction_scratch,
                    )
                    .unwrap();

                    let mut expected = Vec::with_capacity(cells * species);
                    for cell in space.as_slice().chunks_exact(species) {
                        let interaction: Vec<f64> = (0..species)
                            .map(|row| {
                                (0..species)
                                    .map(|column| {
                                        matrix_values[row * species + column] * cell[column]
                                    })
                                    .sum()
                            })
                            .collect();
                        let mean_fitness = match dynamics {
                            SpatialDynamics::Glv => 0.0,
                            SpatialDynamics::Replicator => cell
                                .iter()
                                .enumerate()
                                .map(|(index, value)| {
                                    value * (growth.as_slice()[index] + interaction[index])
                                })
                                .sum(),
                        };
                        expected.extend(cell.iter().enumerate().map(|(index, value)| {
                            let fitness = growth.as_slice()[index] + interaction[index];
                            match dynamics {
                                SpatialDynamics::Glv => value * fitness,
                                SpatialDynamics::Replicator => value * (fitness - mean_fitness),
                            }
                        }));
                    }
                    assert_close(actual.as_slice(), &expected);
                }
            }
        }
    }
}
