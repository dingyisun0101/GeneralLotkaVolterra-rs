//! Shared species-last layout, diffusion, and midpoint RK2 facilities.

use ndarray::{Array1, ArrayD, IxDyn};
use physics_in_parallel::prelude::basic::{BoundaryCondition, SquareLatticeConfig};

use crate::kernel::core::{KernelCore, KernelStateView};
use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TimeStep};

use super::{KernelAlgorithmError, validate_values};

/// Per-species diffusion paired with PiP's authoritative lattice geometry.
#[derive(Clone, Debug)]
pub struct Diffusion {
    coefficients: Array1<f64>,
    space: SquareLatticeConfig,
}

impl Diffusion {
    /// Validates diffusion coefficients and spacing independently of a layout.
    pub fn new(
        coefficients: Array1<f64>,
        space: SquareLatticeConfig,
    ) -> Result<Self, KernelAlgorithmError> {
        for (index, value) in coefficients.iter().copied().enumerate() {
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
        coefficients: Array1<f64>,
        shape: &[usize],
        boundary: BoundaryCondition,
    ) -> Result<Self, KernelAlgorithmError> {
        let space = SquareLatticeConfig::try_new(shape, boundary, None)?;
        Self::new(coefficients, space)
    }

    /// Borrows per-species diffusion coefficients.
    pub const fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }

    /// Borrows PiP's complete lattice geometry and finite-difference policy.
    pub const fn space_config(&self) -> &SquareLatticeConfig {
        &self.space
    }

    fn conservative_time_step_limit(&self) -> Option<f64> {
        let maximum_diffusion = self.coefficients.iter().copied().fold(0.0_f64, f64::max);
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
    elements: usize,
    growth: Array1<f64>,
    diffusion: Diffusion,
    k1: ArrayD<f64>,
    temporary: ArrayD<f64>,
    output: ArrayD<f64>,
}

impl SpatialRk2 {
    pub(super) fn new(
        growth: Array1<f64>,
        diffusion: Diffusion,
    ) -> Result<Self, KernelAlgorithmError> {
        let species = growth.len();
        if species == 0 {
            return Err(KernelAlgorithmError::EmptySpecies);
        }
        if diffusion.coefficients.len() != species {
            return Err(KernelAlgorithmError::DiffusionLength {
                expected: species,
                actual: diffusion.coefficients.len(),
            });
        }
        let cells = diffusion.space.num_sites();
        let elements = cells.checked_mul(species).ok_or_else(|| {
            let mut shape = diffusion.space.shape().to_vec();
            shape.push(species);
            KernelAlgorithmError::SpatialShapeOverflow { shape }
        })?;
        let mut shape = diffusion.space.shape().to_vec();
        shape.push(species);
        if let Some((index, value)) = growth
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KernelAlgorithmError::NonFiniteGrowth { index, value });
        }
        let dynamic_shape = IxDyn(&shape);
        Ok(Self {
            shape: shape.into_boxed_slice(),
            species,
            elements,
            growth,
            diffusion,
            k1: ArrayD::zeros(dynamic_shape.clone()),
            temporary: ArrayD::zeros(dynamic_shape.clone()),
            output: ArrayD::zeros(dynamic_shape),
        })
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) const fn species(&self) -> usize {
        self.species
    }

    pub(super) const fn growth(&self) -> &Array1<f64> {
        &self.growth
    }

    pub(super) const fn diffusion(&self) -> &Diffusion {
        &self.diffusion
    }

    pub(super) fn scratch_lengths(&self) -> [usize; 3] {
        [self.k1.len(), self.temporary.len(), self.output.len()]
    }

    pub(super) fn validate(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), KernelAlgorithmError> {
        self.validate_layout(core, state)?;
        validate_values(ABUNDANCE_FIELD, state.abundance().iter().copied())?;
        validate_values(
            SPACE_FIELD,
            state
                .space()
                .expect("spatial layout was validated")
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
        if state.abundance().len() != self.species {
            return Err(KernelAlgorithmError::SpeciesMismatch {
                expected: self.species,
                actual: state.abundance().len(),
            });
        }
        let space = state.space().ok_or(KernelAlgorithmError::SpaceRequired)?;
        if space.shape() != self.shape() {
            return Err(KernelAlgorithmError::SpaceShapeMismatch {
                expected: self.shape().to_vec(),
                actual: space.shape().to_vec(),
            });
        }
        if space.as_slice().is_none() {
            return Err(KernelAlgorithmError::NonStandardLayout);
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
    ) -> Result<ndarray::ArrayViewD<'algorithm, f64>, KernelAlgorithmError> {
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
        let input = space.as_slice().expect("spatial state was validated");
        let first_stage = self.k1.as_slice().expect("scratch is standard contiguous");
        let temporary = self
            .temporary
            .as_slice_mut()
            .expect("scratch is standard contiguous");
        let half_dt = 0.5 * time_step.get();
        for linear_index in 0..self.elements {
            temporary[linear_index] = input[linear_index] + half_dt * first_stage[linear_index];
        }
        rhs(
            core,
            &self.growth,
            &self.diffusion,
            dynamics,
            &self.temporary,
            &mut self.k1,
            &mut self.output,
        )?;
        let second_stage = self.k1.as_slice().expect("scratch is standard contiguous");
        let output = self
            .output
            .as_slice_mut()
            .expect("scratch is standard contiguous");
        for linear_index in 0..self.elements {
            output[linear_index] =
                input[linear_index] + time_step.get() * second_stage[linear_index];
        }
        Ok(self.output.view())
    }

    pub(super) fn residual<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        dynamics: SpatialDynamics,
    ) -> Result<ndarray::ArrayViewD<'algorithm, f64>, KernelAlgorithmError> {
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
        Ok(self.k1.view())
    }
}

fn rhs(
    core: &KernelCore,
    growth: &Array1<f64>,
    diffusion: &Diffusion,
    dynamics: SpatialDynamics,
    space: &ArrayD<f64>,
    output: &mut ArrayD<f64>,
    interaction_output: &mut ArrayD<f64>,
) -> Result<(), KernelAlgorithmError> {
    let species_count = growth.len();
    let cells = diffusion.space.num_sites();
    let input = space
        .as_slice()
        .expect("spatial input is standard contiguous");
    let target = output
        .as_slice_mut()
        .expect("spatial scratch is standard contiguous");
    let interactions = interaction_output
        .as_slice_mut()
        .expect("interaction scratch is standard contiguous");
    core.apply_interactions(input, interactions)
        .expect("kernel and spatial species dimensions were validated");
    diffusion
        .space
        .laplacian(input, species_count, target)
        .map_err(KernelAlgorithmError::SpaceConfig)?;
    for cell in 0..cells {
        let base = cell * species_count;
        let cell_interactions = &interactions[base..base + species_count];
        let mean_fitness = match dynamics {
            SpatialDynamics::Glv => 0.0,
            SpatialDynamics::Replicator => (0..species_count)
                .map(|species| {
                    input[base + species] * (growth[species] + cell_interactions[species])
                })
                .sum(),
        };

        for species in 0..species_count {
            let center_index = base + species;
            let center = input[center_index];
            let interaction = cell_interactions[species];
            let laplacian = target[center_index];
            let reaction = match dynamics {
                SpatialDynamics::Glv => center * (growth[species] + interaction),
                SpatialDynamics::Replicator => {
                    center * (growth[species] + interaction - mean_fitness)
                }
            };
            target[center_index] = reaction + diffusion.coefficients[species] * laplacian;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::InteractionMatrix;
    use ndarray::{Array2, IxDyn};
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
    fn batched_interactions_match_per_cell_ndarray_linalg() {
        for species in [1, 2, 5] {
            let matrix = Array2::from_shape_fn((species, species), |(row, column)| {
                ((row * species + column + 1) as f64).sin() / species as f64
            });
            let interaction = InteractionMatrix::from_matrix(DenseMatrix::from_ndarray(&matrix))
                .expect("test matrix is square and finite");
            let core = KernelCore::new(interaction);
            let growth = Array1::from_shape_fn(species, |index| 0.1 * (index + 1) as f64);

            for cells in [1, 3, 17] {
                let lattice =
                    SquareLatticeConfig::try_new(&[cells], BoundaryCondition::Periodic, None)
                        .unwrap();
                let diffusion = Diffusion::new(Array1::zeros(species), lattice).unwrap();
                let shape = IxDyn(&[cells, species]);
                let space = ArrayD::from_shape_fn(shape.clone(), |index| {
                    0.25 + (index[0] * species + index[1] + 1) as f64 / 37.0
                });

                for dynamics in [SpatialDynamics::Glv, SpatialDynamics::Replicator] {
                    let mut actual = ArrayD::zeros(shape.clone());
                    let mut interaction_scratch = ArrayD::zeros(shape.clone());
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
                    for cell in space.as_slice().unwrap().chunks_exact(species) {
                        let interaction = matrix.dot(&Array1::from_vec(cell.to_vec()));
                        let mean_fitness = match dynamics {
                            SpatialDynamics::Glv => 0.0,
                            SpatialDynamics::Replicator => cell
                                .iter()
                                .enumerate()
                                .map(|(index, value)| value * (growth[index] + interaction[index]))
                                .sum(),
                        };
                        expected.extend(cell.iter().enumerate().map(|(index, value)| {
                            let fitness = growth[index] + interaction[index];
                            match dynamics {
                                SpatialDynamics::Glv => value * fitness,
                                SpatialDynamics::Replicator => value * (fitness - mean_fitness),
                            }
                        }));
                    }
                    assert_close(actual.as_slice().unwrap(), &expected);
                }
            }
        }
    }
}
