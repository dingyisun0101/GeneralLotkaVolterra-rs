//! Shared species-last layout, diffusion, and midpoint RK2 facilities.

use ndarray::{Array1, ArrayD, IxDyn};

use crate::kernel::core::{KernelCore, KernelStateView};
use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TimeStep};

use super::{KernelAlgorithmError, validate_values};

/// Finite-difference behavior at every spatial edge.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Boundary {
    /// Wrap the grid to its opposite edge.
    Periodic,
    /// Impose zero flux by reusing the edge value outside the grid.
    Neumann,
}

/// Validated per-species diffusion and per-axis grid spacing.
#[derive(Clone, Debug)]
pub struct Diffusion {
    coefficients: Array1<f64>,
    spacing: Box<[f64]>,
    inverse_spacing_squared: Box<[f64]>,
    boundary: Boundary,
}

impl Diffusion {
    /// Validates diffusion coefficients and spacing independently of a layout.
    pub fn new(
        coefficients: Array1<f64>,
        spacing: Vec<f64>,
        boundary: Boundary,
    ) -> Result<Self, KernelAlgorithmError> {
        for (index, value) in coefficients.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(KernelAlgorithmError::InvalidDiffusion { index, value });
            }
        }
        let mut inverse_spacing_squared = Vec::with_capacity(spacing.len());
        for (index, value) in spacing.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(KernelAlgorithmError::InvalidSpacing { index, value });
            }
            inverse_spacing_squared.push(1.0 / (value * value));
        }
        Ok(Self {
            coefficients,
            spacing: spacing.into_boxed_slice(),
            inverse_spacing_squared: inverse_spacing_squared.into_boxed_slice(),
            boundary,
        })
    }

    /// Creates validated diffusion on a unit-spaced grid.
    pub fn unit_spacing(
        coefficients: Array1<f64>,
        spatial_dimensions: usize,
        boundary: Boundary,
    ) -> Result<Self, KernelAlgorithmError> {
        Self::new(coefficients, vec![1.0; spatial_dimensions], boundary)
    }

    /// Borrows per-species diffusion coefficients.
    pub const fn coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }

    /// Borrows per-axis grid spacing.
    pub fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    /// Returns the edge behavior.
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }

    fn conservative_time_step_limit(&self) -> Option<f64> {
        let maximum_diffusion = self.coefficients.iter().copied().fold(0.0_f64, f64::max);
        if maximum_diffusion == 0.0 {
            return None;
        }
        let inverse_spacing_sum: f64 = self.inverse_spacing_squared.iter().sum();
        Some(1.0 / (2.0 * maximum_diffusion * inverse_spacing_sum))
    }
}

/// Cached facts for a standard contiguous species-last spatial array.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpatialLayout {
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    spatial_dimensions: usize,
    species: usize,
    cells: usize,
    elements: usize,
}

impl SpatialLayout {
    /// Validates a shape and computes checked row-major strides once.
    pub fn new(shape: Vec<usize>) -> Result<Self, KernelAlgorithmError> {
        if shape.len() < 2 {
            return Err(KernelAlgorithmError::SpatialRank);
        }
        let spatial_dimensions = shape.len() - 1;
        if shape[spatial_dimensions] == 0 {
            return Err(KernelAlgorithmError::EmptySpecies);
        }
        if let Some((axis, _)) = shape[..spatial_dimensions]
            .iter()
            .enumerate()
            .find(|(_, length)| **length == 0)
        {
            return Err(KernelAlgorithmError::EmptySpatialAxis { axis });
        }
        let species = shape[spatial_dimensions];
        let mut strides = vec![1_usize; shape.len()];
        for axis in (0..spatial_dimensions).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape[axis + 1])
                .ok_or_else(|| KernelAlgorithmError::SpatialShapeOverflow {
                    shape: shape.clone(),
                })?;
        }
        let cells = shape[..spatial_dimensions]
            .iter()
            .try_fold(1_usize, |product, length| product.checked_mul(*length))
            .ok_or_else(|| KernelAlgorithmError::SpatialShapeOverflow {
                shape: shape.clone(),
            })?;
        let elements = cells.checked_mul(species).ok_or_else(|| {
            KernelAlgorithmError::SpatialShapeOverflow {
                shape: shape.clone(),
            }
        })?;
        Ok(Self {
            shape: shape.into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            spatial_dimensions,
            species,
            cells,
            elements,
        })
    }

    /// Borrows the full species-last shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the count of spatial axes, excluding species.
    pub const fn spatial_dimensions(&self) -> usize {
        self.spatial_dimensions
    }

    /// Returns the final-axis species count.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Returns the product of all spatial axis lengths.
    pub const fn cells(&self) -> usize {
        self.cells
    }

    /// Returns the complete array element count.
    pub const fn elements(&self) -> usize {
        self.elements
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
    layout: SpatialLayout,
    growth: Array1<f64>,
    diffusion: Diffusion,
    k1: ArrayD<f64>,
    temporary: ArrayD<f64>,
    output: ArrayD<f64>,
}

impl SpatialRk2 {
    pub(super) fn new(
        shape: Vec<usize>,
        growth: Array1<f64>,
        diffusion: Diffusion,
    ) -> Result<Self, KernelAlgorithmError> {
        let layout = SpatialLayout::new(shape)?;
        if growth.len() != layout.species {
            return Err(KernelAlgorithmError::GrowthLength {
                expected: layout.species,
                actual: growth.len(),
            });
        }
        if let Some((index, value)) = growth
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KernelAlgorithmError::NonFiniteGrowth { index, value });
        }
        if diffusion.coefficients.len() != layout.species {
            return Err(KernelAlgorithmError::DiffusionLength {
                expected: layout.species,
                actual: diffusion.coefficients.len(),
            });
        }
        if diffusion.spacing.len() != layout.spatial_dimensions {
            return Err(KernelAlgorithmError::SpacingLength {
                expected: layout.spatial_dimensions,
                actual: diffusion.spacing.len(),
            });
        }
        let dynamic_shape = IxDyn(layout.shape());
        Ok(Self {
            layout,
            growth,
            diffusion,
            k1: ArrayD::zeros(dynamic_shape.clone()),
            temporary: ArrayD::zeros(dynamic_shape.clone()),
            output: ArrayD::zeros(dynamic_shape),
        })
    }

    pub(super) const fn layout(&self) -> &SpatialLayout {
        &self.layout
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
        if core.species() != self.layout.species {
            return Err(KernelAlgorithmError::CoreSpeciesMismatch {
                expected: self.layout.species,
                actual: core.species(),
            });
        }
        if state.abundance().len() != self.layout.species {
            return Err(KernelAlgorithmError::SpeciesMismatch {
                expected: self.layout.species,
                actual: state.abundance().len(),
            });
        }
        validate_values(ABUNDANCE_FIELD, state.abundance().iter().copied())?;
        let space = state.space().ok_or(KernelAlgorithmError::SpaceRequired)?;
        if space.shape() != self.layout.shape() {
            return Err(KernelAlgorithmError::SpaceShapeMismatch {
                expected: self.layout.shape().to_vec(),
                actual: space.shape().to_vec(),
            });
        }
        if space.as_slice().is_none() {
            return Err(KernelAlgorithmError::NonStandardLayout);
        }
        validate_values(SPACE_FIELD, space.iter().copied())
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
        self.validate(core, state)?;
        self.validate_time_step(time_step)?;
        let space = state.space().expect("spatial state was validated");
        rhs(
            core,
            &self.layout,
            &self.growth,
            &self.diffusion,
            dynamics,
            space,
            &mut self.k1,
        );
        let input = space.as_slice().expect("spatial state was validated");
        let first_stage = self.k1.as_slice().expect("scratch is standard contiguous");
        let temporary = self
            .temporary
            .as_slice_mut()
            .expect("scratch is standard contiguous");
        let half_dt = 0.5 * time_step.get();
        for linear_index in 0..self.layout.elements {
            temporary[linear_index] = input[linear_index] + half_dt * first_stage[linear_index];
        }
        rhs(
            core,
            &self.layout,
            &self.growth,
            &self.diffusion,
            dynamics,
            &self.temporary,
            &mut self.k1,
        );
        let second_stage = self.k1.as_slice().expect("scratch is standard contiguous");
        let output = self
            .output
            .as_slice_mut()
            .expect("scratch is standard contiguous");
        for linear_index in 0..self.layout.elements {
            output[linear_index] =
                input[linear_index] + time_step.get() * second_stage[linear_index];
        }
        Ok(self.output.view())
    }
}

fn rhs(
    core: &KernelCore,
    layout: &SpatialLayout,
    growth: &Array1<f64>,
    diffusion: &Diffusion,
    dynamics: SpatialDynamics,
    space: &ArrayD<f64>,
    output: &mut ArrayD<f64>,
) {
    let input = space
        .as_slice()
        .expect("spatial input is standard contiguous");
    let target = output
        .as_slice_mut()
        .expect("spatial scratch is standard contiguous");
    let interaction_matrix = core.interaction();
    let species_count = layout.species;

    for cell in 0..layout.cells {
        let base = cell * species_count;
        let mean_fitness = match dynamics {
            SpatialDynamics::Glv => 0.0,
            SpatialDynamics::Replicator => {
                let mut value = 0.0;
                for species in 0..species_count {
                    let mut interaction = 0.0;
                    for neighbor_species in 0..species_count {
                        interaction += interaction_matrix[(species, neighbor_species)]
                            * input[base + neighbor_species];
                    }
                    value += input[base + species] * (growth[species] + interaction);
                }
                value
            }
        };

        for species in 0..species_count {
            let center_index = base + species;
            let center = input[center_index];
            let mut interaction = 0.0;
            for neighbor_species in 0..species_count {
                interaction += interaction_matrix[(species, neighbor_species)]
                    * input[base + neighbor_species];
            }
            let mut laplacian = 0.0;
            for axis in 0..layout.spatial_dimensions {
                let axis_length = layout.shape[axis];
                let stride = layout.strides[axis];
                let coordinate = (base / stride) % axis_length;
                let plus_index = if coordinate + 1 < axis_length {
                    center_index + stride
                } else {
                    match diffusion.boundary {
                        Boundary::Periodic => center_index - (axis_length - 1) * stride,
                        Boundary::Neumann => center_index,
                    }
                };
                let minus_index = if coordinate > 0 {
                    center_index - stride
                } else {
                    match diffusion.boundary {
                        Boundary::Periodic => center_index + (axis_length - 1) * stride,
                        Boundary::Neumann => center_index,
                    }
                };
                laplacian += (input[plus_index] + input[minus_index] - 2.0 * center)
                    * diffusion.inverse_spacing_squared[axis];
            }
            let reaction = match dynamics {
                SpatialDynamics::Glv => center * (growth[species] + interaction),
                SpatialDynamics::Replicator => {
                    center * (growth[species] + interaction - mean_fitness)
                }
            };
            target[center_index] = reaction + diffusion.coefficients[species] * laplacian;
        }
    }
}
