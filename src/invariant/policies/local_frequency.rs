//! Per-cell spatial frequency normalization and aggregate refresh.

use crate::{AggregateAbundance, SpatialAbundance, TotalAbundance};

use crate::invariant::core::{
    InvariantPolicy, InvariantPolicyError, close, validate_abundance_values, validate_space_values,
    validate_species_and_cutoff,
};

/// Species-last spatial frequency invariant.
#[derive(Debug)]
pub struct LocalFrequencyInvariant {
    species: usize,
    cutoff: f64,
    totals: Vec<f64>,
}

impl LocalFrequencyInvariant {
    /// Creates a policy with fixed reusable aggregate scratch.
    pub fn new(species: usize, cutoff: f64) -> Result<Self, InvariantPolicyError> {
        validate_species_and_cutoff(species, cutoff)?;
        Ok(Self {
            species,
            cutoff,
            totals: vec![0.0; species],
        })
    }

    /// Returns the configured species count.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Returns the hard local cutoff.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the fixed reusable scratch length.
    pub fn scratch_len(&self) -> usize {
        self.totals.len()
    }

    fn spatial_layout<'a>(
        &self,
        space: &'a SpatialAbundance,
    ) -> Result<(&'a [f64], usize), InvariantPolicyError> {
        let space = space.as_ref().ok_or(InvariantPolicyError::SpaceRequired)?;
        let spatial_species = space
            .shape()
            .last()
            .copied()
            .ok_or(InvariantPolicyError::MissingSpeciesAxis)?;
        if spatial_species != self.species {
            return Err(InvariantPolicyError::SpaceSpeciesMismatch {
                expected: self.species,
                actual: spatial_species,
            });
        }
        let values = space.as_slice();
        let cells = values.len() / self.species;
        if cells == 0 {
            return Err(InvariantPolicyError::EmptySpatialDomain);
        }
        Ok((values, cells))
    }
}

impl InvariantPolicy for LocalFrequencyInvariant {
    type Error = InvariantPolicyError;

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
        total: &TotalAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.size() != self.species {
            return Err(InvariantPolicyError::SpeciesMismatch {
                expected: self.species,
                actual: abundance.size(),
            });
        }
        validate_abundance_values(abundance)?;
        let (values, cells) = self.spatial_layout(space)?;
        validate_space_values(values)?;
        let mut totals = vec![0.0; self.species];
        for (cell, values) in values.chunks_exact(self.species).enumerate() {
            let mut sum = 0.0;
            for (species, value) in values.iter().copied().enumerate() {
                sum += value;
                totals[species] += value;
            }
            if !close(1.0, sum) {
                return Err(InvariantPolicyError::SimplexViolation { cell, sum });
            }
        }
        let abundance = abundance.as_slice();
        for species in 0..self.species {
            let expected = totals[species] / cells as f64;
            if !close(expected, abundance[species]) {
                return Err(InvariantPolicyError::AggregateMismatch {
                    species,
                    expected,
                    actual: abundance[species],
                });
            }
        }
        if !total.is_finite() || !close(1.0, *total) {
            return Err(InvariantPolicyError::TotalMismatch {
                expected: 1.0,
                actual: *total,
            });
        }
        Ok(())
    }

    fn enforce(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        total: &mut TotalAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.size() != self.species {
            return Err(InvariantPolicyError::SpeciesMismatch {
                expected: self.species,
                actual: abundance.size(),
            });
        }
        let space = space.as_mut().ok_or(InvariantPolicyError::SpaceRequired)?;
        let spatial_species = space
            .shape()
            .last()
            .copied()
            .ok_or(InvariantPolicyError::MissingSpeciesAxis)?;
        if spatial_species != self.species {
            return Err(InvariantPolicyError::SpaceSpeciesMismatch {
                expected: self.species,
                actual: spatial_species,
            });
        }
        let values = space.as_mut_slice();
        let cells = values.len() / self.species;
        if cells == 0 {
            return Err(InvariantPolicyError::EmptySpatialDomain);
        }

        self.totals.fill(0.0);
        let uniform = 1.0 / self.species as f64;
        for cell in values.chunks_exact_mut(self.species) {
            let mut sum = 0.0;
            for value in cell.iter_mut() {
                if !value.is_finite() || *value <= 0.0 || *value < self.cutoff {
                    *value = 0.0;
                }
                sum += *value;
            }
            if sum > 0.0 {
                for (species, value) in cell.iter_mut().enumerate() {
                    *value /= sum;
                    self.totals[species] += *value;
                }
            } else {
                for (species, value) in cell.iter_mut().enumerate() {
                    *value = uniform;
                    self.totals[species] += uniform;
                }
            }
        }
        for (value, sum) in abundance.as_mut_slice().iter_mut().zip(&self.totals) {
            *value = *sum / cells as f64;
        }
        *total = 1.0;
        Ok(())
    }
}
