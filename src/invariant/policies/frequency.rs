//! Aggregate frequency cutoff and simplex normalization.

use crate::tensor_compat::DenseTensorExt;
use crate::{AggregateAbundance, SpatialAbundance, TotalAbundance};

use crate::invariant::core::{
    InvariantPolicy, InvariantPolicyError, close, validate_abundance_values,
    validate_species_and_cutoff,
};

/// Non-spatial frequency invariant with a hard abundance cutoff.
#[derive(Clone, Copy, Debug)]
pub struct FrequencyInvariant {
    species: usize,
    cutoff: f64,
}

impl FrequencyInvariant {
    /// Creates a validated aggregate frequency policy.
    pub fn new(species: usize, cutoff: f64) -> Result<Self, InvariantPolicyError> {
        validate_species_and_cutoff(species, cutoff)?;
        Ok(Self { species, cutoff })
    }

    /// Returns the configured species count.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Returns the hard cutoff applied before normalization.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl InvariantPolicy for FrequencyInvariant {
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
        if space.is_some() {
            return Err(InvariantPolicyError::UnexpectedSpace);
        }
        validate_abundance_values(abundance)?;
        let sum = abundance.sum_serial();
        if !close(1.0, sum) {
            return Err(InvariantPolicyError::SimplexViolation { cell: 0, sum });
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
        if space.is_some() {
            return Err(InvariantPolicyError::UnexpectedSpace);
        }
        for value in abundance.as_mut_slice() {
            if !value.is_finite() || *value <= 0.0 || *value < self.cutoff {
                *value = 0.0;
            }
        }
        let sum = abundance.sum_serial();
        if sum > 0.0 {
            abundance.map_in_place(|value| value / sum);
        } else {
            abundance.fill(1.0 / self.species as f64);
        }
        *total = 1.0;
        Ok(())
    }
}
