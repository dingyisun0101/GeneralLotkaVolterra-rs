//! Spatial population feasibility, capacity, and aggregate synchronization.

use crate::{AggregateAbundance, SpatialAbundance, TotalAbundance};

use crate::invariant::core::{
    InvariantPolicy, InvariantPolicyError, close, validate_abundance_values, validate_space_values,
    validate_species_and_cutoff,
};

/// Species-last spatial population invariant.
///
/// `total` deliberately uses the rounded exact aggregate sum. Spatial
/// coefficients and per-species aggregates remain unrounded.
#[derive(Debug)]
pub struct PopulationInvariant {
    species: usize,
    cutoff: f64,
    carrying_capacity: Option<f64>,
    totals: Vec<f64>,
}

impl PopulationInvariant {
    /// Creates a validated population policy with fixed reusable scratch.
    pub fn new(
        species: usize,
        cutoff: f64,
        carrying_capacity: Option<f64>,
    ) -> Result<Self, InvariantPolicyError> {
        validate_species_and_cutoff(species, cutoff)?;
        if let Some(value) = carrying_capacity
            && (!value.is_finite() || value < 0.0)
        {
            return Err(InvariantPolicyError::InvalidCarryingCapacity { value });
        }
        Ok(Self {
            species,
            cutoff,
            carrying_capacity,
            totals: vec![0.0; species],
        })
    }

    /// Returns the configured species count.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Returns the hard population cutoff.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the optional global carrying capacity.
    pub const fn carrying_capacity(&self) -> Option<f64> {
        self.carrying_capacity
    }

    /// Returns the fixed reusable scratch length.
    pub fn scratch_len(&self) -> usize {
        self.totals.len()
    }

    fn spatial_layout<'a>(
        &self,
        space: &'a SpatialAbundance,
    ) -> Result<&'a [f64], InvariantPolicyError> {
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
        let values = space
            .as_slice()
            .ok_or(InvariantPolicyError::NonStandardSpaceLayout)?;
        if values.is_empty() {
            return Err(InvariantPolicyError::EmptySpatialDomain);
        }
        Ok(values)
    }
}

impl InvariantPolicy for PopulationInvariant {
    type Error = InvariantPolicyError;

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
        total: &TotalAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.len() != self.species {
            return Err(InvariantPolicyError::SpeciesMismatch {
                expected: self.species,
                actual: abundance.len(),
            });
        }
        validate_abundance_values(abundance)?;
        let values = self.spatial_layout(space)?;
        validate_space_values(values)?;
        for species in 0..self.species {
            let expected = values
                .chunks_exact(self.species)
                .map(|cell| cell[species])
                .sum::<f64>();
            if !close(expected, abundance[species]) {
                return Err(InvariantPolicyError::AggregateMismatch {
                    species,
                    expected,
                    actual: abundance[species],
                });
            }
        }
        let exact_total = abundance.sum();
        if let Some(capacity) = self.carrying_capacity
            && exact_total > capacity
            && !close(capacity, exact_total)
        {
            return Err(InvariantPolicyError::CarryingCapacityExceeded {
                total: exact_total,
                capacity,
            });
        }
        let expected_total = exact_total.round().max(0.0);
        if !total.is_finite() || !close(expected_total, *total) {
            return Err(InvariantPolicyError::TotalMismatch {
                expected: expected_total,
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
        if abundance.len() != self.species {
            return Err(InvariantPolicyError::SpeciesMismatch {
                expected: self.species,
                actual: abundance.len(),
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
        let values = space
            .as_slice_mut()
            .ok_or(InvariantPolicyError::NonStandardSpaceLayout)?;
        if values.is_empty() {
            return Err(InvariantPolicyError::EmptySpatialDomain);
        }

        self.totals.fill(0.0);
        for cell in values.chunks_exact_mut(self.species) {
            for (species, value) in cell.iter_mut().enumerate() {
                if !value.is_finite() || *value <= 0.0 || *value < self.cutoff {
                    *value = 0.0;
                }
                self.totals[species] += *value;
            }
        }
        let mut exact_total = self.totals.iter().sum::<f64>();
        if let Some(capacity) = self.carrying_capacity {
            if capacity == 0.0 {
                values.fill(0.0);
                self.totals.fill(0.0);
                exact_total = 0.0;
            } else if exact_total > capacity {
                let scale = capacity / exact_total;
                values.iter_mut().for_each(|value| *value *= scale);
                self.totals.iter_mut().for_each(|value| *value *= scale);
                exact_total = capacity;
            }
        }
        for (value, sum) in abundance.iter_mut().zip(&self.totals) {
            *value = *sum;
        }
        *total = exact_total.round().max(0.0);
        Ok(())
    }
}
