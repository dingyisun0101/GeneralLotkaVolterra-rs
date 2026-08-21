//! Allocation-free mean-field replicator RK4 evolution.

use ndarray::Array1;

use crate::kernel::core::{
    KernelAlgorithm, KernelCore, KernelResidual, KernelStateView, KernelUpdate,
};
use crate::{ABUNDANCE_FIELD, TimeStep};

use super::{KernelAlgorithmError, validate_values};

/// Classical RK4 integration of mean-field replicator dynamics.
#[derive(Debug)]
pub struct MeanFieldReplicatorRk4 {
    growth: Array1<f64>,
    k1: Array1<f64>,
    k2: Array1<f64>,
    k3: Array1<f64>,
    k4: Array1<f64>,
    temporary: Array1<f64>,
    interaction: Array1<f64>,
    output: Array1<f64>,
}

impl MeanFieldReplicatorRk4 {
    /// Creates fixed RK4 storage for one validated growth vector.
    pub fn new(growth: Array1<f64>) -> Result<Self, KernelAlgorithmError> {
        let species = growth.len();
        if species == 0 {
            return Err(KernelAlgorithmError::EmptySpecies);
        }
        if let Some((index, value)) = growth
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KernelAlgorithmError::NonFiniteGrowth { index, value });
        }
        Ok(Self {
            growth,
            k1: Array1::zeros(species),
            k2: Array1::zeros(species),
            k3: Array1::zeros(species),
            k4: Array1::zeros(species),
            temporary: Array1::zeros(species),
            interaction: Array1::zeros(species),
            output: Array1::zeros(species),
        })
    }

    /// Creates a zero-growth RK4 algorithm.
    pub fn zero_growth(species: usize) -> Result<Self, KernelAlgorithmError> {
        Self::new(Array1::zeros(species))
    }

    /// Borrows the immutable growth vector.
    pub const fn growth(&self) -> &Array1<f64> {
        &self.growth
    }

    /// Returns every fixed scratch length for allocation-reuse checks.
    pub fn scratch_lengths(&self) -> [usize; 7] {
        [
            self.k1.len(),
            self.k2.len(),
            self.k3.len(),
            self.k4.len(),
            self.temporary.len(),
            self.interaction.len(),
            self.output.len(),
        ]
    }
}

impl KernelAlgorithm for MeanFieldReplicatorRk4 {
    type Error = KernelAlgorithmError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        self.validate_layout(core, state)?;
        validate_values(ABUNDANCE_FIELD, state.abundance().iter().copied())
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        self.validate_layout(core, state)?;
        let abundance = state.abundance();
        let dt = time_step.get();
        let half_dt = 0.5 * dt;

        rhs(
            core,
            &self.growth,
            abundance,
            &mut self.interaction,
            &mut self.k1,
        )?;
        for species in 0..abundance.len() {
            self.temporary[species] = abundance[species] + half_dt * self.k1[species];
        }
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k2,
        )?;
        for species in 0..abundance.len() {
            self.temporary[species] = abundance[species] + half_dt * self.k2[species];
        }
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k3,
        )?;
        for species in 0..abundance.len() {
            self.temporary[species] = abundance[species] + dt * self.k3[species];
        }
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k4,
        )?;
        let dt_over_six = dt / 6.0;
        for species in 0..abundance.len() {
            let increment = dt_over_six
                * (self.k1[species]
                    + 2.0 * self.k2[species]
                    + 2.0 * self.k3[species]
                    + self.k4[species]);
            let value = abundance[species] + increment;
            self.output[species] = if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            };
        }
        Ok(KernelUpdate::abundance(self.output.view()))
    }

    fn residual<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<Option<KernelResidual<'algorithm>>, Self::Error> {
        self.validate_layout(core, state)?;
        rhs(
            core,
            &self.growth,
            state.abundance(),
            &mut self.interaction,
            &mut self.k1,
        )?;
        Ok(Some(KernelResidual::Abundance(self.k1.view())))
    }
}

impl MeanFieldReplicatorRk4 {
    fn validate_layout(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), KernelAlgorithmError> {
        let species = self.growth.len();
        if core.species() != species {
            return Err(KernelAlgorithmError::CoreSpeciesMismatch {
                expected: species,
                actual: core.species(),
            });
        }
        if state.abundance().len() != species {
            return Err(KernelAlgorithmError::SpeciesMismatch {
                expected: species,
                actual: state.abundance().len(),
            });
        }
        if state.space().is_some() {
            return Err(KernelAlgorithmError::UnexpectedSpace);
        }
        if state.abundance().as_slice().is_none() {
            return Err(KernelAlgorithmError::NonStandardLayout);
        }
        Ok(())
    }
}

fn rhs(
    core: &KernelCore,
    growth: &Array1<f64>,
    abundance: &Array1<f64>,
    interaction: &mut Array1<f64>,
    output: &mut Array1<f64>,
) -> Result<(), KernelAlgorithmError> {
    core.apply_interaction(
        abundance
            .as_slice()
            .ok_or(KernelAlgorithmError::NonStandardLayout)?,
        interaction
            .as_slice_mut()
            .expect("RK4 interaction scratch is standard contiguous"),
    )?;
    let mut mean_fitness = 0.0;
    for species in 0..abundance.len() {
        mean_fitness += abundance[species] * (growth[species] + interaction[species]);
    }
    for species in 0..abundance.len() {
        output[species] =
            abundance[species] * (growth[species] + interaction[species] - mean_fitness);
    }
    Ok(())
}
