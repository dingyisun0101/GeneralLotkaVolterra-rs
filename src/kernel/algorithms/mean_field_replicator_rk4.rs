//! Allocation-free mean-field replicator RK4 evolution.

use physics_in_parallel::prelude::basic::Tensor;

use crate::kernel::core::{
    KernelAlgorithm, KernelCore, KernelResidual, KernelStateView, KernelUpdate,
};
use crate::{ABUNDANCE_FIELD, TimeStep};

use super::{KernelAlgorithmError, validate_values};

/// Classical RK4 integration of mean-field replicator dynamics.
#[derive(Debug)]
pub struct MeanFieldReplicatorRk4 {
    growth: Tensor<f64>,
    k1: Tensor<f64>,
    k2: Tensor<f64>,
    k3: Tensor<f64>,
    k4: Tensor<f64>,
    temporary: Tensor<f64>,
    interaction: Tensor<f64>,
    output: Tensor<f64>,
}

impl MeanFieldReplicatorRk4 {
    /// Creates fixed RK4 storage for one validated growth vector.
    pub fn new(growth: Tensor<f64>) -> Result<Self, KernelAlgorithmError> {
        let species = growth.size();
        if species == 0 {
            return Err(KernelAlgorithmError::EmptySpecies);
        }
        if growth.rank() != 1 {
            return Err(KernelAlgorithmError::CoefficientRank {
                field: "growth",
                actual: growth.rank(),
            });
        }
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
            growth,
            k1: Tensor::zeros(&[species]),
            k2: Tensor::zeros(&[species]),
            k3: Tensor::zeros(&[species]),
            k4: Tensor::zeros(&[species]),
            temporary: Tensor::zeros(&[species]),
            interaction: Tensor::zeros(&[species]),
            output: Tensor::zeros(&[species]),
        })
    }

    /// Creates a zero-growth RK4 algorithm.
    pub fn zero_growth(species: usize) -> Result<Self, KernelAlgorithmError> {
        Self::new(Tensor::zeros(&[species]))
    }

    /// Borrows the immutable growth vector.
    pub const fn growth(&self) -> &Tensor<f64> {
        &self.growth
    }

    /// Returns every fixed scratch length for allocation-reuse checks.
    pub fn scratch_lengths(&self) -> [usize; 7] {
        [
            self.k1.size(),
            self.k2.size(),
            self.k3.size(),
            self.k4.size(),
            self.temporary.size(),
            self.interaction.size(),
            self.output.size(),
        ]
    }
}

impl KernelAlgorithm for MeanFieldReplicatorRk4 {
    type Error = KernelAlgorithmError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        self.validate_layout(core, state)?;
        validate_values(
            ABUNDANCE_FIELD,
            state.abundance().as_slice().iter().copied(),
        )
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
        self.temporary
            .zip_from(abundance, &self.k1, |value, rate| value + half_dt * rate)
            .expect("RK4 scratch shapes are fixed at construction");
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k2,
        )?;
        self.temporary
            .zip_from(abundance, &self.k2, |value, rate| value + half_dt * rate)
            .expect("RK4 scratch shapes are fixed at construction");
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k3,
        )?;
        self.temporary
            .zip_from(abundance, &self.k3, |value, rate| value + dt * rate)
            .expect("RK4 scratch shapes are fixed at construction");
        rhs(
            core,
            &self.growth,
            &self.temporary,
            &mut self.interaction,
            &mut self.k4,
        )?;
        let dt_over_six = dt / 6.0;
        let abundance_values = abundance.as_slice();
        let k1 = self.k1.as_slice();
        let k2 = self.k2.as_slice();
        let k3 = self.k3.as_slice();
        let k4 = self.k4.as_slice();
        self.output.for_each_chunk_mut(1, |species, output| {
            let increment =
                dt_over_six * (k1[species] + 2.0 * k2[species] + 2.0 * k3[species] + k4[species]);
            let value = abundance_values[species] + increment;
            output[0] = if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            };
        });
        Ok(KernelUpdate::abundance(&self.output))
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
        Ok(Some(KernelResidual::Abundance(&self.k1)))
    }
}

impl MeanFieldReplicatorRk4 {
    fn validate_layout(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), KernelAlgorithmError> {
        let species = self.growth.size();
        if core.species() != species {
            return Err(KernelAlgorithmError::CoreSpeciesMismatch {
                expected: species,
                actual: core.species(),
            });
        }
        if state.abundance().shape() != [species] {
            return Err(KernelAlgorithmError::SpeciesMismatch {
                expected: species,
                actual: state.abundance().size(),
            });
        }
        if state.space().is_some() {
            return Err(KernelAlgorithmError::UnexpectedSpace);
        }
        Ok(())
    }
}

fn rhs(
    core: &KernelCore,
    growth: &Tensor<f64>,
    abundance: &Tensor<f64>,
    interaction: &mut Tensor<f64>,
    output: &mut Tensor<f64>,
) -> Result<(), KernelAlgorithmError> {
    core.apply_interaction(abundance.as_slice(), interaction.as_mut_slice())?;
    let mut mean_fitness = 0.0;
    let abundance = abundance.as_slice();
    let growth = growth.as_slice();
    let interaction = interaction.as_slice();
    for species in 0..abundance.len() {
        mean_fitness += abundance[species] * (growth[species] + interaction[species]);
    }
    output.for_each_chunk_mut(1, |species, output| {
        output[0] = abundance[species] * (growth[species] + interaction[species] - mean_fitness);
    });
    Ok(())
}
