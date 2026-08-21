//! Shared composition contract for stochastic updates.

use std::error::Error;
use std::fmt;

use physics_in_parallel::prelude::basic::{
    RandType, RngConfig, RngConfigError, RngMethod, TensorRandError, TensorRandFiller,
};
use scientific_workflow::prelude::basics::{RngRecord, RngRecordError, StateError, SystemState};
use thiserror::Error as ThisError;

use crate::{ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TimeStep};

/// Fixed payload domain and scratch dimensions for a stochastic plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NoiseDomain {
    /// Noise applies directly to aggregate abundance.
    Aggregate {
        /// Species-vector length.
        species: usize,
    },
    /// Noise applies independently to each species-last spatial cell.
    Spatial {
        /// Exact spatial payload shape.
        shape: Box<[usize]>,
        /// Cached species-axis length.
        species: usize,
        /// Checked flattened element count.
        elements: usize,
    },
}

impl NoiseDomain {
    /// Creates a validated aggregate noise domain.
    pub fn aggregate(species: usize) -> Result<Self, NoisePluginError> {
        if species == 0 {
            return Err(NoisePluginError::EmptySpecies);
        }
        Ok(Self::Aggregate { species })
    }

    /// Creates a validated species-last spatial noise domain.
    pub fn spatial(shape: impl Into<Box<[usize]>>) -> Result<Self, NoisePluginError> {
        let shape = shape.into();
        let species = shape
            .last()
            .copied()
            .ok_or(NoisePluginError::MissingSpeciesAxis)?;
        if species == 0 {
            return Err(NoisePluginError::EmptySpecies);
        }
        let elements = shape
            .iter()
            .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension));
        let elements = elements.ok_or_else(|| NoisePluginError::ShapeOverflow {
            shape: shape.to_vec(),
        })?;
        if elements == 0 {
            return Err(NoisePluginError::EmptySpatialDomain);
        }
        Ok(Self::Spatial {
            shape,
            species,
            elements,
        })
    }

    /// Returns the fixed species dimension.
    pub const fn species(&self) -> usize {
        match self {
            Self::Aggregate { species } | Self::Spatial { species, .. } => *species,
        }
    }

    /// Returns the target payload element count.
    pub const fn elements(&self) -> usize {
        match self {
            Self::Aggregate { species } => *species,
            Self::Spatial { elements, .. } => *elements,
        }
    }

    /// Reports whether this domain targets spatial storage.
    pub const fn is_spatial(&self) -> bool {
        matches!(self, Self::Spatial { .. })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GaussianKind {
    Demographic,
    Proportional,
}

pub(crate) struct GaussianWorkspace {
    domain: NoiseDomain,
    sigma: f64,
    rng_record: RngRecord,
    filler: TensorRandFiller,
    proposed: Vec<f64>,
}

impl fmt::Debug for GaussianWorkspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GaussianWorkspace")
            .field("domain", &self.domain)
            .field("sigma", &self.sigma)
            .field("rng", &self.filler.rng_config())
            .field("proposed_len", &self.proposed.len())
            .finish_non_exhaustive()
    }
}

impl GaussianWorkspace {
    pub(crate) fn new(
        sigma: f64,
        rng: RngConfig,
        domain: NoiseDomain,
        namespace: &'static str,
    ) -> Result<Self, NoisePluginError> {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(NoisePluginError::InvalidSigma { value: sigma });
        }
        let elements = domain.elements();
        let rng = rng
            .resolve_for(
                namespace,
                RngMethod::ChaCha12,
                &[
                    RngMethod::Pcg64,
                    RngMethod::Pcg64Mcg,
                    RngMethod::SmallRng,
                    RngMethod::ChaCha8,
                    RngMethod::ChaCha12,
                    RngMethod::ChaCha20,
                ],
            )
            .map_err(NoisePluginError::RngConfig)?;
        let filler = TensorRandFiller::try_new(
            RandType::Normal {
                mean: 0.0,
                std: 1.0,
            },
            rng,
        )
        .map_err(NoisePluginError::TensorRng)?;
        let rng = filler.rng_config();
        let method = rng.method().expect("PiP resolves the noise RNG method");
        let mut parameters = serde_json::Map::new();
        parameters.insert(
            "distribution".to_owned(),
            serde_json::Value::from("standard_normal"),
        );
        parameters.insert(
            "sampling_layout".to_owned(),
            serde_json::Value::from("flat_species_last_v1"),
        );
        let rng_record = RngRecord::new(
            namespace,
            format!("{}+standard_normal", method.name()),
            method.version(),
            method.seed_encoding(),
            rng.encode_seed().expect("PiP resolves the noise seed"),
            Some(parameters),
        )
        .map_err(NoisePluginError::RngRecord)?;
        Ok(Self {
            domain,
            sigma,
            rng_record,
            filler,
            proposed: vec![0.0; elements],
        })
    }

    pub(crate) const fn sigma(&self) -> f64 {
        self.sigma
    }

    pub(crate) fn rng_config(&self) -> RngConfig {
        self.filler.rng_config()
    }

    pub(crate) const fn domain(&self) -> &NoiseDomain {
        &self.domain
    }

    pub(crate) const fn rng_record(&self) -> &RngRecord {
        &self.rng_record
    }

    pub(crate) fn scratch_capacity(&self) -> usize {
        self.proposed.capacity()
    }

    pub(crate) fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
        kind: GaussianKind,
    ) -> Result<(), NoisePluginError> {
        self.validate_target(abundance, space, kind)?;
        Ok(())
    }

    fn validate_target<'a>(
        &self,
        abundance: &'a AggregateAbundance,
        space: &'a SpatialAbundance,
        kind: GaussianKind,
    ) -> Result<&'a [f64], NoisePluginError> {
        if abundance.len() != self.domain.species() {
            return Err(NoisePluginError::SpeciesMismatch {
                expected: self.domain.species(),
                actual: abundance.len(),
            });
        }
        let target = self.target(abundance, space)?;
        let field = if self.domain.is_spatial() {
            SPACE_FIELD
        } else {
            ABUNDANCE_FIELD
        };
        for (cell, values) in target.chunks_exact(self.domain.species()).enumerate() {
            validate_cell(values, kind, field, cell, self.domain.species())?;
        }
        Ok(target)
    }

    pub(crate) fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        time_step: TimeStep,
        kind: GaussianKind,
    ) -> Result<(), NoisePluginError> {
        let target = self.validate_target(abundance, space, kind)?;
        let scale = self.sigma * time_step.get().sqrt();
        if !scale.is_finite() {
            return Err(NoisePluginError::ScaleOverflow {
                sigma: self.sigma,
                time_step: time_step.get(),
            });
        }
        if self.sigma == 0.0 {
            return Ok(());
        }
        self.filler
            .try_fill_slice(&mut self.proposed)
            .map_err(NoisePluginError::TensorRng)?;
        for (values, eta) in target
            .chunks_exact(self.domain.species())
            .zip(self.proposed.chunks_exact_mut(self.domain.species()))
        {
            match kind {
                GaussianKind::Demographic => {
                    apply_demographic(values, eta, scale);
                }
                GaussianKind::Proportional => {
                    apply_proportional(values, eta, scale);
                }
            }
        }
        match &self.domain {
            NoiseDomain::Aggregate { .. } => abundance
                .as_slice_mut()
                .expect("aggregate abundance uses standard contiguous storage")
                .copy_from_slice(&self.proposed),
            NoiseDomain::Spatial { .. } => space
                .as_mut()
                .expect("space presence was validated before noise commit")
                .as_slice_mut()
                .expect("space layout was validated before noise commit")
                .copy_from_slice(&self.proposed),
        }
        Ok(())
    }

    fn target<'a>(
        &self,
        abundance: &'a AggregateAbundance,
        space: &'a SpatialAbundance,
    ) -> Result<&'a [f64], NoisePluginError> {
        match &self.domain {
            NoiseDomain::Aggregate { .. } => {
                if space.is_some() {
                    return Err(NoisePluginError::UnexpectedSpace);
                }
                abundance
                    .as_slice()
                    .ok_or(NoisePluginError::NonStandardAbundanceLayout)
            }
            NoiseDomain::Spatial { shape, .. } => {
                let space = space.as_ref().ok_or(NoisePluginError::SpaceRequired)?;
                if space.shape() != shape.as_ref() {
                    return Err(NoisePluginError::SpaceShapeMismatch {
                        expected: shape.to_vec(),
                        actual: space.shape().to_vec(),
                    });
                }
                space
                    .as_slice()
                    .ok_or(NoisePluginError::NonStandardSpaceLayout)
            }
        }
    }
}

fn validate_cell(
    values: &[f64],
    kind: GaussianKind,
    field: &'static str,
    cell: usize,
    species: usize,
) -> Result<(), NoisePluginError> {
    let mut statistic = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        let linear_index = cell * species + index;
        if !value.is_finite() {
            return Err(NoisePluginError::NonFiniteInput {
                field,
                linear_index,
                value,
            });
        }
        if value < 0.0 {
            return Err(NoisePluginError::NegativeInput {
                field,
                linear_index,
                value,
            });
        }
        statistic += match kind {
            GaussianKind::Proportional => value,
            GaussianKind::Demographic => value.sqrt(),
        };
    }
    if statistic.is_finite() {
        Ok(())
    } else {
        Err(NoisePluginError::CellStatisticOverflow { cell })
    }
}

fn apply_proportional(values: &[f64], eta: &mut [f64], scale: f64) {
    let total = values.iter().sum::<f64>();
    let eta_bar = if total > 0.0 {
        values
            .iter()
            .zip(eta.iter())
            .map(|(value, eta)| value / total * *eta)
            .sum()
    } else {
        0.0
    };
    for (value, eta) in values.iter().zip(eta.iter_mut()) {
        let proposed = *value * (1.0 + scale * (*eta - eta_bar));
        *eta = if proposed.is_finite() && proposed > 0.0 {
            proposed
        } else {
            0.0
        };
    }
}

fn apply_demographic(values: &[f64], eta: &mut [f64], scale: f64) {
    let denominator = values.iter().map(|value| value.sqrt()).sum::<f64>();
    let eta_bar = if denominator > 0.0 {
        values
            .iter()
            .zip(eta.iter())
            .map(|(value, eta)| value.sqrt() * *eta)
            .sum::<f64>()
            / denominator
    } else {
        0.0
    };
    for (value, eta) in values.iter().zip(eta.iter_mut()) {
        let proposed = *value + scale * value.sqrt() * (*eta - eta_bar);
        *eta = if proposed.is_finite() && proposed > 0.0 {
            proposed
        } else {
            0.0
        };
    }
}

/// Configuration and state-domain failures for built-in noise algorithms.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum NoisePluginError {
    /// Universal RNG settings were incompatible with Gaussian noise.
    #[error("invalid noise RNG configuration: {0}")]
    RngConfig(#[source] RngConfigError),
    /// PiP rejected random filling or distribution configuration.
    #[error("noise random sampling failed: {0}")]
    TensorRng(#[source] TensorRandError),
    /// Workflow rejected the immutable RNG provenance declaration.
    #[error("invalid noise RNG record: {0}")]
    RngRecord(#[source] RngRecordError),
    /// Every noise target requires at least one species.
    #[error("noise species dimension must be greater than zero")]
    EmptySpecies,
    /// Spatial noise requires a species axis.
    #[error("spatial noise shape must contain a species axis")]
    MissingSpeciesAxis,
    /// A spatial shape cannot be represented as a platform-sized element count.
    #[error("spatial noise shape {shape:?} overflows its element count")]
    ShapeOverflow {
        /// Rejected shape.
        shape: Vec<usize>,
    },
    /// Spatial noise requires at least one cell.
    #[error("spatial noise domain must contain at least one cell")]
    EmptySpatialDomain,
    /// Gaussian strength must be finite and nonnegative.
    #[error("noise sigma must be finite and nonnegative, found {value}")]
    InvalidSigma {
        /// Rejected standard-deviation scale.
        value: f64,
    },
    /// Aggregate abundance disagrees with the configured species count.
    #[error("abundance length {actual} does not match noise species count {expected}")]
    SpeciesMismatch {
        /// Configured species count.
        expected: usize,
        /// State abundance length.
        actual: usize,
    },
    /// Aggregate noise cannot target a spatial state.
    #[error("aggregate noise requires `space = None`")]
    UnexpectedSpace,
    /// Spatial noise requires a populated spatial payload.
    #[error("spatial noise requires populated `space`")]
    SpaceRequired,
    /// Spatial state shape must remain fixed for scratch reuse.
    #[error("space shape {actual:?} does not match noise shape {expected:?}")]
    SpaceShapeMismatch {
        /// Configured shape.
        expected: Vec<usize>,
        /// State shape.
        actual: Vec<usize>,
    },
    /// Aggregate storage unexpectedly was not standard contiguous storage.
    #[error("aggregate abundance must use standard contiguous storage")]
    NonStandardAbundanceLayout,
    /// Spatial storage must be standard row-major for species-cell chunks.
    #[error("spatial abundance must use standard contiguous row-major storage")]
    NonStandardSpaceLayout,
    /// Noise input values must be finite after pre-noise invariant enforcement.
    #[error("{field} value at linear index {linear_index} is not finite: {value}")]
    NonFiniteInput {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// Noise input values must be nonnegative after invariant enforcement.
    #[error("{field} value at linear index {linear_index} is negative: {value}")]
    NegativeInput {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// A local mass statistic overflowed before sampling.
    #[error("noise input statistic overflowed in cell {cell}")]
    CellStatisticOverflow {
        /// Zero-based aggregate or spatial cell index.
        cell: usize,
    },
    /// Sigma and the time step produced a non-finite sampling scale.
    #[error("noise scale overflow for sigma {sigma} and time step {time_step}")]
    ScaleOverflow {
        /// Configured strength.
        sigma: f64,
        /// Validated time step.
        time_step: f64,
    },
}

/// One stochastic algorithm plugged into [`Noise`].
///
/// Implementations own RNG and reusable sampling scratch. Total synchronization
/// and final invariant enforcement remain engine responsibilities.
pub trait NoiseAlgorithm {
    /// Algorithm-specific validation or stochastic-update failure.
    type Error: Error + Send + Sync + 'static;

    /// Returns immutable RNG provenance, or explicitly declares deterministic behavior.
    fn rng_record(&self) -> Option<&RngRecord>;

    /// Reports whether applying this algorithm would leave every payload
    /// unchanged. The conservative default keeps custom plugins active.
    fn is_noop(&self) -> bool {
        false
    }

    /// Validates a state domain before evolution begins.
    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error>;

    /// Applies one stochastic update without enforcing final invariants.
    ///
    /// Implementations must complete fallible sampling and calculations in
    /// owned scratch before mutating payloads. Returning an error after a
    /// partial mutation violates the plugin contract.
    fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        time_step: TimeStep,
    ) -> Result<(), Self::Error>;
}

/// Shared stochastic component composed with one noise algorithm.
#[derive(Debug)]
pub struct Noise<N> {
    algorithm: N,
}

impl<N> Noise<N> {
    /// Creates a stochastic component from one algorithm.
    pub const fn new(algorithm: N) -> Self {
        Self { algorithm }
    }

    /// Borrows the algorithm, RNG, and scratch immutably.
    pub const fn algorithm(&self) -> &N {
        &self.algorithm
    }

    /// Borrows the algorithm, RNG, and scratch mutably.
    pub const fn algorithm_mut(&mut self) -> &mut N {
        &mut self.algorithm
    }

    /// Returns the algorithm by ownership transfer.
    pub fn into_algorithm(self) -> N {
        self.algorithm
    }
}

impl<N> Noise<N>
where
    N: NoiseAlgorithm,
{
    /// Reports whether the selected algorithm can be skipped for this run.
    pub fn is_noop(&self) -> bool {
        self.algorithm.is_noop()
    }

    /// Returns the algorithm's immutable RNG provenance when it is stochastic.
    pub fn rng_record(&self) -> Option<&RngRecord> {
        self.algorithm.rng_record()
    }

    /// Validates canonical state payloads without mutating them.
    pub fn validate_state(&self, state: &SystemState) -> Result<(), NoiseStepError<N::Error>> {
        let (abundance, space) = state
            .borrow_payloads::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(NoiseStepError::State)?;
        self.algorithm
            .validate(abundance, space)
            .map_err(NoiseStepError::Algorithm)
    }

    /// Applies one stochastic update without advancing state time.
    pub fn apply(
        &mut self,
        state: &mut SystemState,
        time_step: TimeStep,
    ) -> Result<(), NoiseStepError<N::Error>> {
        let (abundance, space) = state
            .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(NoiseStepError::State)?;
        self.algorithm
            .apply(abundance, space, time_step)
            .map_err(NoiseStepError::Algorithm)
    }
}

/// Failure while validating or applying a stochastic plugin.
#[derive(Debug)]
#[non_exhaustive]
pub enum NoiseStepError<E> {
    /// Canonical Workflow payload access failed.
    State(StateError),
    /// The selected noise algorithm rejected the state or update.
    Algorithm(E),
}

impl<E> fmt::Display for NoiseStepError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => fmt::Display::fmt(error, formatter),
            Self::Algorithm(error) => write!(formatter, "noise algorithm failed: {error}"),
        }
    }
}

impl<E> Error for NoiseStepError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::Algorithm(error) => Some(error),
        }
    }
}
