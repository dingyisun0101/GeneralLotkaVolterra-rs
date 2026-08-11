//! Shared composition contract for stochastic updates.

use std::error::Error;
use std::fmt;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};
use scientific_workflow::system_state::{StateError, SystemState};
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
    seed: u64,
    rng: ChaCha12Rng,
    normal: StandardNormal,
    eta: Vec<f64>,
    proposed: Vec<f64>,
}

impl fmt::Debug for GaussianWorkspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GaussianWorkspace")
            .field("domain", &self.domain)
            .field("sigma", &self.sigma)
            .field("seed", &self.seed)
            .field("eta_len", &self.eta.len())
            .field("proposed_len", &self.proposed.len())
            .finish_non_exhaustive()
    }
}

impl GaussianWorkspace {
    pub(crate) fn new(
        sigma: f64,
        seed: u64,
        domain: NoiseDomain,
    ) -> Result<Self, NoisePluginError> {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(NoisePluginError::InvalidSigma { value: sigma });
        }
        let species = domain.species();
        let elements = domain.elements();
        Ok(Self {
            domain,
            sigma,
            seed,
            rng: ChaCha12Rng::seed_from_u64(seed),
            normal: StandardNormal,
            eta: vec![0.0; species],
            proposed: vec![0.0; elements],
        })
    }

    pub(crate) const fn sigma(&self) -> f64 {
        self.sigma
    }

    pub(crate) const fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) const fn domain(&self) -> &NoiseDomain {
        &self.domain
    }

    pub(crate) fn scratch_capacities(&self) -> (usize, usize) {
        (self.eta.capacity(), self.proposed.capacity())
    }

    pub(crate) fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), NoisePluginError> {
        if abundance.len() != self.domain.species() {
            return Err(NoisePluginError::SpeciesMismatch {
                expected: self.domain.species(),
                actual: abundance.len(),
            });
        }
        validate_noise_values(ABUNDANCE_FIELD, abundance.iter().copied())?;
        let target = self.target(abundance, space)?;
        validate_noise_values(
            if self.domain.is_spatial() {
                SPACE_FIELD
            } else {
                ABUNDANCE_FIELD
            },
            target.iter().copied(),
        )
    }

    pub(crate) fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        time_step: TimeStep,
        kind: GaussianKind,
    ) -> Result<(), NoisePluginError> {
        self.validate(abundance, space)?;
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
        let target = self.target(abundance, space)?;
        self.proposed.copy_from_slice(target);
        for (cell_index, cell) in self
            .proposed
            .chunks_exact(self.domain.species())
            .enumerate()
        {
            preflight_cell(cell, kind, cell_index)?;
        }
        for cell in self.proposed.chunks_exact_mut(self.domain.species()) {
            for eta in &mut self.eta {
                *eta = self.normal.sample(&mut self.rng);
            }
            match kind {
                GaussianKind::Demographic => {
                    apply_demographic(cell, &self.eta, scale);
                }
                GaussianKind::Proportional => {
                    apply_proportional(cell, &self.eta, scale);
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

fn preflight_cell(values: &[f64], kind: GaussianKind, cell: usize) -> Result<(), NoisePluginError> {
    let statistic = match kind {
        GaussianKind::Proportional => values.iter().sum::<f64>(),
        GaussianKind::Demographic => values.iter().map(|value| value.sqrt()).sum::<f64>(),
    };
    if statistic.is_finite() {
        Ok(())
    } else {
        Err(NoisePluginError::CellStatisticOverflow { cell })
    }
}

fn apply_proportional(values: &mut [f64], eta: &[f64], scale: f64) {
    let total = values.iter().sum::<f64>();
    let eta_bar = if total > 0.0 {
        values
            .iter()
            .zip(eta)
            .map(|(value, eta)| value / total * eta)
            .sum()
    } else {
        0.0
    };
    for (value, eta) in values.iter_mut().zip(eta) {
        let current = *value;
        let proposed = current * (1.0 + scale * (*eta - eta_bar));
        *value = if proposed.is_finite() && proposed > 0.0 {
            proposed
        } else {
            0.0
        };
    }
}

fn apply_demographic(values: &mut [f64], eta: &[f64], scale: f64) {
    let denominator = values.iter().map(|value| value.sqrt()).sum::<f64>();
    let eta_bar = if denominator > 0.0 {
        values
            .iter()
            .zip(eta)
            .map(|(value, eta)| value.sqrt() * eta)
            .sum::<f64>()
            / denominator
    } else {
        0.0
    };
    for (value, eta) in values.iter_mut().zip(eta) {
        let current = *value;
        let proposed = current + scale * current.sqrt() * (*eta - eta_bar);
        *value = if proposed.is_finite() && proposed > 0.0 {
            proposed
        } else {
            0.0
        };
    }
}

fn validate_noise_values(
    field: &'static str,
    values: impl Iterator<Item = f64>,
) -> Result<(), NoisePluginError> {
    for (linear_index, value) in values.enumerate() {
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
    }
    Ok(())
}

/// Configuration and state-domain failures for built-in noise algorithms.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum NoisePluginError {
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
