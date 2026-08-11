//! Shared ownership and behavior for deterministic kernels.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use ndarray::Array2;
use scientific_workflow::system_state::{StateError, SystemState};
use thiserror::Error as ThisError;

use crate::{ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance};

/// Invalid shared interaction configuration or matrix application.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum KernelCoreError {
    /// A model cannot evolve an empty species domain.
    #[error("interaction matrix must contain at least one species")]
    EmptySpecies,
    /// Interaction matrices are square by scientific contract.
    #[error("interaction matrix must be square, found {rows} rows and {columns} columns")]
    NonSquare {
        /// Matrix row count.
        rows: usize,
        /// Matrix column count.
        columns: usize,
    },
    /// Every coefficient must be finite before a run begins.
    #[error("interaction matrix entry ({row}, {column}) is not finite: {value}")]
    NonFiniteEntry {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        column: usize,
        /// Rejected coefficient.
        value: f64,
    },
    /// Matrix input must match the validated species count.
    #[error("interaction input length {actual} does not match species count {expected}")]
    InputLength {
        /// Validated species count.
        expected: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// Matrix output must match the validated species count.
    #[error("interaction output length {actual} does not match species count {expected}")]
    OutputLength {
        /// Validated species count.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
}

/// Immutable interaction facilities shared by every deterministic algorithm.
///
/// Source resolution and durable provenance are added by the interaction-source
/// stage. This foundation already establishes validated immutable ownership and
/// zero-allocation application, so algorithm implementations never duplicate
/// matrix checks or multiplication.
#[derive(Clone, Debug)]
pub struct KernelCore {
    interaction: Arc<Array2<f64>>,
    species: usize,
}

impl KernelCore {
    /// Validates and takes shared ownership of an interaction matrix.
    pub fn new(interaction: Array2<f64>) -> Result<Self, KernelCoreError> {
        Self::from_shared(Arc::new(interaction))
    }

    /// Validates an already-shared interaction matrix without cloning values.
    pub fn from_shared(interaction: Arc<Array2<f64>>) -> Result<Self, KernelCoreError> {
        let [rows, columns] = interaction
            .shape()
            .try_into()
            .expect("Array2 always has exactly two dimensions");
        if rows == 0 {
            return Err(KernelCoreError::EmptySpecies);
        }
        if rows != columns {
            return Err(KernelCoreError::NonSquare { rows, columns });
        }
        if let Some(((row, column), value)) = interaction
            .indexed_iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KernelCoreError::NonFiniteEntry {
                row,
                column,
                value: *value,
            });
        }
        Ok(Self {
            interaction,
            species: rows,
        })
    }

    /// Returns the validated species dimension.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Borrows the exact immutable interaction matrix.
    pub fn interaction(&self) -> &Array2<f64> {
        &self.interaction
    }

    /// Clones only the shared matrix handle, never its coefficient allocation.
    pub fn shared_interaction(&self) -> Arc<Array2<f64>> {
        Arc::clone(&self.interaction)
    }

    /// Computes `output = interaction * input` without allocating.
    pub fn apply_interaction(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), KernelCoreError> {
        if input.len() != self.species {
            return Err(KernelCoreError::InputLength {
                expected: self.species,
                actual: input.len(),
            });
        }
        if output.len() != self.species {
            return Err(KernelCoreError::OutputLength {
                expected: self.species,
                actual: output.len(),
            });
        }
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut value = 0.0;
            for (column, input_value) in input.iter().copied().enumerate() {
                value += self.interaction[(row, column)] * input_value;
            }
            *output_value = value;
        }
        Ok(())
    }
}

/// One deterministic numerical algorithm plugged into [`Kernel`].
///
/// Implementations own their integration scratch. They may mutate abundance
/// and optional spatial abundance, but total synchronization, noise, recording,
/// progress, and simulation-time advancement belong to other layers.
pub trait KernelAlgorithm {
    /// Algorithm-specific validation or evolution failure.
    type Error: Error + Send + Sync + 'static;

    /// Validates a state domain before evolution begins.
    fn validate(
        &self,
        core: &KernelCore,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error>;

    /// Performs exactly one deterministic numerical transition.
    fn step(
        &mut self,
        core: &KernelCore,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        physical_time_increment: f64,
    ) -> Result<(), Self::Error>;
}

/// Shared deterministic kernel composed with one algorithm implementation.
#[derive(Debug)]
pub struct Kernel<A> {
    core: KernelCore,
    algorithm: A,
}

impl<A> Kernel<A> {
    /// Creates a kernel from validated shared facilities and an algorithm.
    pub const fn new(core: KernelCore, algorithm: A) -> Self {
        Self { core, algorithm }
    }

    /// Borrows shared kernel facilities.
    pub const fn core(&self) -> &KernelCore {
        &self.core
    }

    /// Borrows the algorithm and its scratch immutably.
    pub const fn algorithm(&self) -> &A {
        &self.algorithm
    }

    /// Borrows the algorithm and its scratch mutably.
    pub const fn algorithm_mut(&mut self) -> &mut A {
        &mut self.algorithm
    }

    /// Returns the shared facilities and algorithm by ownership transfer.
    pub fn into_parts(self) -> (KernelCore, A) {
        (self.core, self.algorithm)
    }
}

impl<A> Kernel<A>
where
    A: KernelAlgorithm,
{
    /// Validates canonical state payloads without mutating them.
    pub fn validate_state(&self, state: &SystemState) -> Result<(), KernelStepError<A::Error>> {
        let (abundance, space) = state
            .borrow_payloads::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(KernelStepError::State)?;
        self.algorithm
            .validate(&self.core, abundance, space)
            .map_err(KernelStepError::Algorithm)
    }

    /// Applies one deterministic transition without advancing state time.
    pub fn step(
        &mut self,
        state: &mut SystemState,
        physical_time_increment: f64,
    ) -> Result<(), KernelStepError<A::Error>> {
        if !physical_time_increment.is_finite() {
            return Err(KernelStepError::InvalidPhysicalTimeIncrement {
                value: physical_time_increment,
            });
        }
        let (abundance, space) = state
            .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(KernelStepError::State)?;
        self.algorithm
            .step(&self.core, abundance, space, physical_time_increment)
            .map_err(KernelStepError::Algorithm)
    }
}

/// Failure while validating or applying a deterministic kernel.
#[derive(Debug)]
#[non_exhaustive]
pub enum KernelStepError<E> {
    /// Canonical Workflow payload access failed.
    State(StateError),
    /// The physical-time increment was not finite.
    InvalidPhysicalTimeIncrement {
        /// Rejected increment.
        value: f64,
    },
    /// The selected algorithm rejected the state or transition.
    Algorithm(E),
}

impl<E> fmt::Display for KernelStepError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => fmt::Display::fmt(error, formatter),
            Self::InvalidPhysicalTimeIncrement { value } => {
                write!(
                    formatter,
                    "kernel physical-time increment is not finite: {value}"
                )
            }
            Self::Algorithm(error) => write!(formatter, "kernel algorithm failed: {error}"),
        }
    }
}

impl<E> Error for KernelStepError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::InvalidPhysicalTimeIncrement { .. } => None,
            Self::Algorithm(error) => Some(error),
        }
    }
}
