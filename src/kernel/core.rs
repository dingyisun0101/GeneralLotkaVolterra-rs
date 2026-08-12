//! Shared ownership and behavior for deterministic kernels.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use ndarray::{ArrayD, ArrayView1, ArrayViewD};
use physics_in_parallel::math::prelude::{DenseMatrix, MatrixError};
use scientific_workflow::system_state::{StateError, SystemState};
use thiserror::Error as ThisError;

use crate::interaction::{InteractionMatrix, InteractionProvenance};
use crate::{ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TimeStep};

/// Invalid shared matrix application.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum KernelCoreError {
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
#[derive(Clone, Debug)]
pub struct KernelCore {
    interaction: InteractionMatrix,
    species: usize,
}

impl KernelCore {
    /// Takes ownership of an already-resolved and validated interaction matrix.
    pub fn new(interaction: InteractionMatrix) -> Self {
        let species = interaction.species();
        Self {
            interaction,
            species,
        }
    }

    /// Returns the validated species dimension.
    pub const fn species(&self) -> usize {
        self.species
    }

    /// Borrows the exact immutable interaction matrix.
    pub fn interaction(&self) -> &DenseMatrix<f64> {
        self.interaction.values()
    }

    /// Clones only the shared matrix handle, never its coefficient allocation.
    pub fn shared_interaction(&self) -> Arc<DenseMatrix<f64>> {
        self.interaction.shared_values()
    }

    /// Borrows complete matrix-source provenance.
    pub const fn provenance(&self) -> &InteractionProvenance {
        self.interaction.provenance()
    }

    /// Borrows the resolved interaction wrapper.
    pub const fn resolved_interaction(&self) -> &InteractionMatrix {
        &self.interaction
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
        self.interaction
            .mul_vector_into(input, output)
            .map_err(|error| match error {
                MatrixError::InputLength { expected, actual } => {
                    KernelCoreError::InputLength { expected, actual }
                }
                MatrixError::OutputLength { expected, actual } => {
                    KernelCoreError::OutputLength { expected, actual }
                }
                _ => unreachable!("matrix was validated before kernel construction"),
            })
    }
}

/// Read-only canonical abundance payloads presented to a kernel algorithm.
///
/// This view deliberately omits Workflow time, total abundance, payload-slot
/// ownership, and every recording concern.
#[derive(Clone, Copy, Debug)]
pub struct KernelStateView<'a> {
    abundance: &'a AggregateAbundance,
    space: Option<&'a ArrayD<f64>>,
}

impl<'a> KernelStateView<'a> {
    fn new(abundance: &'a AggregateAbundance, space: &'a SpatialAbundance) -> Self {
        Self {
            abundance,
            space: space.as_ref(),
        }
    }

    /// Borrows aggregate abundance.
    pub const fn abundance(self) -> &'a AggregateAbundance {
        self.abundance
    }

    /// Borrows spatial abundance when this is a spatial state.
    pub const fn space(self) -> Option<&'a ArrayD<f64>> {
        self.space
    }
}

/// A scratch-backed deterministic update proposed by a kernel algorithm.
///
/// At least one canonical abundance payload is present. `Kernel` validates all
/// proposed values and shapes before committing any of them.
#[derive(Debug)]
pub enum KernelUpdate<'a> {
    /// Replace aggregate abundance values only.
    Abundance(ArrayView1<'a, f64>),
    /// Replace spatial abundance values only.
    Space(ArrayViewD<'a, f64>),
    /// Replace aggregate and spatial values as one atomic update.
    Both {
        /// Proposed aggregate abundance.
        abundance: ArrayView1<'a, f64>,
        /// Proposed spatial abundance.
        space: ArrayViewD<'a, f64>,
    },
}

/// Model-owned instantaneous right-hand side evaluated at canonical state.
pub enum KernelResidual<'a> {
    Abundance(ArrayView1<'a, f64>),
    Space(ArrayViewD<'a, f64>),
}

impl<'a> KernelUpdate<'a> {
    /// Creates an aggregate-only update.
    pub const fn abundance(values: ArrayView1<'a, f64>) -> Self {
        Self::Abundance(values)
    }

    /// Creates a spatial-only update.
    pub const fn space(values: ArrayViewD<'a, f64>) -> Self {
        Self::Space(values)
    }

    /// Creates a coordinated aggregate-and-spatial update.
    pub const fn both(abundance: ArrayView1<'a, f64>, space: ArrayViewD<'a, f64>) -> Self {
        Self::Both { abundance, space }
    }
}

/// One deterministic numerical algorithm plugged into [`Kernel`].
///
/// Implementations own their integration scratch. [`Self::compute`] receives
/// only immutable state and returns views into that scratch; it cannot mutate
/// the authoritative Workflow state. Total synchronization, noise, recording,
/// progress, and simulation-time advancement belong to other layers.
pub trait KernelAlgorithm {
    /// Algorithm-specific validation or evolution failure.
    type Error: Error + Send + Sync + 'static;

    /// Validates a state domain before evolution begins.
    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error>;

    /// Computes exactly one deterministic transition into owned scratch.
    fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error>;

    /// Evaluates the complete deterministic RHS without advancing state.
    ///
    /// Custom algorithms may return `None`; fixed-point termination then
    /// rejects that composition rather than substituting an inferred rate.
    fn residual<'algorithm>(
        &'algorithm mut self,
        _core: &KernelCore,
        _state: KernelStateView<'_>,
    ) -> Result<Option<KernelResidual<'algorithm>>, Self::Error> {
        Ok(None)
    }
}

/// Shared deterministic kernel composed with one algorithm implementation.
#[derive(Debug)]
pub struct Kernel<A> {
    core: KernelCore,
    algorithm: A,
}

impl<A> Kernel<A> {
    /// Creates a kernel from resolved shared facilities and an algorithm.
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
            .validate(&self.core, KernelStateView::new(abundance, space))
            .map_err(KernelStepError::Algorithm)
    }

    /// Computes and atomically commits one deterministic transition.
    ///
    /// Workflow time is not advanced. If computation or update validation
    /// fails, no canonical state payload is mutated.
    pub fn step(
        &mut self,
        state: &mut SystemState,
        time_step: TimeStep,
    ) -> Result<(), KernelStepError<A::Error>> {
        let update = {
            let (abundance, space) = state
                .borrow_payloads::<(AggregateAbundance, SpatialAbundance)>((
                    ABUNDANCE_FIELD,
                    SPACE_FIELD,
                ))
                .map_err(KernelStepError::State)?;
            self.algorithm
                .compute(
                    &self.core,
                    KernelStateView::new(abundance, space),
                    time_step,
                )
                .map_err(KernelStepError::Algorithm)?
        };

        let (abundance, space) = state
            .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(KernelStepError::State)?;
        validate_update(abundance, space, &update).map_err(KernelStepError::Update)?;
        commit_update(abundance, space, update);
        Ok(())
    }

    /// Returns the maximum component-wise scaled model residual.
    pub fn maximum_scaled_residual(
        &mut self,
        state: &SystemState,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Option<f64>, KernelStepError<A::Error>> {
        for (name, value) in [
            ("absolute", absolute_tolerance),
            ("relative", relative_tolerance),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(KernelStepError::Residual(
                    KernelResidualError::InvalidTolerance { name, value },
                ));
            }
        }
        let residual = {
            let (abundance, space) = state
                .borrow_payloads::<(AggregateAbundance, SpatialAbundance)>((
                    ABUNDANCE_FIELD,
                    SPACE_FIELD,
                ))
                .map_err(KernelStepError::State)?;
            let view = KernelStateView::new(abundance, space);
            self.algorithm
                .residual(&self.core, view)
                .map_err(KernelStepError::Algorithm)?
        };
        let maximum = match residual {
            None => return Ok(None),
            Some(KernelResidual::Abundance(values)) => {
                let abundance = state
                    .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
                    .map_err(KernelStepError::State)?;
                if values.len() != abundance.len() {
                    return Err(KernelStepError::Residual(
                        KernelResidualError::AbundanceLength {
                            expected: abundance.len(),
                            actual: values.len(),
                        },
                    ));
                }
                maximum_scaled_component(
                    values.iter().copied(),
                    abundance.iter().copied(),
                    absolute_tolerance,
                    relative_tolerance,
                )
                .map_err(KernelStepError::Residual)?
            }
            Some(KernelResidual::Space(values)) => {
                let space = state
                    .payload::<SpatialAbundance>(SPACE_FIELD)
                    .map_err(KernelStepError::State)?;
                let space = space.as_ref().ok_or(KernelStepError::Residual(
                    KernelResidualError::SpaceUnavailable,
                ))?;
                if values.shape() != space.shape() {
                    return Err(KernelStepError::Residual(KernelResidualError::SpaceShape {
                        expected: space.shape().to_vec(),
                        actual: values.shape().to_vec(),
                    }));
                }
                maximum_scaled_component(
                    values.iter().copied(),
                    space.iter().copied(),
                    absolute_tolerance,
                    relative_tolerance,
                )
                .map_err(KernelStepError::Residual)?
            }
        };
        Ok(Some(maximum))
    }
}

fn maximum_scaled_component(
    residuals: impl Iterator<Item = f64>,
    state_values: impl Iterator<Item = f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<f64, KernelResidualError> {
    let mut maximum = 0.0_f64;
    for (linear_index, (residual, value)) in residuals.zip(state_values).enumerate() {
        if !residual.is_finite() {
            return Err(KernelResidualError::NonFiniteValue {
                linear_index,
                value: residual,
            });
        }
        let denominator = absolute_tolerance + relative_tolerance * value.abs();
        let scaled = if denominator > 0.0 {
            residual.abs() / denominator
        } else if residual == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        maximum = maximum.max(scaled);
    }
    Ok(maximum)
}

fn validate_update(
    abundance: &AggregateAbundance,
    space: &SpatialAbundance,
    update: &KernelUpdate<'_>,
) -> Result<(), KernelUpdateError> {
    match update {
        KernelUpdate::Abundance(values) => validate_abundance(abundance, values.view()),
        KernelUpdate::Space(values) => validate_space(space, values.view()),
        KernelUpdate::Both {
            abundance: values,
            space: spatial_values,
        } => {
            validate_abundance(abundance, values.view())?;
            validate_space(space, spatial_values.view())
        }
    }
}

fn validate_abundance(
    current: &AggregateAbundance,
    proposed: ArrayView1<'_, f64>,
) -> Result<(), KernelUpdateError> {
    if proposed.len() != current.len() {
        return Err(KernelUpdateError::AbundanceLength {
            expected: current.len(),
            actual: proposed.len(),
        });
    }
    if let Some((index, value)) = proposed
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KernelUpdateError::NonFiniteValue {
            target: ABUNDANCE_FIELD,
            linear_index: index,
            value,
        });
    }
    Ok(())
}

fn validate_space(
    current: &SpatialAbundance,
    proposed: ArrayViewD<'_, f64>,
) -> Result<(), KernelUpdateError> {
    let current = current
        .as_ref()
        .ok_or(KernelUpdateError::SpaceUnavailable)?;
    if proposed.shape() != current.shape() {
        return Err(KernelUpdateError::SpaceShape {
            expected: current.shape().to_vec(),
            actual: proposed.shape().to_vec(),
        });
    }
    if let Some((index, value)) = proposed
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KernelUpdateError::NonFiniteValue {
            target: SPACE_FIELD,
            linear_index: index,
            value,
        });
    }
    Ok(())
}

fn commit_update(
    abundance: &mut AggregateAbundance,
    space: &mut SpatialAbundance,
    update: KernelUpdate<'_>,
) {
    match update {
        KernelUpdate::Abundance(values) => abundance.assign(&values),
        KernelUpdate::Space(values) => space
            .as_mut()
            .expect("space presence was validated before commit")
            .assign(&values),
        KernelUpdate::Both {
            abundance: values,
            space: spatial_values,
        } => {
            abundance.assign(&values);
            space
                .as_mut()
                .expect("space presence was validated before commit")
                .assign(&spatial_values);
        }
    }
}

/// Rejection of a scratch-backed deterministic update before state mutation.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum KernelUpdateError {
    /// Aggregate output must preserve the authoritative allocation shape.
    #[error("kernel abundance update length {actual} does not match state length {expected}")]
    AbundanceLength {
        /// Current state length.
        expected: usize,
        /// Proposed length.
        actual: usize,
    },
    /// A spatial update cannot be applied to a non-spatial state.
    #[error("kernel proposed a spatial update for a non-spatial state")]
    SpaceUnavailable,
    /// Spatial output must preserve the authoritative allocation shape.
    #[error("kernel space update shape {actual:?} does not match state shape {expected:?}")]
    SpaceShape {
        /// Current state shape.
        expected: Vec<usize>,
        /// Proposed shape.
        actual: Vec<usize>,
    },
    /// Kernel outputs must be finite before any value is committed.
    #[error("kernel {target} update at linear index {linear_index} is not finite: {value}")]
    NonFiniteValue {
        /// Canonical update target.
        target: &'static str,
        /// Row-major linear position in the proposed view.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
}

/// Rejection of an algorithm-provided deterministic residual.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum KernelResidualError {
    #[error("{name} residual tolerance must be finite and nonnegative, got {value}")]
    InvalidTolerance { name: &'static str, value: f64 },
    #[error("kernel residual length {actual} does not match state length {expected}")]
    AbundanceLength { expected: usize, actual: usize },
    #[error("kernel returned a spatial residual for a non-spatial state")]
    SpaceUnavailable,
    #[error("kernel residual shape {actual:?} does not match state shape {expected:?}")]
    SpaceShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("kernel residual at linear index {linear_index} is not finite: {value}")]
    NonFiniteValue { linear_index: usize, value: f64 },
}

/// Failure while validating or applying a deterministic kernel.
#[derive(Debug)]
#[non_exhaustive]
pub enum KernelStepError<E> {
    /// Canonical Workflow payload access failed.
    State(StateError),
    /// The selected algorithm rejected the state or transition.
    Algorithm(E),
    /// The proposed update was rejected before committing any values.
    Update(KernelUpdateError),
    /// The model RHS could not be compared safely with canonical state.
    Residual(KernelResidualError),
}

impl<E> fmt::Display for KernelStepError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => fmt::Display::fmt(error, formatter),
            Self::Algorithm(error) => write!(formatter, "kernel algorithm failed: {error}"),
            Self::Update(error) => fmt::Display::fmt(error, formatter),
            Self::Residual(error) => fmt::Display::fmt(error, formatter),
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
            Self::Algorithm(error) => Some(error),
            Self::Update(error) => Some(error),
            Self::Residual(error) => Some(error),
        }
    }
}
