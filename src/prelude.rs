/*!
User-facing convenience imports.

Purpose:
    Import `general_lotka_volterra_rs::prelude::*` in examples and applications
    to bring the common solver API, state constructors, spatial configuration,
    output helpers, and ndarray array types into scope.
*/

pub use ndarray::{Array1, Array2, ArrayD, IxDyn};

pub use crate::io::WriterStats;
pub use crate::io::signal::{SignalRecord, SignalSeries, load_signal_series};
pub use crate::io::space::{SpaceRecord, SpaceSeries, load_space_series};
pub use crate::solvers::noise::{Noise, NoiseContext, NoiseKind};
pub use crate::solvers::spatial::{Boundary, Diffusion};
pub use crate::solvers::termination::{
    AdaptiveFixedPointConfig, AdaptiveOscillationConfig, SolveOutcome, SteadyStateConfig,
    TerminationConfig, TerminationObservable, TerminationReason,
};
pub use crate::solvers::{Dynamics, SolveConfig, Space, solve};
pub use crate::tasks::metadata::{
    TaskOutcome, load_metadata, output_label, prepare_output_dir, save_metadata,
};
pub use crate::utils::{
    create_uniform_spatial_frequency_gs, create_uniform_spatial_population_gs, create_well_mixed_gs,
};
pub use crate::{Mode, SIGNAL_OUTPUT_FILE_SIZE, SPACE_OUTPUT_FILE_SIZE, Scalar, SystemState};
