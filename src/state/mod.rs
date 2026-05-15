/*!
State module surface.

Purpose:
    `state` owns the common snapshot, mode, scalar, and time-series types used
    by solvers and task runners.
*/

pub mod system_state;
pub mod time_series;

pub use system_state::{Mode, Scalar, SystemState};
pub use time_series::{SystemStateRecord, SystemStateTimeSeries};
