/*!
Persisted output formats.

Purpose:
    `io` owns JSON-facing signal and space streams. Solvers keep using
    `SystemState` internally, while output is written as dedicated aggregate
    signal and spatial snapshot series.
*/

pub mod signal;
pub mod space;
