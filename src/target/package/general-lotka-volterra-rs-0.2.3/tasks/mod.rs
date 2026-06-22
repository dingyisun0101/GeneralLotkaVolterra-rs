/*!
Task module surface.

Purpose:
    `tasks` exposes experiment-level entry points. Ready tasks wire
    non-spatial and spatial solver stacks into total-step runs whose signal and
    space output streams are automatically chunked.
*/

pub mod metadata;
pub mod replicator_demographic;
pub mod replicator_deterministic;
pub mod replicator_diffusive_deterministic;

pub mod lv_demographic;
pub mod lv_deterministic;
pub mod lv_diffusive_deterministic;
