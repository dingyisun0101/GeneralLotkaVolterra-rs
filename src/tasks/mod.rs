/*!
Task module surface.

Purpose:
    `tasks` exposes experiment-level entry points. Ready tasks wire
    non-spatial and spatial solver stacks into repeatable epoch runs.
*/

pub mod replicator_demographic;
pub mod replicator_deterministic;
pub mod replicator_diffusive_deterministic;

pub mod lv_demographic;
pub mod lv_deterministic;
pub mod lv_diffusive_deterministic;
