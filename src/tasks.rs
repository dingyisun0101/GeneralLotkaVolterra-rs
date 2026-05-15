/*!
Task module surface.

Purpose:
    `tasks` exposes experiment-level entry points. Ready tasks use the
    replicator solver stack; GLV task names currently return unsupported
    placeholders.
*/

pub mod replicator_demographic;
pub mod replicator_deterministic;

pub mod lv_demographic;
pub mod lv_deterministic;
