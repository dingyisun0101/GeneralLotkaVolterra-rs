/*!
Solver module surface.

Purpose:
    `solvers` groups numerical evolution backends. The active implementation is
    the non-spatial replicator solver; `spatial` is reserved for future spatial
    evolution.
*/

pub mod non_spatial;
pub mod spatial;
pub mod termination;
