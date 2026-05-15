/*!
General Lotka-Volterra crate root.

Purpose:
    This crate provides state containers, non-spatial solver machinery, task
    runners, and examples for ecological dynamical-system experiments.

Current implementation boundary:
    The ready solver path is replicator-form. GLV-named task modules are
    placeholders until a dedicated GLV right-hand side and integrator are added.
*/

pub mod examples;
pub mod solvers;
pub mod state;
pub mod tasks;
pub mod utils;
