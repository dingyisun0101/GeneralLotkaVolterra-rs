/*!
Spatial solver module.

Purpose:
    This module groups spatial dynamics. Spatial solvers evolve an arbitrary-
    dimensional field whose final axis is the taxon/species axis.
*/

pub mod noise;
pub mod rk2;

pub use rk2::{Boundary, Diffusion};
