/*!
Single spatiotemporal state.

Purpose:
    `SystemState` stores one ecological snapshot: representation mode, integer
    time, global taxon vector, optional spatial field, and cached mass.

Representation modes:
    - `Frequency`: entries live on the simplex and `mass` is normalized to one.
    - `Population`: entries carry absolute counts and may be capped by carrying
      capacity.

Invariant boundary:
    `sanitize` removes invalid entries, applies cutoff/capacity rules, and
    refreshes cached mass. Solvers should call it after numerical updates that
    may leave the valid state domain.
*/

use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayD, IxDyn};
use num_traits::Float;
use serde::{Deserialize, Serialize};

pub trait Scalar: Float + Copy + Default + Send + Sync + Serialize + std::iter::Sum<Self> {}

impl<T> Scalar for T where T: Float + Copy + Default + Send + Sync + Serialize + std::iter::Sum<T> {}

/// Representation convention for the global state vector.
#[derive(Clone, Serialize, Deserialize)]
pub enum Mode<T> {
    Frequency {
        cutoff: Option<T>,
    },
    Population {
        cutoff: Option<T>,
        carrying_capacity: Option<T>,
    },
}

/// Snapshot at one integer time index.
#[derive(Clone, Serialize, Deserialize)]
pub struct SystemState<T> {
    pub mode: Mode<T>,
    pub time: usize,
    pub state: Array1<T>,
    pub space: Option<ArrayD<T>>,
    pub mass: T,
}

impl<T> SystemState<T>
where
    T: Scalar,
{
    /// Construct a state from owned vector and optional spatial arrays.
    ///
    /// Details:
    /// - Purpose: Creates the canonical state representation while enforcing
    ///   the initial mode convention.
    /// - Parameters:
    ///   - `mode`: Frequency or population interpretation.
    ///   - `time`: Integer simulation time.
    ///   - `state`: Global taxon vector.
    ///   - `space`: Optional spatial field.
    #[inline]
    pub fn from_arrays(
        mode: Mode<T>,
        time: usize,
        mut state: Array1<T>,
        space: Option<ArrayD<T>>,
    ) -> Self {
        let sum: T = state.iter().copied().sum();

        let mass = match &mode {
            Mode::Frequency { .. } => {
                if sum > T::zero() {
                    let inv = T::one() / sum;
                    state.iter_mut().for_each(|x| *x = *x * inv);
                    T::one()
                } else {
                    let d = state.len();
                    if d > 0 {
                        let v = T::one() / T::from(d).unwrap();
                        state.iter_mut().for_each(|x| *x = v);
                        T::one()
                    } else {
                        T::zero()
                    }
                }
            }
            Mode::Population { .. } => sum,
        };

        Self {
            mode,
            time,
            state,
            space,
            mass,
        }
    }

    /// Construct an empty state with optional spatial storage.
    ///
    /// Details:
    /// - Purpose: Allocates zero/default state storage for a known size.
    /// - Parameters:
    ///   - `mode`: Frequency or population interpretation.
    ///   - `time`: Integer simulation time.
    ///   - `num_taxa`: Global vector length.
    ///   - `space_shape`: Optional shape for `ArrayD` spatial storage.
    #[inline]
    pub fn empty(
        mode: Mode<T>,
        time: usize,
        num_taxa: usize,
        space_shape: Option<&[usize]>,
    ) -> Self {
        let state = Array1::from_elem(num_taxa, T::default());
        let space = match space_shape {
            Some(shape) => Some(ArrayD::from_elem(IxDyn(shape), T::default())),
            None => None,
        };

        Self::from_arrays(mode, time, state, space)
    }

    /// Build a state from a discrete species-id grid.
    ///
    /// Details:
    /// - Purpose: Counts positive grid ids into the global vector and stores a
    ///   typed copy of the grid as spatial state.
    /// - Parameters:
    ///   - `mode`: Frequency or population interpretation.
    ///   - `time`: Integer simulation time.
    ///   - `grid`: Discrete grid where `0` is vacant and `1..` are species ids.
    pub fn from_grid(mode: Mode<T>, time: usize, grid: &ArrayD<usize>) -> Self {
        let num_taxa = grid.iter().copied().max().unwrap_or(0);
        let zero = T::zero();
        let one = T::one();
        let mut state = Array1::from_elem(num_taxa, zero);

        for &cell in grid.iter() {
            if cell == 0 {
                continue;
            }
            let idx = cell - 1;
            state[idx] = state[idx] + one;
        }

        let space = Some(grid.mapv(|v| T::from(v).unwrap()));
        let mut gs = Self::from_arrays(mode, time, state, space);

        if let Mode::Frequency { .. } = gs.mode {
            gs.sanitize();
        }

        gs
    }

    #[inline]
    fn species_index(&self, i: usize) -> usize {
        let idx = i.checked_sub(1).expect("species index starts at 1");
        assert!(idx < self.state.len(), "species index {i} out of range");
        idx
    }

    /// Return the value for species i (1-based indexing).
    #[inline]
    pub fn get(&self, i: usize) -> T {
        let idx = self.species_index(i);
        self.state[idx]
    }

    /// Set the value for species i (1-based indexing).
    #[inline]
    pub fn set(&mut self, i: usize, value: T) {
        let idx = self.species_index(i);
        let old = self.state[idx];
        self.state[idx] = value;

        if let Mode::Population { .. } = self.mode {
            self.mass = (self.mass + value - old).max(T::zero());
        }
    }

    /// Increase species i by 1 (1-based indexing).
    #[inline]
    pub fn increase(&mut self, i: usize) {
        let idx = self.species_index(i);
        self.state[idx] = self.state[idx] + T::one();

        if let Mode::Population { .. } = self.mode {
            self.mass = self.mass + T::one();
        }
    }

    /// Decrease species i by 1 (1-based indexing).
    #[inline]
    pub fn decrease(&mut self, i: usize) {
        let idx = self.species_index(i);
        self.state[idx] = self.state[idx] - T::one();

        if let Mode::Population { .. } = self.mode {
            self.mass = (self.mass - T::one()).max(T::zero());
        }
    }

    // Hard-threshold invalid, nonpositive, and below-cutoff entries to zero.
    #[inline]
    fn apply_cutoff(&mut self, cutoff: T, zero: T) {
        self.state.par_iter_mut().for_each(|x| {
            if !x.is_finite() || *x <= zero || *x < cutoff {
                *x = zero;
            }
        });
    }

    /// Enforce mode-specific invariants and update cached mass.
    ///
    /// Details:
    /// - Purpose: Restores validity after numerical updates, noise, or direct
    ///   mutation.
    /// - Parameters:
    ///   - (none): Uses this state's mode and storage.
    #[inline]
    pub fn sanitize(&mut self) {
        let zero = T::zero();

        match self.mode {
            Mode::Frequency { cutoff } => {
                // Cutoff sanitize.
                let cutoff = cutoff.unwrap_or(zero).max(zero);
                self.apply_cutoff(cutoff, zero);

                // Renormalize onto simplex; fallback to uniform if all mass removed.
                let sum: T = self.state.par_iter().copied().sum();
                if sum > zero {
                    let inv = T::one() / sum;
                    self.state.par_iter_mut().for_each(|x| *x = *x * inv);
                } else {
                    let d = self.state.len();
                    if d > 0 {
                        let v = T::one() / T::from(d).unwrap();
                        self.state.par_iter_mut().for_each(|x| *x = v);
                    }
                }

                self.mass = T::one(); // simplex convention
            }
            Mode::Population {
                cutoff,
                carrying_capacity,
            } => {
                // Cutoff sanitize.
                let cutoff = cutoff.unwrap_or(zero).max(zero);
                self.apply_cutoff(cutoff, zero);

                let sum: T = self.state.par_iter().copied().sum();

                // No capacity constraint: just report (rounded) mass.
                let Some(capacity) = carrying_capacity else {
                    self.mass = sum.round().max(zero);
                    return;
                };

                // Degenerate capacity: zero out.
                if capacity <= zero {
                    self.state.par_iter_mut().for_each(|x| *x = zero);
                    self.mass = zero;
                    return;
                }

                // Under capacity: keep as-is.
                if sum < capacity {
                    self.mass = sum.round().max(zero);
                    return;
                }

                // Exactly at capacity: keep as-is (avoid divide-by-zero concerns elsewhere).
                if sum == capacity {
                    self.mass = capacity.round().max(zero);
                    return;
                }

                // Over capacity: rescale down to hit the cap exactly.
                let scale = capacity / sum;
                self.state.par_iter_mut().for_each(|x| *x = *x * scale);

                self.mass = capacity.round().max(zero);
            }
        }
    }
}
