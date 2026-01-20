/// ==============================================================================================
/// =============================== Single Spatiotemporal State ==================================
/// ==============================================================================================

use serde::{Deserialize, Serialize};
use num_traits::Float;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayD, IxDyn};

pub trait Scalar:
    Float + Copy + Default + Send + Sync + Serialize + std::iter::Sum<Self>
{
}

impl<T> Scalar for T where
    T: Float + Copy + Default + Send + Sync + Serialize + std::iter::Sum<T>
{
}

/// Representation mode:
///     Two mutually exclusive conventions for what `state` means:
///         - `Frequency`: entries live on the simplex (mass = 1),
///         - `Population`: entries carry absolute counts (mass not necessarily 1)
///     where 'cutoff' is an absorbing boundary.
#[derive(Clone, Serialize, Deserialize)]
pub enum Mode<T> {
    Frequency { cutoff: T },
    Population { cutoff: T, carrying_capacity: Option<T> },
}

/// Snapshot at an integer time index.
///     - `state`: well-mixed / global vector (d)
///     - `space`: spatial field (shape arbitrary: [X, Y, Z, ...])
#[derive(Clone, Serialize, Deserialize)]
pub struct SystemState<T> {
    pub mode: Mode<T>,
    pub time: usize,
    pub state: Array1<T>,
    pub space: Option<ArrayD<T>>,
    pub mass: T,
}

/// Constructors and accessors.
impl<T> SystemState<T>
where
    T: Scalar,
{
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

    /// Build a SystemState from a discrete species-ID grid (0 = vacant, 1 = species 1, ...).
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

    // Hard-threshold invalid / nonpositive / below-cutoff entries to `zero`.
    //     - `cutoff` is assumed nonnegative by the caller
    //     - `zero` is carried to avoid repeated `T::zero()` calls
    #[inline]
    fn apply_cutoff(&mut self, cutoff: T, zero: T) {
        self.state.par_iter_mut().for_each(|x| {
            if !x.is_finite() || *x <= zero || *x < cutoff {
                *x = zero;
            }
        });
    }

    // Enforce mode-specific invariants and update `mass`.
    //     - `Frequency`: project onto simplex (sum = 1), set `mass = 1`
    //     - `Population`: apply cutoff, optionally cap at `carrying_capacity`, set `mass ≈ round(sum)`
    #[inline]
    pub fn sanitize(&mut self) {
        let zero = T::zero();

        match self.mode {
            Mode::Frequency { cutoff } => {
                // Cutoff sanitize.
                let cutoff = cutoff.max(zero);
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
                let cutoff = cutoff.max(zero);
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
