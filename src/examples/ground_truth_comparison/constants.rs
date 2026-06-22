// Temporary output root. The example removes this directory at start-up so each
// run compares fresh Rust and SciPy outputs.
pub const OUTPUT: &str = "output/ground_truth_comparison";

// Fixed-step settings for the Rust solvers. SciPy uses adaptive integration
// with much tighter tolerances, so Rust should converge toward the SciPy answer
// as DT decreases.
pub const DT: f64 = 0.001;
pub const NUM_STEPS: usize = 100;
pub const SAVE_INTERVAL: usize = NUM_STEPS;

// External reference tolerances. These are intentionally looser than SciPy's
// internal tolerances because they compare adaptive high-accuracy integration
// against this crate's fixed-step solvers plus sanitization.
pub const REPLICATOR_TOLERANCE: f64 = 1e-7;
pub const SPATIAL_GLV_TOLERANCE: f64 = 1e-6;

// Tiny systems keep the comparison fast and make failures easy to inspect.
pub const NUM_REPLICATOR_SPECIES: usize = 3;
pub const NUM_GLV_SPECIES: usize = 2;
pub const NUM_SPATIAL_CELLS: usize = 4;
