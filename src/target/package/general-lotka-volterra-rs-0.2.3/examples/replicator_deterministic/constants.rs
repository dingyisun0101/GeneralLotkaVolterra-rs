// Human-readable name used in metadata and plot titles.
pub const LABEL: &str = "replicator_deterministic";

// All JSON output, metadata, and rendered plots for this example go here.
// `prepare_output_dir` clears stale signal/space/metadata outputs before a run.
pub const OUTPUT: &str = "output/replicator_deterministic";

// Number of interacting strategies/species in the well-mixed state vector.
pub const NUM_STRAINS: usize = 10;

// Maximum number of solver steps. Early termination may stop before this.
pub const TOTAL_STEPS: usize = 10_000;

// Save one aggregate signal sample every N solver steps.
pub const SAVE_INTERVAL: usize = 500;

// RK4 time step. Smaller values are safer for stiff or noisy systems.
pub const DT: f64 = 0.005;

// Frequency entries below this threshold are removed during sanitization.
pub const CUTOFF: f64 = 1e-5;

// Random pairwise interaction range used to build the example matrix.
pub const RANDOM_INTERACTION_MIN: f64 = -0.5;
pub const RANDOM_INTERACTION_MAX: f64 = 0.5;
