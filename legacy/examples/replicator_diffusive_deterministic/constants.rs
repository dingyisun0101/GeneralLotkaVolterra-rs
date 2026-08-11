// Human-readable name used in metadata and plot titles.
pub const LABEL: &str = "replicator_diffusive_deterministic";

// All JSON output, metadata, and rendered plots for this example go here.
// Spatial runs write both `signal/` and `space/` streams.
pub const OUTPUT: &str = "output/replicator_diffusive_deterministic";

// Number of species in each local spatial simplex.
pub const NUM_STRAINS: usize = 10;

// Maximum number of solver steps. Early termination may stop before this.
pub const TOTAL_STEPS: usize = 10_000;

// Save both aggregate signal and full spatial field every N solver steps.
pub const SAVE_INTERVAL: usize = 500;

// RK2 time step. Spatial explicit diffusion can require smaller values.
pub const DT: f64 = 0.003;

// Local frequencies below this threshold are removed during cell sanitization.
pub const CUTOFF: f64 = 1e-9;

// Grid shape excluding the final species axis. The solver stores species-last:
// space[x, y, species].
pub const SPATIAL_SHAPE: [usize; 2] = [128, 128];

// Per-species diffusion coefficients are generated as base - step * species.
pub const DIFFUSION_BASE: f64 = 0.020;
pub const DIFFUSION_STEP: f64 = 0.001;

// Growth vector entries are generated as base - step * species.
pub const GROWTH_BASE: f64 = 0.02;
pub const GROWTH_STEP: f64 = 0.004;

// Strength of the cyclic anti-symmetric interaction pattern.
pub const INTERACTION_STRENGTH: f64 = 1.0;
