// Human-readable name used in metadata and plot titles.
pub const LABEL: &str = "lv_diffusive_deterministic";

// All JSON output, metadata, and rendered plots for this example go here.
// Spatial runs write both `signal/` and `space/` streams.
pub const OUTPUT: &str = "output/lv_diffusive_deterministic";

// Number of species in the population field.
pub const NUM_STRAINS: usize = 10;

// Maximum number of solver steps. Early termination may stop before this.
pub const TOTAL_STEPS: usize = 10_000;

// Save both aggregate signal and full spatial field every N solver steps.
pub const SAVE_INTERVAL: usize = 500;

// RK4 time step. Spatial explicit diffusion can require smaller values.
pub const DT: f64 = 0.003;

// Population entries below this threshold are removed during sanitization.
pub const CUTOFF: f64 = 1e-9;

// Grid shape excluding the final species axis. The solver stores species-last:
// space[x, y, species].
pub const SPATIAL_SHAPE: [usize; 2] = [128, 128];

// Per-species diffusion coefficients are generated as base - step * species.
pub const DIFFUSION_BASE: f64 = 0.025;
pub const DIFFUSION_STEP: f64 = 0.0015;

// Growth vector entries are generated as base - step * species.
pub const GROWTH_BASE: f64 = 0.35;
pub const GROWTH_STEP: f64 = 0.015;

// Diagonal interaction coefficient. Negative values limit each species' growth.
pub const SELF_LIMITATION: f64 = -0.70;

// Off-diagonal interaction strength in the cyclic pairwise interaction pattern.
pub const PAIR_INTERACTION: f64 = 0.08;

// Optional global cap applied after each spatial population update.
pub const CARRYING_CAPACITY: Option<f64> = Some(40_000.0);

// Initial density in every grid cell for every species.
pub const INITIAL_POPULATION: f64 = 0.25;
