//! Built-in invariant-policy implementations.

mod frequency;
mod local_frequency;
mod population;

pub use frequency::FrequencyInvariant;
pub use local_frequency::LocalFrequencyInvariant;
pub use population::PopulationInvariant;
