//! Canonical GLV input loading and task registration.

use std::path::PathBuf;

use scientific_workflow::prelude::basics::{ExecutionScope, ExecutionScopeError};
use scientific_workflow::prelude::study::{PhaseBuilder, Task};
use thiserror::Error;

use crate::GlvTemplate;
use crate::study_inputs::{GlvInputs, GlvInputsError, load_glv_inputs};

/// A fully loaded GLV workload ready to register its expanded tasks.
pub struct GlvWorkload {
    inputs: GlvInputs,
    execution: ExecutionScope,
    template: GlvTemplate,
}

impl GlvWorkload {
    /// Loads and validates one self-contained workload directory.
    pub fn load(
        directory: impl Into<PathBuf>,
        template: GlvTemplate,
    ) -> Result<Self, GlvWorkloadError> {
        let inputs = load_glv_inputs(directory)?;
        let execution = ExecutionScope::create_generated(inputs.resolve_path("recordings")?)?;
        Ok(Self {
            inputs,
            execution,
            template,
        })
    }

    /// Loads a workload whose semantic task directories live directly beneath
    /// the configured recording root.
    pub fn load_in_place(
        directory: impl Into<PathBuf>,
        template: GlvTemplate,
    ) -> Result<Self, GlvWorkloadError> {
        let inputs = load_glv_inputs(directory)?;
        let execution = ExecutionScope::open_or_create(inputs.resolve_path("recordings")?)?;
        Ok(Self {
            inputs,
            execution,
            template,
        })
    }

    pub const fn inputs(&self) -> &GlvInputs {
        &self.inputs
    }
    pub const fn execution(&self) -> &ExecutionScope {
        &self.execution
    }
    pub const fn template(&self) -> GlvTemplate {
        self.template
    }

    /// Returns the conventional study-record path for this workload.
    pub fn record_path(&self) -> PathBuf {
        self.execution.directory().join("study-record.json")
    }

    /// Adds every expanded GLV configuration to an application-owned phase.
    pub fn register(&self, builder: PhaseBuilder) -> PhaseBuilder {
        let category = self.template.as_str();
        self.register_as(builder, category)
    }

    /// Adds every expanded configuration under an application-selected task namespace.
    ///
    /// This permits several independent GLV workload directories using the same
    /// template to coexist in one phase without task-ID collisions.
    pub fn register_as(&self, builder: PhaseBuilder, category: impl Into<String>) -> PhaseBuilder {
        let template = self.template;
        let execution = self.execution.clone();
        let category = category.into();
        let tasks = self.inputs.combinations().map(move |configuration| {
            let ordinal = configuration.ordinal();
            let execution = execution.clone();
            Task::progress(
                format!("{category}-{ordinal}"),
                format!("{category} {ordinal}"),
                move |context| Ok(template.run_task(&execution, &configuration, context)?),
            )
            .category(category.clone())
            .metadata("configuration_ordinal", ordinal)
        });
        builder.tasks(tasks)
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvWorkloadError {
    #[error(transparent)]
    Inputs(#[from] GlvInputsError),
    #[error(transparent)]
    Configuration(#[from] scientific_workflow::configuration::ConfigurationError),
    #[error(transparent)]
    Execution(#[from] ExecutionScopeError),
}
