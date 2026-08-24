//! Canonical GLV input loading and task registration.

use std::path::PathBuf;

use scientific_workflow::prelude::basics::ExecutionScope;
use scientific_workflow::prelude::study::{PhaseBuilder, Task};
use thiserror::Error;

use crate::GlvTemplate;
use crate::study_inputs::{GlvInputs, GlvInputsError};

/// A fully loaded GLV workload ready to register its expanded tasks.
pub struct GlvWorkload {
    inputs: GlvInputs,
    execution: ExecutionScope,
    template: GlvTemplate,
}

impl GlvWorkload {
    /// Loads one workload into an application-selected execution scope.
    ///
    /// Replicate dispatch and output-scope creation belong to Workflow. GLV
    /// only maps its resolved configurations into tasks beneath the supplied
    /// scope.
    pub fn load(
        directory: impl Into<PathBuf>,
        template: GlvTemplate,
        execution: ExecutionScope,
    ) -> Result<Self, GlvWorkloadError> {
        let inputs = GlvInputs::load(directory)?;
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
        let project_paths = self.inputs.project_paths().clone();
        let tasks = self.inputs.combinations().map(move |configuration| {
            let execution = execution.clone();
            let resolved = configuration.configuration().clone();
            Task::progress_for_configuration(category.clone(), &resolved, move |context| {
                template.run_task(&execution, &configuration, context)
            })
            .with_project_paths(&project_paths)
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
}
