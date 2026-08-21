//! Canonical GLV workload-directory loading and runtime registration.

use std::path::PathBuf;

use scientific_workflow::prelude::basics::{
    ExecutionScope, ExecutionScopeError, ScientificProject,
};
use scientific_workflow::prelude::runtime::PhaseBuilder;
use thiserror::Error;

use crate::GlvTemplate;
use crate::project::{GlvProjectError, load_glv_project};

/// A fully loaded GLV workload ready to register its expanded tasks.
pub struct GlvWorkload {
    project: ScientificProject,
    execution: ExecutionScope,
    template: GlvTemplate,
}

impl GlvWorkload {
    /// Loads and validates one self-contained workload directory.
    pub fn load(
        directory: impl Into<PathBuf>,
        template: GlvTemplate,
    ) -> Result<Self, GlvWorkloadError> {
        let project = load_glv_project(directory)?;
        let execution = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
        Ok(Self {
            project,
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
        let project = load_glv_project(directory)?;
        let execution = ExecutionScope::open_or_create(project.resolve_path("recordings")?)?;
        Ok(Self {
            project,
            execution,
            template,
        })
    }

    pub const fn project(&self) -> &ScientificProject {
        &self.project
    }
    pub const fn execution(&self) -> &ExecutionScope {
        &self.execution
    }
    pub const fn template(&self) -> GlvTemplate {
        self.template
    }

    /// Returns the conventional execution-record path for this workload.
    pub fn execution_record_path(&self) -> PathBuf {
        self.execution.directory().join("execution-record.json")
    }

    /// Adds every expanded GLV configuration to an application-owned phase.
    pub fn register(&self, builder: PhaseBuilder) -> PhaseBuilder {
        let kind = self.template.as_str();
        self.register_as(builder, kind)
    }

    /// Adds every expanded configuration under an application-selected task namespace.
    ///
    /// This permits several independent GLV workload directories using the same
    /// template to coexist in one Workflow phase without task-ID collisions.
    pub fn register_as(&self, builder: PhaseBuilder, kind: impl Into<String>) -> PhaseBuilder {
        let template = self.template;
        let execution = self.execution.clone();
        builder.progress_tasks_from_project(&self.project, kind, move |context| {
            template.run_task(&execution, context)
        })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvWorkloadError {
    #[error(transparent)]
    Project(#[from] GlvProjectError),
    #[error(transparent)]
    Configuration(#[from] scientific_workflow::configuration::ConfigurationError),
    #[error(transparent)]
    Execution(#[from] ExecutionScopeError),
}
