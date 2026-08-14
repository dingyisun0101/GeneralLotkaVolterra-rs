//! Canonical GLV workload-directory loading and runtime registration.

use std::path::PathBuf;

use scientific_workflow::execution::{ExecutionScope, ExecutionScopeError};
use scientific_workflow::project::ScientificProject;
use scientific_workflow::runtime::PhaseBuilder;
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

    pub const fn project(&self) -> &ScientificProject {
        &self.project
    }
    pub const fn execution(&self) -> &ExecutionScope {
        &self.execution
    }
    pub const fn template(&self) -> GlvTemplate {
        self.template
    }

    /// Adds every expanded GLV configuration to an application-owned phase.
    pub fn register(self, builder: PhaseBuilder) -> PhaseBuilder {
        let template = self.template;
        let execution = self.execution;
        builder.progress_tasks_from_project(&self.project, template.as_str(), move |context| {
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
