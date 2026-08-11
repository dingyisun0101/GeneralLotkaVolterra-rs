//! GLV project loading with one crate-owned Scientific Workflow schema.
//!
//! Workflow owns project loading and task expansion. This module supplies the
//! model-family schema and checks that it names the three canonical GLV fields
//! in their canonical order.

use std::path::PathBuf;

use scientific_workflow::project::{ScientificProject, ScientificProjectError};
use scientific_workflow::system_state::StateError;
use thiserror::Error;

use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD, load_state_schema};

const GLV_FIELDS: [&str; 3] = [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD];

/// Failure while loading or validating a Workflow project for GLV use.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvProjectError {
    /// The crate-owned canonical state schema could not be loaded.
    #[error(transparent)]
    State(#[from] StateError),
    /// Scientific Workflow rejected the conventional project documents.
    #[error(transparent)]
    Workflow(#[from] ScientificProjectError),
    /// The project state schema does not declare the canonical GLV fields.
    #[error("GLV project state fields must be {expected:?}, found {actual:?}")]
    StateFields {
        /// Required canonical field order.
        expected: [&'static str; 3],
        /// Field order loaded from the project.
        actual: Vec<String>,
    },
}

/// Loads task configuration with GLV's crate-owned canonical state schema.
pub fn load_glv_project(
    project_root: impl Into<PathBuf>,
) -> Result<ScientificProject, GlvProjectError> {
    let project = ScientificProject::load_with_state_schema(project_root, load_state_schema()?)?;
    validate_glv_project(&project)?;
    Ok(project)
}

/// Validates that an already loaded project uses the canonical GLV fields.
pub fn validate_glv_project(project: &ScientificProject) -> Result<(), GlvProjectError> {
    let actual = project
        .state_schema()
        .field_schemas()
        .iter()
        .map(|field| field.name().to_owned())
        .collect::<Vec<_>>();
    if actual.iter().map(String::as_str).eq(GLV_FIELDS) {
        Ok(())
    } else {
        Err(GlvProjectError::StateFields {
            expected: GLV_FIELDS,
            actual,
        })
    }
}
