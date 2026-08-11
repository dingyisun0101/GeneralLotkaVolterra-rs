//! GLV validation for conventional Scientific Workflow projects.
//!
//! Workflow owns project loading and task expansion. This module adds only the
//! model-family contract that the project state schema must name the three
//! canonical GLV fields in their canonical order.

use std::path::PathBuf;

use scientific_workflow::project::{ScientificProject, ScientificProjectError};
use thiserror::Error;

use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD};

const GLV_FIELDS: [&str; 3] = [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD];

/// Failure while loading or validating a Workflow project for GLV use.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvProjectError {
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

/// Loads a conventional Workflow project and validates its GLV state schema.
pub fn load_glv_project(
    project_root: impl Into<PathBuf>,
) -> Result<ScientificProject, GlvProjectError> {
    let project = ScientificProject::load(project_root)?;
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
