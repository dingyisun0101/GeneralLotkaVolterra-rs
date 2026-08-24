use general_lotka_volterra_rs::GlvConfiguration;
use general_lotka_volterra_rs::advanced::prelude::{GlvTemplate, TemplateTaskError};
use scientific_workflow::prelude::basics::ExecutionScope;
use scientific_workflow::prelude::study::TaskContext;

#[test]
fn built_in_template_exposes_a_study_task_contract() {
    let run_task: fn(
        GlvTemplate,
        &ExecutionScope,
        &GlvConfiguration,
        &TaskContext,
    ) -> Result<(), TemplateTaskError> = GlvTemplate::run_task;
    let _ = run_task;
}
