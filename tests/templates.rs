use general_lotka_volterra_rs::advanced::prelude::{
    ExecutionScope, GlvTemplate, TaskContext, TemplateTaskError,
};

#[test]
fn built_in_template_exposes_a_runtime_workload_contract() {
    let run_task: fn(GlvTemplate, &ExecutionScope, &TaskContext) -> Result<(), TemplateTaskError> =
        GlvTemplate::run_task;
    let _ = run_task;
}
