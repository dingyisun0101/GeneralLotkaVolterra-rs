use general_lotka_volterra_rs::advanced::prelude::{
    ExecutionScope, GlvProjectTemplate, GlvTemplate, ProgressReporter, TaskConfig,
    TemplateTaskError,
};

struct AdvancedTemplate;

impl GlvProjectTemplate for AdvancedTemplate {
    fn name(&self) -> &str {
        "advanced_test_template"
    }

    fn run_task(
        &mut self,
        _scope: &ExecutionScope,
        _reporter: &ProgressReporter,
        _task: TaskConfig,
    ) -> Result<(), TemplateTaskError> {
        Ok(())
    }
}

fn require_template<T: GlvProjectTemplate>(_template: T) {}

#[test]
fn built_in_and_advanced_templates_share_one_contract() {
    require_template(GlvTemplate::MeanFieldReplicator);
    require_template(AdvancedTemplate);
}
