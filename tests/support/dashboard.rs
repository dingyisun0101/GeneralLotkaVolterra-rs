//! Test-only PTY driver for Workflow's required dashboard.
#![allow(dead_code)]

use scientific_workflow::runtime::{RunSummary, RuntimeError};
use scientific_workflow::study::Study;
use std::path::Path;

pub fn in_dashboard(test: &str) -> bool {
    if std::env::var("WORKFLOW_DASHBOARD_TEST").as_deref() == Ok(test) {
        return true;
    }
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(include_str!("dashboard.py"))
        .arg(std::env::current_exe().unwrap())
        .arg(test)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

fn announce(output_root: &Path) {
    let control = std::env::var_os("WORKFLOW_DASHBOARD_CONTROL")
        .expect("execution test must first enter its dashboard");
    let existing: Vec<_> = std::fs::read_dir(output_root)
        .ok()
        .into_iter()
        .flatten()
        .map(|entry| entry.unwrap().path())
        .collect();
    let document = serde_json::json!({"root": output_root, "existing": existing});
    let temporary = Path::new(&control).with_extension("tmp");
    std::fs::write(&temporary, serde_json::to_vec(&document).unwrap()).unwrap();
    std::fs::rename(temporary, control).unwrap();
}

pub fn execute(study: Study) -> Result<RunSummary, RuntimeError> {
    announce(study.output_root());
    scientific_workflow::runtime::execute(study)
}

pub fn run(project: &Path) -> Result<(), scientific_workflow::WorkflowError> {
    announce(&project.join("output"));
    scientific_workflow::run(project)
}
