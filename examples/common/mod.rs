use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use general_lotka_volterra_rs::tasks::metadata::TaskOutcome;

pub struct ExampleProgress {
    pub counter: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    bar: ProgressBar,
}

impl ExampleProgress {
    pub fn start(label: &'static str, total_steps: usize) -> Self {
        let total = total_steps as u64;
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {msg} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("valid progress template")
            .progress_chars("#>-"),
        );
        bar.set_message(label);

        let counter = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let thread_counter = Arc::clone(&counter);
        let thread_done = Arc::clone(&done);
        let thread_bar = bar.clone();

        let handle = thread::spawn(move || {
            let mut completed_steps = 0usize;
            let mut previous_step = 0usize;

            while !thread_done.load(Ordering::Relaxed) {
                let step = thread_counter.load(Ordering::Relaxed);
                if step < previous_step {
                    completed_steps = completed_steps.saturating_add(previous_step);
                }
                previous_step = step;

                let position = completed_steps.saturating_add(step).min(total_steps);
                thread_bar.set_position(position as u64);
                thread::sleep(Duration::from_millis(100));
            }
        });

        Self {
            counter,
            done,
            handle: Some(handle),
            bar,
        }
    }

    pub fn finish(mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.bar.finish();
    }

    pub fn finish_at(mut self, position: usize) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.bar.set_position(position as u64);
        self.bar.finish();
    }
}

pub fn render_output_plot(output_path: &Path, title: &str) -> Result<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_path = output_path.canonicalize()?;
    let outdir = output_path.join("plot");
    let status = Command::new("python")
        .current_dir(manifest_dir)
        .args([
            "-m",
            "examples.plotting.render_from_output",
            output_path
                .to_str()
                .ok_or_else(|| Error::other("output path is not valid UTF-8"))?,
            "--outdir",
            outdir
                .to_str()
                .ok_or_else(|| Error::other("plot output path is not valid UTF-8"))?,
            "--title",
            title,
        ])
        .status()?;

    if !status.success() {
        return Err(Error::other(format!(
            "plot renderer failed with status {status}; activate a Python environment with numpy and matplotlib"
        )));
    }

    Ok(outdir.join("plot.png"))
}

pub fn run_and_render<F>(label: &'static str, total_steps: usize, output_path: &Path, run: F)
where
    F: FnOnce(Option<&AtomicUsize>) -> Result<TaskOutcome>,
{
    let progress = ExampleProgress::start(label, total_steps);

    let outcome = match run(Some(progress.counter.as_ref())) {
        Ok(outcome) => outcome,
        Err(err) => {
            progress.finish();
            eprintln!("{label} failed: {err}");
            std::process::exit(1);
        }
    };

    progress.finish_at(outcome.steps_run);
    println!(
        "steps: {} / {}; reason: {:?}; signal files: {}; space files: {}; metadata: {}",
        outcome.steps_run,
        outcome.requested_steps,
        outcome.termination_reason,
        outcome.signal.files,
        outcome.space.map(|space| space.files).unwrap_or(0),
        output_path.join("metadata.json").display()
    );

    match render_output_plot(output_path, label) {
        Ok(plot_path) => println!("plot: {}", plot_path.display()),
        Err(err) => {
            eprintln!("{label} plot rendering failed: {err}");
            std::process::exit(1);
        }
    }
}
