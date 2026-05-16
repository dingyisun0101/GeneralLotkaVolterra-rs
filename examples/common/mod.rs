use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

pub struct ExampleProgress {
    pub counter: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    bar: ProgressBar,
}

impl ExampleProgress {
    pub fn start(label: &'static str, epoch_len: usize, num_epochs: usize) -> Self {
        let total = (epoch_len * num_epochs) as u64;
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
            let mut completed_epochs = 0usize;
            let mut previous_step = 0usize;

            while !thread_done.load(Ordering::Relaxed) {
                let step = thread_counter.load(Ordering::Relaxed);
                if step < previous_step {
                    completed_epochs += 1;
                }
                previous_step = step;

                let position = completed_epochs
                    .saturating_mul(epoch_len)
                    .saturating_add(step)
                    .min(epoch_len * num_epochs);
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
}
