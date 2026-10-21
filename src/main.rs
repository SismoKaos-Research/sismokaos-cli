mod calibrate;
mod catalog;
mod cli;
#[cfg(test)]
mod cli_tests;
mod config;
mod engine;
mod export;
mod features;
mod filter;
mod preprocess;
mod preprocess_only;
mod raw_binary;
mod types;

use clap::Parser;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cli::{Cli, Commands};
use config::AppConfig;
use indicatif::{FormattedDuration, ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use types::PipelineEvent;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Progress bar with ETA on a terminal; a plain line every 5% otherwise, so a
/// run redirected to a log file still says how far along it is.
struct Progress {
    bar: ProgressBar,
    started: bool,
    next_log_pct: u64,
}

impl Progress {
    fn new() -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(ProgressStyle::with_template(" {spinner} {msg}").unwrap());
        bar.set_message("Unpacking and indexing input...");
        bar.enable_steady_tick(Duration::from_millis(120));
        Progress {
            bar,
            started: false,
            next_log_pct: 5,
        }
    }

    fn update(&mut self, done: usize, total: usize) {
        if !self.started {
            self.started = true;
            self.bar.set_style(
                ProgressStyle::with_template(
                    " [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent:>3}%  ETA {eta_precise}  {msg}",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            self.bar.set_length(total as u64);
            self.bar.reset_eta();
        }
        self.bar.set_position(done as u64);

        if self.bar.is_hidden() && total > 0 {
            let pct = done as u64 * 100 / total as u64;
            if pct >= self.next_log_pct {
                println!(
                    " {:>3}%  elapsed {}  ETA {}",
                    pct,
                    FormattedDuration(self.bar.elapsed()),
                    FormattedDuration(self.bar.eta())
                );
                self.next_log_pct = (pct / 5 + 1) * 5;
            }
        }
    }

    /// Starts the bar over for another pass over the input.
    fn restart(&mut self) {
        if self.started {
            self.bar.reset();
            self.started = false;
            self.next_log_pct = 5;
        }
    }

    fn message(&self, msg: String) {
        self.bar.set_message(msg);
    }

    /// Prints above the bar; a hidden bar would swallow `bar.println`.
    fn say(&self, line: String) {
        if self.bar.is_hidden() {
            println!("{}", line);
        } else {
            self.bar.println(line);
        }
    }

    fn finish(&self) {
        self.bar.finish_and_clear();
    }
}
fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init { out } => {
            let default_config = AppConfig::default();
            if let Err(e) = default_config.save_to_json(out) {
                eprintln!("[ERROR] Failed to save config: {}", e);
            } else {
                println!("[SUCCESS] Created default configuration at {:?}", out);
            }
        }
        Commands::Run { dry_run, .. } | Commands::Preprocess { dry_run, .. } => {
            let mut app_config = AppConfig::from_json(&cli.config);
            app_config.merge_with_cli(&cli);

            let runs = match app_config.band_runs() {
                Ok(runs) => runs,
                Err(e) => {
                    eprintln!("[ERROR] {}", e);
                    std::process::exit(2);
                }
            };

            // Checked before any work starts. These are combinations whose
            // damage is invisible in the output, so failing late would mean
            // shipping a corrupt dataset that looks perfectly fine. Every
            // band is checked up front, so a bad last band cannot fail a run
            // hours in.
            let mut problems = app_config.validate();
            if !app_config.bands.is_empty() {
                for run in &runs {
                    let label = run.band.as_deref().unwrap_or_default();
                    problems.extend(run.validate().into_iter().map(|p| format!("band {}: {}", label, p)));
                }
            }
            if !problems.is_empty() {
                eprintln!("[ERROR] Invalid configuration:");
                for p in &problems {
                    eprintln!("  - {}", p);
                }
                std::process::exit(2);
            }

            if *dry_run {
                for run in runs.iter().filter(|r| r.band.is_some()) {
                    println!(
                        " band {:<14} {}-{} Hz at {} Hz, {} s windows every {} s -> {:?}",
                        run.band.as_deref().unwrap_or_default(),
                        run.freqmin,
                        run.freqmax,
                        run.fs,
                        run.win_sec,
                        run.step_sec,
                        run.output_root
                    );
                }
                println!("[DRY RUN ENABLED] Exiting without processing.");
                return;
            }

            println!("=== STARTING PIPELINE ===");

            let (tx, rx) = mpsc::channel();
            let run_preprocess = matches!(&cli.command, Commands::Preprocess { .. });

            // Bands run one after another; one failing does not stop the rest.
            let engine_handle = thread::spawn(move || {
                let total = runs.len();
                let mut outcomes = Vec::new();
                for (i, run) in runs.into_iter().enumerate() {
                    let label = run.band.clone();
                    if let Some(label) = &label {
                        let _ = tx.send(PipelineEvent::Info(format!(
                            "=== Band {} ({}/{}): {}-{} Hz at {} Hz ===",
                            label,
                            i + 1,
                            total,
                            run.freqmin,
                            run.freqmax,
                            run.fs
                        )));
                    }
                    let result = if run_preprocess {
                        preprocess_only::run_preprocess_only(run, tx.clone())
                    } else {
                        engine::run_pipeline(run, tx.clone())
                    };
                    if let Err(e) = &result {
                        let _ = tx.send(PipelineEvent::Warning(format!("Pipeline Error: {}", e)));
                    }
                    outcomes.push((label, result));
                }
                outcomes
            });

            let mut progress = Progress::new();
            let mut windows: u64 = 0;
            let mut samples: u64 = 0;

            for event in rx {
                match event {
                    PipelineEvent::FolderStarted(folder) => {
                        progress.say(format!(" Input: {}", folder));
                    }
                    PipelineEvent::Progress { done, total } => {
                        progress.update(done, total);
                    }
                    PipelineEvent::WindowProcessed => {
                        windows += 1;
                        if windows % 120 == 0 {
                            progress.message(format!("{} windows", windows));
                        }
                    }
                    PipelineEvent::ChunkProcessed(sample_count) => {
                        samples += sample_count as u64;
                        progress.message(format!("{} samples", samples));
                    }
                    PipelineEvent::Completed => {
                        progress.finish();
                        if run_preprocess {
                            println!(" Samples processed: {}", samples);
                        } else {
                            println!(" Windows processed: {}", windows);
                        }
                        windows = 0;
                        samples = 0;
                        progress = Progress::new();
                    }
                    PipelineEvent::Phase(name) => {
                        progress.restart();
                        progress.say(format!(" {}", name));
                    }
                    PipelineEvent::Info(line) => {
                        progress.say(line);
                    }
                    PipelineEvent::Warning(msg) => {
                        progress.say(format!(" Warning: {}", msg));
                    }
                    PipelineEvent::Finished => {
                        progress.finish();
                    }
                }
            }
            progress.finish();

            let outcomes = engine_handle.join().unwrap_or_default();
            let failed: Vec<String> = outcomes
                .iter()
                .filter(|(_, r)| r.is_err())
                .map(|(label, _)| label.clone().unwrap_or_else(|| "run".to_string()))
                .collect();
            if failed.is_empty() {
                println!("All processing finished successfully");
            } else {
                eprintln!("[ERROR] Failed: {}", failed.join(", "));
                std::process::exit(1);
            }
        }
    }
}
