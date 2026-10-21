use crate::cli::{Cli, Commands};
use clap::Parser;
use std::path::PathBuf;

#[test]
fn cli_parses_preprocess_subcommand() {
    let cli = Cli::parse_from([
        "sismokaos-cli",
        "preprocess",
        "--data-dir",
        "./data",
        "--out-dir",
        "./out",
        "--fs",
        "50.0",
        "--freqmin",
        "0.1",
        "--freqmax",
        "20.0",
    ]);

    match cli.command {
        Commands::Preprocess {
            data_dir,
            out_dir,
            fs,
            freqmin,
            freqmax,
            dry_run,
        } => {
            assert_eq!(data_dir, PathBuf::from("./data"));
            assert_eq!(out_dir, PathBuf::from("./out"));
            assert_eq!(fs, Some(50.0));
            assert_eq!(freqmin, Some(0.1));
            assert_eq!(freqmax, Some(20.0));
            assert!(!dry_run);
        }
        _ => panic!("expected preprocess subcommand"),
    }
}

#[test]
fn cli_parses_band_and_chaos_flags() {
    let cli = Cli::parse_from([
        "sismokaos-cli",
        "run",
        "--data-dir",
        "./data",
        "--out-dir",
        "./out",
        "--chaos-only",
        "--band",
        "1-5Hz",
        "--band",
        "5-20Hz",
        "--calibration",
        "cal.json",
    ]);

    match cli.command {
        Commands::Run {
            chaos_only,
            band,
            calibration,
            catalog,
            ..
        } => {
            assert!(chaos_only);
            assert_eq!(band, vec!["1-5Hz".to_string(), "5-20Hz".to_string()]);
            assert_eq!(calibration, Some(PathBuf::from("cal.json")));
            assert_eq!(catalog, None);
        }
        _ => panic!("expected run subcommand"),
    }
}

#[test]
fn bands_expand_into_one_run_each() {
    use crate::config::AppConfig;

    let json = r#"{
        "station": "ELBA", "fs": 5.0, "win_sec": 200, "step_sec": 50,
        "sta_sec": 0.5, "lta_sec": 60, "freqmin": 0.1, "freqmax": 2.0,
        "bands": [
            {"freqmin": 1.0, "freqmax": 5.0, "fs": 20.0, "win_sec": 50, "step_sec": 50, "sta_sec": 1.0, "lta_sec": 20},
            {"name": "hi", "freqmin": 5.0, "freqmax": 20.0, "fs": 50.0, "win_sec": 20, "step_sec": 50, "sta_sec": 0.2, "lta_sec": 8, "spectral_split_hz": 9.0}
        ]
    }"#;
    let mut config: AppConfig = serde_json::from_str(json).unwrap();
    config.output_root = PathBuf::from("out");

    let runs = config.band_runs().unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].band.as_deref(), Some("1-5Hz"));
    assert_eq!(runs[0].output_root, PathBuf::from("out/1-5Hz"));
    assert_eq!((runs[0].win_size, runs[0].step_size), (1000, 1000));
    assert!((runs[0].spectral_split_hz - 5f64.sqrt()).abs() < 1e-12);
    assert!(runs[0].bands.is_empty());
    assert_eq!(runs[1].spectral_split_hz, 9.0);
    assert!(runs.iter().all(|r| r.validate().is_empty()));

    config.band_filter = vec!["hi".to_string()];
    let runs = config.band_runs().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].fs, 50.0);

    config.band_filter = vec!["nope".to_string()];
    assert!(config.band_runs().is_err());

    config.band_filter.clear();
    config.cli_band_overrides = vec!["--fs"];
    assert_eq!(config.validate().len(), 1);
}

#[test]
fn configs_without_the_new_fields_still_load() {
    use crate::config::AppConfig;

    // config_chaos_5hz.json as it was before calibration and bands existed.
    let json = r#"{
        "station": "BODT", "fs": 5.0, "win_sec": 200, "step_sec": 50,
        "sta_sec": 0.5, "lta_sec": 60, "freqmin": 0.1, "freqmax": 2.0,
        "chaos_tau": 5, "chaos_dim": 5, "chaos_evolve": 5, "chaos_min_samples": 500,
        "chaos_slope_ros": [0.5, 5.0, 5.0, 20.0], "chaos_mean_period": 1.0,
        "chaos_sampent_m": 2, "chaos_sampent_r": 0.2
    }"#;
    let config: AppConfig = serde_json::from_str(json).unwrap();
    assert!(config.bands.is_empty() && !config.chaos_only);
    assert_eq!(config.chaos_ros_rise, [0.1, 0.9]);
    assert_eq!(config.spectral_split_hz, 1.0);
    assert!(config.chaos_calibration.catalog.is_file(), "the repo's catalog should be the default");
}
