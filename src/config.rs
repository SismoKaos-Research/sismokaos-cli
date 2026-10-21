use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

fn default_chaos_tau() -> usize {
    5
}
fn default_chaos_dim() -> usize {
    5
}
fn default_chaos_evolve() -> usize {
    5
}
fn default_chaos_min_samples() -> usize {
    500
}
fn default_chaos_ros_rise() -> [f64; 2] {
    [0.1, 0.9]
}
fn default_chaos_ros_horizon() -> f64 {
    20.0
}
fn default_chaos_mean_period() -> f64 {
    1.0
}
fn default_chaos_sampent_m() -> usize {
    2
}
fn default_chaos_sampent_r() -> f64 {
    0.2
}

/// Where quiet periods come from and how the chaos parameters are estimated
/// on them. See src/calibrate.rs.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct CalibrationConfig {
    /// AFAD catalog CSV, optionally gzipped. Relative paths are taken from the
    /// working directory.
    pub catalog: PathBuf,
    /// Station list with code/latitude/longitude columns.
    pub station_coords: PathBuf,
    /// Overrides the station list, for codes it lacks or lists twice.
    pub station_lat: Option<f64>,
    pub station_lon: Option<f64>,
    /// Events below this magnitude or farther than this are ignored.
    pub min_mag: f64,
    pub max_dist_km: f64,
    /// Seconds excluded before each origin time.
    pub pre_sec: f64,
    /// Seconds excluded after it: the larger of `min_post_sec` and
    /// `coda_factor` times the magnitude's coda duration.
    pub min_post_sec: f64,
    pub coda_factor: f64,
    /// Quiet windows sampled from the whole run.
    pub max_windows: usize,
    /// A sampled window louder than this times the median std of its
    /// component is an uncatalogued event or a glitch, not background.
    pub loud_factor: f64,
    pub seed: u64,
    /// Search limits for tau (AMI lags, in samples) and dim (FNN).
    pub max_tau: usize,
    pub max_dim: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        // The repo carries its own catalog, so a binary built from it finds
        // one wherever it is run from.
        let catalogs = Path::new(env!("CARGO_MANIFEST_DIR")).join("catalogs");
        Self {
            catalog: catalogs.join("catalog_afad_2026-08-30.csv.gz"),
            station_coords: catalogs.join("istasyon_katalog.csv"),
            station_lat: None,
            station_lon: None,
            min_mag: 0.0,
            max_dist_km: 300.0,
            pre_sec: 120.0,
            min_post_sec: 300.0,
            coda_factor: 3.0,
            max_windows: 300,
            loud_factor: 3.0,
            seed: 0,
            max_tau: 20,
            max_dim: 10,
        }
    }
}

fn default_spectral_split_hz() -> f64 {
    1.0
}

/// One frequency band to extract features from. Each band is a run of its
/// own: preprocessed, calibrated and windowed with these settings, written to
/// `<out-dir>/<label>/`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Band {
    /// Defaults to `"<freqmin>-<freqmax>Hz"`.
    #[serde(default)]
    pub name: Option<String>,
    pub freqmin: f64,
    pub freqmax: f64,
    pub fs: f64,
    pub win_sec: u32,
    pub step_sec: u32,
    pub sta_sec: f64,
    pub lta_sec: u32,
    /// Split between the low- and high-frequency energy features; defaults to
    /// the band's geometric centre.
    #[serde(default)]
    pub spectral_split_hz: Option<f64>,
}

impl Band {
    pub fn label(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("{}-{}Hz", self.freqmin, self.freqmax))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AppConfig {
    pub station: String,
    pub fs: f64,
    pub win_sec: u32,
    pub step_sec: u32,
    pub sta_sec: f64,
    pub lta_sec: u32,
    pub freqmin: f64,
    pub freqmax: f64,

    // Chaotic feature parameters (Wolf/Rosenstein Lyapunov exponents, Sample
    // Entropy, Correlation Dimension) -- see src/features/chaos.rs. Each has
    // a `#[serde(default = ...)]` so existing config.json files written
    // before these fields existed keep loading instead of erroring out.
    #[serde(default = "default_chaos_tau")]
    pub chaos_tau: usize,
    #[serde(default = "default_chaos_dim")]
    pub chaos_dim: usize,
    #[serde(default = "default_chaos_evolve")]
    pub chaos_evolve: usize,
    #[serde(default = "default_chaos_min_samples")]
    pub chaos_min_samples: usize,
    #[serde(default = "default_chaos_mean_period")]
    pub chaos_mean_period: f64,
    /// Rosenstein fit range, as fractions of each window's divergence-curve
    /// rise from its first step to its peak.
    #[serde(default = "default_chaos_ros_rise")]
    pub chaos_ros_rise: [f64; 2],
    /// How far the divergence curve is followed, in mean periods.
    #[serde(default = "default_chaos_ros_horizon")]
    pub chaos_ros_horizon: f64,
    #[serde(default = "default_chaos_sampent_m")]
    pub chaos_sampent_m: usize,
    #[serde(default = "default_chaos_sampent_r")]
    pub chaos_sampent_r: f64,
    /// Absolute Sample Entropy tolerance per component (E, N, Z):
    /// `chaos_sampent_r` times the quiet-period std. `None` falls back to `r`
    /// times each window's own std.
    #[serde(default)]
    pub chaos_sampent_tol: Option<[f64; 3]>,

    // `chaos_tau`, `chaos_dim`, `chaos_mean_period` and `chaos_sampent_tol`
    // are overwritten by the calibration on quiet periods before a run.
    #[serde(default)]
    pub chaos_calibration: CalibrationConfig,

    /// Compute only the chaotic features.
    #[serde(default)]
    pub chaos_only: bool,

    /// Split between the LOW_FREQ_ENERGY and HIGH_FREQ_ENERGY features, Hz.
    #[serde(default = "default_spectral_split_hz")]
    pub spectral_split_hz: f64,

    /// Frequency bands to run one after another, each overriding the rate,
    /// filter, window and STA/LTA settings above. Empty runs the settings
    /// above as a single band.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bands: Vec<Band>,
    /// The band a run is for, recorded in its metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<String>,

    // Derived fields (calculated after loading, so we skip them during JSON parsing)
    #[serde(skip)]
    pub data_dir: PathBuf,
    #[serde(skip)]
    pub output_root: PathBuf,
    #[serde(skip)]
    pub win_size: usize,
    #[serde(skip)]
    pub step_size: usize,
    /// A previous run's calibration to reuse instead of estimating one.
    #[serde(skip)]
    pub calibration_file: Option<PathBuf>,
    /// `--band` selections; empty runs every band.
    #[serde(skip)]
    pub band_filter: Vec<String>,
    /// Single-band settings given on the command line, which a band list
    /// would silently override.
    #[serde(skip)]
    pub cli_band_overrides: Vec<&'static str>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            station: "ELZG".to_string(),
            fs: 5.0,
            win_sec: 200,
            step_sec: 50,
            sta_sec: 0.5,
            lta_sec: 60,
            freqmin: 0.1,
            freqmax: 2.0,
            chaos_tau: default_chaos_tau(),
            chaos_dim: default_chaos_dim(),
            chaos_evolve: default_chaos_evolve(),
            chaos_min_samples: default_chaos_min_samples(),
            chaos_mean_period: default_chaos_mean_period(),
            chaos_ros_rise: default_chaos_ros_rise(),
            chaos_ros_horizon: default_chaos_ros_horizon(),
            chaos_sampent_m: default_chaos_sampent_m(),
            chaos_sampent_r: default_chaos_sampent_r(),
            chaos_sampent_tol: None,
            chaos_calibration: CalibrationConfig::default(),
            chaos_only: false,
            spectral_split_hz: default_spectral_split_hz(),
            bands: Vec::new(),
            band: None,
            data_dir: PathBuf::new(),
            output_root: PathBuf::new(),
            win_size: 0,
            step_size: 0,
            calibration_file: None,
            band_filter: Vec::new(),
            cli_band_overrides: Vec::new(),
        }
    }
}

impl AppConfig {
    /// Loads the configuration from a JSON file. Falls back to default if file doesn't exist.
    pub fn from_json<P: AsRef<Path>>(path: P) -> Self {
        if let Ok(json_str) = fs::read_to_string(&path) {
            match serde_json::from_str::<AppConfig>(&json_str) {
                Ok(config) => return config,
                Err(e) => eprintln!("[WARNING] Failed to parse JSON: {}. Using defaults.", e),
            }
        } else {
            eprintln!(
                "[WARNING] Config file {:?} not found. Using defaults.",
                path.as_ref()
            );
        }

        AppConfig::default()
    }

    /// Merges CLI flags into the configuration. CLI flags take precedence.
    pub fn merge_with_cli(&mut self, cli: &crate::cli::Cli) {
        match &cli.command {
            crate::cli::Commands::Run {
                data_dir,
                out_dir,
                station,
                win_sec,
                fs,
                freqmin,
                freqmax,
                chaos_only,
                catalog,
                station_coords,
                calibration,
                band,
                ..
            } => {
                self.band_filter = band.clone();
                for (flag, given) in [
                    ("--win-sec", win_sec.is_some()),
                    ("--fs", fs.is_some()),
                    ("--freqmin", freqmin.is_some()),
                    ("--freqmax", freqmax.is_some()),
                ] {
                    if given {
                        self.cli_band_overrides.push(flag);
                    }
                }
                self.data_dir = data_dir.clone();
                self.output_root = out_dir.clone();
                self.chaos_only |= *chaos_only;
                self.calibration_file = calibration.clone();
                if let Some(c) = catalog {
                    self.chaos_calibration.catalog = c.clone();
                }
                if let Some(c) = station_coords {
                    self.chaos_calibration.station_coords = c.clone();
                }

                if let Some(s) = station {
                    self.station = s.clone();
                }
                if let Some(w) = win_sec {
                    self.win_sec = *w;
                }
                if let Some(f) = fs {
                    self.fs = *f;
                }
                if let Some(f) = freqmin {
                    self.freqmin = *f;
                }
                if let Some(f) = freqmax {
                    self.freqmax = *f;
                }
            }
            crate::cli::Commands::Preprocess {
                data_dir,
                out_dir,
                fs,
                freqmin,
                freqmax,
                ..
            } => {
                self.data_dir = data_dir.clone();
                self.output_root = out_dir.clone();

                if let Some(f) = fs {
                    self.fs = *f;
                }
                if let Some(f) = freqmin {
                    self.freqmin = *f;
                }
                if let Some(f) = freqmax {
                    self.freqmax = *f;
                }
            }
            crate::cli::Commands::Init { .. } => {}
        }

        self.finalize_derived_fields();
    }

    /// The configurations to run: one per selected band, or this one alone
    /// when there are no bands. Each band writes to its own subdirectory.
    pub fn band_runs(&self) -> Result<Vec<AppConfig>, String> {
        if self.bands.is_empty() {
            if !self.band_filter.is_empty() {
                return Err("--band given, but the config defines no bands".to_string());
            }
            return Ok(vec![self.clone()]);
        }
        let labels: Vec<String> = self.bands.iter().map(Band::label).collect();
        if let Some(unknown) = self.band_filter.iter().find(|b| !labels.contains(b)) {
            return Err(format!("no band named {:?}; the config defines {:?}", unknown, labels));
        }
        Ok(self
            .bands
            .iter()
            .zip(labels)
            .filter(|(_, label)| self.band_filter.is_empty() || self.band_filter.contains(label))
            .map(|(band, label)| {
                let mut c = self.clone();
                c.bands = Vec::new();
                c.freqmin = band.freqmin;
                c.freqmax = band.freqmax;
                c.fs = band.fs;
                c.win_sec = band.win_sec;
                c.step_sec = band.step_sec;
                c.sta_sec = band.sta_sec;
                c.lta_sec = band.lta_sec;
                c.spectral_split_hz = band
                    .spectral_split_hz
                    .unwrap_or((band.freqmin * band.freqmax).sqrt());
                c.output_root = self.output_root.join(&label);
                c.band = Some(label);
                c.finalize_derived_fields();
                c
            })
            .collect())
    }

    fn finalize_derived_fields(&mut self) {
        self.win_size = (self.win_sec as f64 * self.fs) as usize;
        self.step_size = (self.step_sec as f64 * self.fs) as usize;
    }

    /// Rejects parameter combinations that would silently corrupt the output.
    ///
    /// Returns the list of problems rather than the first, so one run surfaces
    /// everything wrong with a config instead of one item per attempt.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.fs <= 0.0 {
            problems.push(format!("fs must be positive, got {}", self.fs));
            return problems; // Everything below divides by it.
        }
        if self.freqmin <= 0.0 || self.freqmax <= self.freqmin {
            problems.push(format!(
                "need 0 < freqmin < freqmax, got freqmin={} freqmax={}",
                self.freqmin, self.freqmax
            ));
        }

        // The bandpass IS the anti-alias filter: decimation happens straight
        // after it, with no separate low-pass. If freqmax reaches the output
        // Nyquist, everything above it folds back into the band and there is
        // no way to tell afterwards that it happened.
        let nyquist = self.fs / 2.0;
        if self.freqmax >= nyquist {
            problems.push(format!(
                "freqmax {} Hz is at or above the output Nyquist ({} Hz for fs={}). \
                 The bandpass doubles as the anti-alias filter, so everything above \
                 {} Hz would alias back into the band with no way to detect it \
                 afterwards. Lower freqmax below {} Hz, or raise fs.",
                self.freqmax, nyquist, self.fs, nyquist, nyquist
            ));
        } else if self.freqmax > nyquist * 0.9 {
            eprintln!(
                "[WARNING] freqmax {} Hz sits within 10% of the output Nyquist ({} Hz). \
                 A 4th-order bandpass has not fully rolled off by then; some aliasing \
                 is likely.",
                self.freqmax, nyquist
            );
        }

        if self.win_sec == 0 || self.step_sec == 0 {
            problems.push("win_sec and step_sec must both be non-zero".to_string());
        }

        if !self.bands.is_empty() && !self.cli_band_overrides.is_empty() {
            problems.push(format!(
                "{} cannot be combined with the config's band list; edit the bands instead",
                self.cli_band_overrides.join(", ")
            ));
        }
        if self.calibration_file.is_some() && self.bands.len() > 1 && self.band_filter.len() != 1 {
            problems.push(
                "--calibration holds one band's parameters; pick that band with --band".to_string(),
            );
        }
        let mut labels: Vec<String> = self.bands.iter().map(Band::label).collect();
        labels.sort();
        if labels.windows(2).any(|w| w[0] == w[1]) {
            problems.push(format!("band names must be unique, got {:?}", labels));
        }
        let nsta = (self.sta_sec * self.fs) as usize;
        let nlta = (self.lta_sec as f64 * self.fs) as usize;
        if !self.chaos_only && (nsta == 0 || nlta >= self.win_size) {
            eprintln!(
                "[WARNING] STA/LTA needs 0 < sta_sec*fs and lta_sec*fs < win_size; got \
                 {} and {} samples against {}, so the STA_LTA features will be NaN.",
                nsta, nlta, self.win_size
            );
        }

        let [lo, hi] = self.chaos_ros_rise;
        if !(0.0 <= lo && lo < hi && hi <= 1.0) {
            problems.push(format!(
                "chaos_ros_rise must satisfy 0 <= lo < hi <= 1, got [{}, {}]",
                lo, hi
            ));
        }
        if self.chaos_ros_horizon <= 0.0 {
            problems.push(format!(
                "chaos_ros_horizon must be positive, got {}",
                self.chaos_ros_horizon
            ));
        }

        problems
    }

    /// Warns when the requested output rate is not an exact divisor of the
    /// native rate.
    ///
    /// The decimation factor is `native / requested` truncated to an integer,
    /// so asking for 30 Hz from a 100 Hz archive yields factor 3 and a real
    /// rate of 33.3 Hz. The pipeline stays self-consistent -- timestamps use
    /// the real rate -- but `win_size` was derived from the REQUESTED rate, so
    /// windows would be the wrong length. Better to say so than to let the two
    /// quietly disagree.
    pub fn check_decimation(&self, native_fs: f64) -> Option<String> {
        if native_fs <= 0.0 {
            return None;
        }
        let factor = ((native_fs / self.fs) as usize).max(1);
        let actual = native_fs / factor as f64;
        if (actual - self.fs).abs() > 1e-9 {
            return Some(format!(
                "requested fs={} Hz is not an exact divisor of the native {} Hz; \
                 decimating by {} gives {:.4} Hz instead. Window lengths were derived \
                 from the requested rate and will not match the data. \
                 Use an fs that divides {} exactly.",
                self.fs, native_fs, factor, actual, native_fs
            ));
        }
        None
    }

    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json_str = serde_json::to_string_pretty(self)?;
        fs::write(path, json_str)
    }
}
