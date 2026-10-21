//! Calibrates the chaotic-feature parameters on seismically quiet periods.
//!
//! The embedding delay, embedding dimension, Rosenstein mean period and the
//! Sample Entropy tolerance all depend on the signal they are measured on. Set
//! per window, they would change across an earthquake along with the features
//! themselves, and the features would stop being comparable from one window to
//! the next. So they are estimated once, on background noise, and held fixed
//! for the whole run:
//!
//! 1. Every catalog event near the station becomes an exclusion interval (see
//!    `catalog.rs`). Non-overlapping windows that touch none of them, and hold
//!    no gap-filled samples, are quiet.
//! 2. One streaming pass over the data reservoir-samples `max_windows` of them
//!    from the whole run, seeded, so the pick does not favour its start.
//! 3. Windows far louder than their component's median are dropped: the
//!    catalog misses teleseisms, uncatalogued micro-events and glitches.
//! 4. On what is left, per window and component: tau is the first minimum of
//!    the average mutual information, the mean period the reciprocal of the
//!    power-weighted mean frequency, and, at that tau, dim comes from False
//!    Nearest Neighbours. The run uses the medians. The SampEn tolerance is
//!    `chaos_sampent_r` times each component's median quiet std.

use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::mpsc::Sender;

use crate::catalog::{self, Exclusions};
use crate::config::{AppConfig, CalibrationConfig};
use crate::features::chaos;
use crate::preprocess;
use crate::types::PipelineEvent;

/// Fewer quiet windows than this give medians too noisy to fix a run on.
const MIN_QUIET_WINDOWS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub station: String,
    /// Rate the parameters were measured at; `tau` is in samples of it.
    pub fs: f64,
    pub tau: usize,
    pub dim: usize,
    /// Seconds.
    pub mean_period: f64,
    pub sampent_r: f64,
    /// Absolute SampEn tolerance per component (E, N, Z).
    pub sampent_tol: [f64; 3],
    pub spread: Spread,
    pub windows: Counts,
    pub station_lat: f64,
    pub station_lon: f64,
    pub exclusion_intervals: usize,
    pub settings: CalibrationConfig,
}

/// Interquartile ranges of the per-window estimates the medians came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spread {
    pub tau_iqr: [f64; 2],
    pub dim_iqr: [f64; 2],
    pub mean_period_iqr: [f64; 2],
    pub quiet_std: [f64; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Counts {
    /// Non-overlapping windows in the data.
    pub total: usize,
    /// Of those, gap-free and clear of every exclusion interval.
    pub quiet: usize,
    pub sampled: usize,
    /// Sampled windows dropped as too loud, per component.
    pub loud: [usize; 3],
    /// Window-components measured, and how many had an AMI minimum.
    pub measured: usize,
    pub with_tau: usize,
}

impl Calibration {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
        serde_json::from_str(&text).map_err(|e| format!("Failed to parse {:?}: {}", path, e))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| format!("Failed to write {:?}: {}", path, e))
    }

    /// Fixes the run's chaos parameters to the calibrated ones.
    pub fn apply(&self, config: &mut AppConfig) {
        config.chaos_tau = self.tau;
        config.chaos_dim = self.dim;
        config.chaos_mean_period = self.mean_period;
        config.chaos_sampent_r = self.sampent_r;
        config.chaos_sampent_tol = Some(self.sampent_tol);
    }

    pub fn describe(&self) -> Vec<String> {
        let w = &self.windows;
        vec![
            format!(
                " Quiet windows: {} of {} ({} sampled; dropped as loud E/N/Z: {}/{}/{})",
                w.quiet, w.total, w.sampled, w.loud[0], w.loud[1], w.loud[2]
            ),
            format!(" tau         : {} samples (IQR {:.1}-{:.1})", self.tau, self.spread.tau_iqr[0], self.spread.tau_iqr[1]),
            format!(" dim         : {} (IQR {:.1}-{:.1})", self.dim, self.spread.dim_iqr[0], self.spread.dim_iqr[1]),
            format!(
                " mean period : {:.3} s (IQR {:.3}-{:.3})",
                self.mean_period, self.spread.mean_period_iqr[0], self.spread.mean_period_iqr[1]
            ),
            format!(
                " SampEn tol  : E {:.4e}, N {:.4e}, Z {:.4e} (r = {})",
                self.sampent_tol[0], self.sampent_tol[1], self.sampent_tol[2], self.sampent_r
            ),
        ]
    }
}

/// SplitMix64: a seeded generator for the reservoir, so a rerun picks the
/// same windows.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

fn station_location(config: &AppConfig) -> Result<(f64, f64), String> {
    let opts = &config.chaos_calibration;
    if let (Some(lat), Some(lon)) = (opts.station_lat, opts.station_lon) {
        return Ok((lat, lon));
    }
    catalog::station_coords(&opts.station_coords, &config.station)?.ok_or_else(|| {
        format!(
            "station {} is not in {:?}, so its quiet periods cannot be found. Set \
             chaos_calibration.station_lat/station_lon in the config, or pass \
             --calibration with another run's calibration",
            config.station, opts.station_coords
        )
    })
}

pub fn calibrate(config: &AppConfig, progress_tx: &Sender<PipelineEvent>) -> Result<Calibration, String> {
    let opts = &config.chaos_calibration;
    let say = |line: String| {
        let _ = progress_tx.send(PipelineEvent::Info(line));
    };

    let location = station_location(config)?;
    let events = catalog::load_events(&opts.catalog)?;
    let exclusions = Exclusions::new(&events, location, opts);
    say(format!(
        " Catalog: {} events, {} exclusion intervals within {} km of {} (M >= {})",
        events.len(),
        exclusions.len(),
        opts.max_dist_km,
        config.station,
        opts.min_mag
    ));

    let win = config.win_size;
    let mut counts = Counts::default();
    let mut rng = Rng(opts.seed);
    let mut reservoir: Vec<[Vec<f64>; 3]> = Vec::with_capacity(opts.max_windows);
    let mut fs = config.fs;
    let mut span = (f64::INFINITY, f64::NEG_INFINITY);

    let mut buf: [Vec<f64>; 3] = Default::default();
    let mut interp: [Vec<bool>; 3] = Default::default();
    let mut buf_epoch = f64::NAN;

    preprocess::preprocess_directory_chunked(&config.data_dir, config, |chunk| {
        fs = chunk.fs;
        let _ = progress_tx.send(PipelineEvent::Progress {
            done: chunk.records_done,
            total: chunk.records_total,
        });
        // A chunk that does not continue the buffer starts a new stretch, as
        // in the extraction pass.
        let expected = buf_epoch + buf[0].len() as f64 / fs;
        if buf_epoch.is_nan() || buf[0].is_empty() || (chunk.start_epoch - expected).abs() > 1.5 / fs {
            buf.iter_mut().for_each(Vec::clear);
            interp.iter_mut().for_each(Vec::clear);
            buf_epoch = chunk.start_epoch;
        }
        for (b, data) in buf.iter_mut().zip([chunk.e, chunk.n, chunk.z]) {
            b.extend(data);
        }
        for (b, mask) in interp.iter_mut().zip([chunk.e_interp, chunk.n_interp, chunk.z_interp]) {
            b.extend(mask);
        }

        let mut cursor = 0;
        while buf[0].len() - cursor >= win {
            let range = cursor..cursor + win;
            let t0 = buf_epoch + cursor as f64 / fs;
            let t1 = t0 + win as f64 / fs;
            cursor += win;
            counts.total += 1;
            span = (span.0.min(t0), span.1.max(t1));

            let clean = buf.iter().all(|b| b[range.clone()].iter().all(|v| v.is_finite()))
                && interp.iter().all(|m| !m[range.clone()].iter().any(|&f| f));
            if !clean || exclusions.overlaps(t0, t1) {
                continue;
            }
            counts.quiet += 1;
            // Algorithm R: after k quiet windows, each has had an equal
            // max_windows/k chance of being held.
            let window = || buf.each_ref().map(|b| b[range.clone()].to_vec());
            if reservoir.len() < opts.max_windows {
                reservoir.push(window());
            } else {
                let k = rng.below(counts.quiet);
                if k < opts.max_windows {
                    reservoir[k] = window();
                }
            }
        }
        buf.iter_mut().for_each(|b| drop(b.drain(..cursor)));
        interp.iter_mut().for_each(|m| drop(m.drain(..cursor)));
        buf_epoch += cursor as f64 / fs;
        Ok(())
    })?;

    counts.sampled = reservoir.len();
    if span.0.is_finite() && exclusions.count_within(span.0 - 86400.0, span.1 + 86400.0) == 0 {
        say(" Warning: no catalog event near the station falls within the data; check \
             that the catalog covers these dates"
            .to_string());
    }
    if reservoir.len() < MIN_QUIET_WINDOWS {
        return Err(format!(
            "only {} quiet windows of {} found (need {}); raise chaos_calibration.min_mag \
             or lower max_dist_km, or pass --calibration with another run's calibration",
            reservoir.len(),
            counts.total,
            MIN_QUIET_WINDOWS
        ));
    }

    // Loudness screening, per component: one channel can glitch alone.
    let stds: Vec<[f64; 3]> = reservoir.iter().map(|w| w.each_ref().map(|c| std_dev(c))).collect();
    let mut keep = vec![[true; 3]; reservoir.len()];
    let mut quiet_std = [0.0; 3];
    for c in 0..3 {
        let limit = opts.loud_factor * median(stds.iter().map(|s| s[c]).collect());
        for (k, s) in keep.iter_mut().zip(&stds) {
            k[c] = s[c] <= limit;
            counts.loud[c] += !k[c] as usize;
        }
        quiet_std[c] = median(stds.iter().zip(&keep).filter(|(_, k)| k[c]).map(|(s, _)| s[c]).collect());
    }

    let segments: Vec<(Vec<f64>, &[f64])> = reservoir
        .iter()
        .zip(&keep)
        .flat_map(|(w, k)| w.iter().zip(k).filter(|(_, k)| **k).map(|(c, _)| (zscore(c), c.as_slice())))
        .collect();
    counts.measured = segments.len();

    let per_segment: Vec<(Option<usize>, f64)> = segments
        .par_iter()
        .map(|(z, raw)| (chaos::ami_tau(z, opts.max_tau), 1.0 / mean_frequency(raw, fs)))
        .collect();
    let taus: Vec<f64> = per_segment.iter().filter_map(|(t, _)| t.map(|t| t as f64)).collect();
    counts.with_tau = taus.len();
    if taus.len() < MIN_QUIET_WINDOWS {
        return Err(format!(
            "the average mutual information had a minimum below max_tau = {} in only {} \
             quiet window-components; raise chaos_calibration.max_tau",
            opts.max_tau,
            taus.len()
        ));
    }
    let tau = (median(taus.clone()).round() as usize).max(1);

    let dims: Vec<f64> = segments
        .par_iter()
        .filter_map(|(z, _)| chaos::fnn_dim(z, tau, opts.max_dim).map(|d| d as f64))
        .collect();
    if dims.is_empty() {
        return Err("no quiet window was long enough for False Nearest Neighbours".to_string());
    }
    let periods: Vec<f64> = per_segment.iter().map(|&(_, p)| p).filter(|p| p.is_finite()).collect();

    Ok(Calibration {
        station: config.station.clone(),
        fs,
        tau,
        dim: (median(dims.clone()).round() as usize).max(1),
        mean_period: median(periods.clone()),
        sampent_r: config.chaos_sampent_r,
        sampent_tol: quiet_std.map(|s| config.chaos_sampent_r * s),
        spread: Spread {
            tau_iqr: iqr(taus),
            dim_iqr: iqr(dims),
            mean_period_iqr: iqr(periods),
            quiet_std,
        },
        windows: counts,
        station_lat: location.0,
        station_lon: location.1,
        exclusion_intervals: exclusions.len(),
        settings: opts.clone(),
    })
}

fn std_dev(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn zscore(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if sd > 0.0 { x.iter().map(|v| (v - mean) / sd).collect() } else { x.to_vec() }
}

/// Linear-interpolated percentile, as numpy's default.
fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

fn median(v: Vec<f64>) -> f64 {
    percentile(v, 50.0)
}

fn iqr(v: Vec<f64>) -> [f64; 2] {
    [percentile(v.clone(), 25.0), percentile(v, 75.0)]
}

/// Rosenstein's mean frequency: the power-weighted average over the
/// periodogram, DC excluded. A Hann taper keeps the window edges from leaking
/// power into the high frequencies and biasing the mean upwards.
fn mean_frequency(x: &[f64], fs: f64) -> f64 {
    let n = x.len();
    if n < 4 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut buf: Vec<Complex<f64>> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (n - 1) as f64).cos();
            Complex { re: (v - mean) * w, im: 0.0 }
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buf);
    let (mut num, mut den) = (0.0, 0.0);
    for (k, c) in buf.iter().enumerate().take(n / 2 + 1).skip(1) {
        let p = c.norm_sqr();
        num += k as f64 * fs / n as f64 * p;
        den += p;
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_frequency_of_a_sine_is_its_frequency() {
        let fs = 5.0;
        let x: Vec<f64> = (0..1000).map(|i| (std::f64::consts::TAU * 0.3 * i as f64 / fs).sin()).collect();
        let f = mean_frequency(&x, fs);
        assert!((f - 0.3).abs() < 0.01, "{f}");
    }

    #[test]
    fn percentiles_interpolate_like_numpy() {
        assert_eq!(median(vec![3.0, 1.0, 2.0, 4.0]), 2.5);
        assert_eq!(iqr(vec![1.0, 2.0, 3.0, 4.0, 5.0]), [2.0, 4.0]);
        assert!(median(Vec::new()).is_nan());
    }

    #[test]
    fn reservoir_rng_is_seeded_and_in_range() {
        let draw = |seed| {
            let mut r = Rng(seed);
            (0..50).map(|_| r.below(7)).collect::<Vec<_>>()
        };
        assert_eq!(draw(3), draw(3));
        assert_ne!(draw(3), draw(4));
        assert!(draw(5).iter().all(|&v| v < 7));
    }
}
