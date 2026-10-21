//! Zero-phase Butterworth bandpass as cascaded second-order sections.
//!
//! Replaces the `butterworth` crate, which builds the filter as one
//! transfer-function polynomial. At low corners relative to the sampling rate
//! that form loses the poles to rounding: at 100 Hz an order-4 0.02-0.1 Hz
//! bandpass gets a pole outside the unit circle and blows up, and 0.1-1 Hz
//! comes out 4% hot at band centre. Second-order sections keep each conjugate
//! pole pair in its own quadratic, which stays accurate at any corner.
//!
//! The design follows scipy's `butter(N, [lo, hi], 'band', fs=fs)`: analog
//! prototype, lowpass-to-bandpass transform, bilinear transform with
//! pre-warped corners. The filter has order 2N, and `filtfilt` applies it
//! forward and backward, like the crate's `bidirectional` did.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct C(f64, f64);

impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn scale(self, k: f64) -> C {
        C(self.0 * k, self.1 * k)
    }
    fn abs(self) -> f64 {
        self.0.hypot(self.1)
    }
    fn sqrt(self) -> C {
        let r = self.abs();
        let re = ((r + self.0) / 2.0).sqrt();
        let im = ((r - self.0) / 2.0).sqrt();
        C(re, if self.1 < 0.0 { -im } else { im })
    }
}

/// `[b0, b1, b2, a1, a2]`, with `a0 = 1`.
type Section = [f64; 5];

#[derive(Debug, Clone)]
pub struct BandPass {
    sections: Vec<Section>,
}

impl BandPass {
    /// Order-`order` Butterworth prototype (so a bandpass of order
    /// `2 * order`) passing `lo`..`hi` Hz at sampling rate `fs`.
    pub fn new(order: usize, fs: f64, lo: f64, hi: f64) -> Result<Self, String> {
        if order == 0 || order % 2 != 0 {
            return Err(format!("bandpass order must be even and positive, got {}", order));
        }
        if !(0.0 < lo && lo < hi && hi < fs / 2.0) {
            return Err(format!("bandpass needs 0 < {} < {} < Nyquist ({})", lo, hi, fs / 2.0));
        }
        let fs2 = 2.0 * fs;
        let wl = fs2 * (PI * lo / fs).tan();
        let wh = fs2 * (PI * hi / fs).tan();
        let bw = wh - wl;
        let wo2 = C(wl * wh, 0.0);
        let bilinear = |s: C| C(fs2, 0.0).add(s).div(C(fs2, 0.0).sub(s));

        // Upper-half-plane prototype poles; each conjugate partner yields the
        // conjugate digital poles, so one section per transformed pole.
        let mut sections = Vec::with_capacity(order);
        for k in 0..order / 2 {
            let theta = PI * (2 * k + 1 + order) as f64 / (2 * order) as f64;
            let p = C(theta.cos(), theta.sin()).scale(bw / 2.0);
            let root = p.mul(p).sub(wo2).sqrt();
            for s in [p.add(root), p.sub(root)] {
                let z = bilinear(s);
                // Zeros at z = +1 and z = -1: (1 - z^-1)(1 + z^-1).
                sections.push([1.0, 0.0, -1.0, -2.0 * z.0, z.0 * z.0 + z.1 * z.1]);
            }
        }

        // Unit gain at the digital image of the analog centre frequency, where
        // a Butterworth bandpass peaks at exactly 1.
        let w = 2.0 * (wl * wh).sqrt().atan2(fs2);
        let gain = 1.0 / response(&sections, w);
        for b in sections[0][..3].iter_mut() {
            *b *= gain;
        }
        Ok(BandPass { sections })
    }

    /// Magnitude response at `freq` Hz for sampling rate `fs`.
    #[cfg(test)]
    pub fn magnitude(&self, freq: f64, fs: f64) -> f64 {
        response(&self.sections, 2.0 * PI * freq / fs)
    }

    /// Zero-phase filtering, as scipy's `sosfiltfilt`: the signal is extended
    /// by `padlen` samples of odd reflection at each end so the edge
    /// transients fall on the extension, and each pass starts from the steady
    /// state for its first sample. A reflection is only a guess at what comes
    /// next, so callers that have real samples past an edge should filter
    /// them too and discard them afterwards.
    pub fn filtfilt(&self, x: &[f64], padlen: usize) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let pad = padlen.min(n - 1);
        let mut ext = Vec::with_capacity(n + 2 * pad);
        ext.extend((1..=pad).rev().map(|i| 2.0 * x[0] - x[i]));
        ext.extend_from_slice(x);
        ext.extend((1..=pad).map(|i| 2.0 * x[n - 1] - x[n - 1 - i]));

        self.forward(&mut ext);
        ext.reverse();
        self.forward(&mut ext);
        ext.reverse();
        ext[pad..pad + n].to_vec()
    }

    /// One causal pass in place, transposed direct form II per section.
    fn forward(&self, x: &mut [f64]) {
        // Steady state for a constant input of x[0]: each section starts as
        // if it had always seen the DC level the sections before pass to it.
        let mut dc = x[0];
        for &[b0, b1, b2, a1, a2] in &self.sections {
            let g = (b0 + b1 + b2) / (1.0 + a1 + a2);
            let mut z1 = (b2 - a2 * g) * dc;
            let mut z0 = (b1 - a1 * g) * dc + z1;
            dc *= g;
            for v in x.iter_mut() {
                let u = *v;
                let y = b0 * u + z0;
                z0 = b1 * u - a1 * y + z1;
                z1 = b2 * u - a2 * y;
                *v = y;
            }
        }
    }
}

fn response(sections: &[Section], w: f64) -> f64 {
    let z1 = C(w.cos(), -w.sin());
    let z2 = z1.mul(z1);
    sections
        .iter()
        .map(|&[b0, b1, b2, a1, a2]| {
            let num = C(b0, 0.0).add(z1.scale(b1)).add(z2.scale(b2));
            let den = C(1.0, 0.0).add(z1.scale(a1)).add(z2.scale(a2));
            num.div(den).abs()
        })
        .product()
}

/// Seconds for the impulse response of a bandpass with lower corner `freqmin`
/// to decay: its ringing lasts a few periods of that corner (measured on the
/// order-8 design: 176 s at 0.02 Hz, 26 s at 0.1 Hz).
pub fn settle_seconds(freqmin: f64) -> f64 {
    5.0 / freqmin
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact Butterworth bandpass magnitude after the bilinear transform:
    /// `1 / sqrt(1 + ((W^2 - Wo^2) / (W * B))^(2N))` at the pre-warped
    /// frequency `W`.
    fn ideal(freq: f64, fs: f64, lo: f64, hi: f64, order: i32) -> f64 {
        let warp = |f: f64| 2.0 * fs * (PI * f / fs).tan();
        let (w, wl, wh) = (warp(freq), warp(lo), warp(hi));
        let x = (w * w - wl * wh) / (w * (wh - wl));
        1.0 / (1.0 + x.powi(2 * order)).sqrt()
    }

    #[test]
    fn response_is_butterworth_for_every_band() {
        for &(fs, lo, hi) in &[(100.0, 0.02, 0.1), (100.0, 0.1, 1.0), (100.0, 0.1, 2.0), (100.0, 1.0, 5.0), (100.0, 5.0, 20.0), (5.0, 0.1, 2.0)] {
            let f = BandPass::new(4, fs, lo, hi).unwrap();
            for &freq in &[lo / 4.0, lo, (lo * hi).sqrt(), hi, (hi * 2.0).min(fs / 2.1)] {
                let (got, want) = (f.magnitude(freq, fs), ideal(freq, fs, lo, hi, 4));
                assert!((got - want).abs() < 1e-9, "{lo}-{hi} Hz at {freq}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn low_band_stays_stable_and_passes_its_centre() {
        let (fs, lo, hi) = (100.0, 0.02, 0.1);
        let f = BandPass::new(4, fs, lo, hi).unwrap();
        let f0 = (lo * hi).sqrt();
        let x: Vec<f64> = (0..(fs * 7200.0) as usize).map(|i| (2.0 * PI * f0 * i as f64 / fs).sin()).collect();
        let y = f.filtfilt(&x, (settle_seconds(lo) * fs) as usize);
        let mid = &y[y.len() / 4..3 * y.len() / 4];
        let peak = mid.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 1e-3, "peak {peak}");
    }

    #[test]
    fn filtfilt_is_zero_phase() {
        let fs = 100.0;
        let f = BandPass::new(4, fs, 0.1, 2.0).unwrap();
        let x: Vec<f64> = (0..60000).map(|i| (2.0 * PI * 0.5 * i as f64 / fs).sin()).collect();
        let y = f.filtfilt(&x, 5000);
        for i in (20000..40000).step_by(997) {
            assert!((y[i] - x[i]).abs() < 1e-3, "sample {i}: {} vs {}", y[i], x[i]);
        }
    }

    #[test]
    fn rejects_bad_corners() {
        assert!(BandPass::new(4, 5.0, 0.1, 2.6).is_err());
        assert!(BandPass::new(4, 5.0, 1.0, 0.5).is_err());
        assert!(BandPass::new(3, 5.0, 0.1, 2.0).is_err());
    }
}
