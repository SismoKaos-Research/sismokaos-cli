//! Nonlinear (chaotic) time-series metrics: Wolf and Rosenstein maximum
//! Lyapunov exponents, Sample Entropy, and Correlation Dimension.
//!
//! Ported from the Python sibling project's `sismokaos/chaos_algorithms.py`
//! (itself adapted from S. Sarwar, A. Likens, N. Stergiou, S. Mastorakis,
//! "A nonlinear analysis software toolkit for biomechanical data",
//! arXiv:2311.06723, 2023), so results stay comparable between the two
//! implementations.

/// Maximum Lyapunov exponent via Wolf (1985).
///
/// `x` must already be z-score normalized. Returns `NaN` if there isn't
/// enough data to embed at the requested `tau`/`dim`/`evolve`.
pub fn wolf_lye(x: &[f64], fs: f64, tau: usize, dim: usize, evolve: usize) -> f64 {
    let n = x.len();
    if dim == 0 || (dim - 1) * tau + evolve >= n {
        return f64::NAN;
    }

    let min_v = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale_mx = (max_v - min_v) / 10.0;
    let dt = 1.0 / fs;

    let m = n - (dim - 1) * tau;
    let mut y = vec![0.0; m * dim];
    for i in 0..dim {
        for j in 0..m {
            y[j * dim + i] = x[i * tau + j];
        }
    }

    let npt = n - (dim - 1) * tau - evolve;
    if npt == 0 {
        return f64::NAN;
    }
    let y_len = (npt + evolve).min(m);
    fn point(y: &[f64], dim: usize, idx: usize) -> &[f64] {
        &y[idx * dim..idx * dim + dim]
    }
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(&p, &q)| (p - q).powi(2)).sum::<f64>().sqrt()
    };

    // Initial neighbour pair: nearest point to index 0, excluding a small
    // temporal window around it.
    let mut current_pair = {
        let mut best_idx = usize::MAX;
        let mut best_dist = f64::INFINITY;
        for i in 0..npt.min(y_len) {
            if i <= 10 {
                continue;
            }
            let d = dist(point(&y, dim, 0), point(&y, dim, i));
            if d > 0.0 && d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        if best_idx == usize::MAX {
            return f64::NAN;
        }
        best_idx
    };

    let mut its: u64 = 0;
    let mut dist_sum = 0.0;
    let mut lye = 0.0;
    let mut i = 0;
    while i < npt {
        let ep = i + evolve;
        if ep >= y_len {
            break;
        }
        let safe = current_pair + evolve < y_len;
        let pair_ep = if safe { current_pair + evolve } else { current_pair + evolve - 1 };
        if pair_ep >= y_len {
            break;
        }

        let start_dist = dist(point(&y, dim, i), point(&y, dim, current_pair));
        let end_dist = dist(point(&y, dim, ep), point(&y, dim, pair_ep));

        if start_dist > 0.0 && end_dist > 0.0 {
            dist_sum += (end_dist / start_dist).log2() / (evolve as f64 * dt);
            its += 1;
            lye = dist_sum / its as f64;
        }

        if end_dist < scale_mx {
            current_pair += evolve;
            if current_pair > npt {
                current_pair -= evolve;
                current_pair = wolf_next_point(true, &y, dim, i, current_pair, npt, y_len, evolve, scale_mx);
            }
        } else {
            current_pair = wolf_next_point(false, &y, dim, i, current_pair, npt, y_len, evolve, scale_mx);
        }

        i += evolve;
    }

    lye
}

/// Selects the next nearest neighbour for the Wolf algorithm. Mirrors
/// `_wolf_next_point` in the Python port.
///
/// `flag1` corresponds to the Python function's `flag == 1` call (made when
/// the running neighbour pair index overflowed `NPT`): it skips straight to
/// the nearest valid neighbour by distance alone, ignoring the angle
/// criterion entirely. `flag1 == false` (Python's `flag == 0`, made when the
/// trajectory pair diverged past `SCALEMX`) first tries the nearest neighbour
/// *within* the 30-degree angle limit, accepting it only if it's also within
/// `SCALEMX`; otherwise it falls through to the same angle-agnostic nearest
/// neighbour as `flag1`.
///
/// Note: in the reference Python implementation the angle limit passed
/// between calls is only ever reset to 30 degrees or left unchanged, so it
/// never actually varies across a run -- it's hardcoded as a constant here
/// rather than threaded through as mutable state.
fn wolf_next_point(
    flag1: bool,
    y: &[f64],
    dim: usize,
    current_point: usize,
    current_point_pair: usize,
    npt: usize,
    y_len: usize,
    evolve: usize,
    scale_mx: f64,
) -> usize {
    const ANGLE_LIMIT_DEG: f64 = 30.0;

    let point = |idx: usize| -> &[f64] { &y[idx * dim..idx * dim + dim] };
    let ep = current_point + evolve;
    if ep >= y_len {
        return current_point_pair;
    }

    let safe = current_point_pair + evolve < y_len;
    let end_idx = if safe { current_point_pair + evolve } else { current_point_pair + evolve - 1 };
    if end_idx >= y_len {
        return current_point_pair;
    }

    let pe = point(ep);
    let pair = point(end_idx);
    let mut end_sq = 0.0;
    for t in 0..dim {
        let v = pe[t] - pair[t];
        end_sq += v * v;
    }
    let end_dist = end_sq.sqrt();

    let bound = npt.min(y_len);
    let in_excl_window = |k: usize| k + 10 >= ep && k <= ep + 10;

    // One sweep finds both the nearest neighbour within the angle limit and
    // the nearest overall; they used to be two sweeps with a `Vec` allocated
    // per candidate. Sums accumulate in the same order as before, so every
    // distance, and therefore every pick, is bit-identical.
    let mut best_angle: Option<(usize, f64)> = None;
    let mut best_idx = usize::MAX;
    let mut best_d = f64::INFINITY;
    for k in 0..bound {
        if in_excl_window(k) {
            continue;
        }
        let pk = point(k);
        let mut sq = 0.0;
        for t in 0..dim {
            let d = pe[t] - pk[t];
            sq += d * d;
        }
        let yd = sq.sqrt();
        if yd <= 0.0 {
            continue;
        }
        if yd < best_d {
            best_d = yd;
            best_idx = k;
        }
        // The angle only matters for a candidate nearer than the best so far.
        if flag1 || best_angle.is_some_and(|(_, bd)| yd >= bd) {
            continue;
        }
        let denom = yd * end_dist;
        if denom <= 0.0 {
            continue;
        }
        let mut dot = 0.0;
        for t in 0..dim {
            dot += (pe[t] - pair[t]) * (pe[t] - pk[t]);
        }
        let theta = (dot / denom).abs().clamp(-1.0, 1.0).acos();
        if theta < ANGLE_LIMIT_DEG.to_radians() {
            best_angle = Some((k, yd));
        }
    }

    if let Some((idx, d)) = best_angle {
        if d <= scale_mx {
            return idx;
        }
    }
    // Otherwise the angle-agnostic nearest neighbour, exactly as the Python
    // `next_pt == -1` fallback does.
    if best_idx == usize::MAX { current_point_pair } else { best_idx }
}

/// Rosenstein (1993) mean log divergence. Element `j` is the average, over
/// every point whose trajectory and its nearest neighbour's both survive `j`
/// steps, of ln(their distance after `j` steps).
///
/// `x` must already be z-score normalized. Neighbours closer in time than
/// `band` samples are excluded. The curve is followed up to step `jmax` and
/// ends early at the first step no pair survives, so every element is a real
/// average.
pub fn divergence_curve(x: &[f64], tau: usize, dim: usize, band: usize, jmax: usize) -> Vec<f64> {
    let n = x.len();
    if dim == 0 || (dim - 1) * tau >= n {
        return Vec::new();
    }
    let m = n - (dim - 1) * tau;
    if m < 2 {
        return Vec::new();
    }

    let mut y = vec![0.0; m * dim];
    for j in 0..dim {
        for i in 0..m {
            y[i * dim + j] = x[j * tau + i];
        }
    }
    let point = |idx: usize| -> &[f64] { &y[idx * dim..idx * dim + dim] };

    // Nearest neighbour (excluding a temporal band around the diagonal) for
    // every embedded point.
    //
    // The distance matrix is symmetric and `(a - b).powi(2) == (b - a).powi(2)`
    // exactly in IEEE, so each unordered pair is evaluated once and used to
    // update both of its rows -- half the distance work. Scan order is
    // preserved: row `r` still sees its candidates in ascending index order
    // (those below `r` arrive from earlier outer iterations, those above from
    // outer iteration `r`), and the update stays strictly `<`, so the same
    // first-minimum wins every tie the old loop did.
    let mut best_idx = vec![usize::MAX; m];
    let mut best_d = vec![f64::INFINITY; m];
    for i in 0..m {
        let pi = point(i);
        for k in (i + band + 1)..m {
            let pk = point(k);
            let mut d = 0.0f64;
            for t in 0..dim {
                let e = pi[t] - pk[t];
                d += e * e;
            }
            if d < best_d[i] {
                best_d[i] = d;
                best_idx[i] = k;
            }
            if d < best_d[k] {
                best_d[k] = d;
                best_idx[k] = i;
            }
        }
    }
    let mut ind2 = vec![0usize; m];
    for i in 0..m {
        ind2[i] = if best_idx[i] == usize::MAX { i } else { best_idx[i] };
    }

    // Only the steps the caller reads are computed: the loop is triangular,
    // so following the curve to its end would cost as much again as the
    // neighbour search.
    let mut curve = Vec::with_capacity(jmax.min(m - 1) + 1);
    for j in 0..=jmax.min(m - 1) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..m {
            let nn = ind2[i];
            if i + j >= m || nn + j >= m {
                continue;
            }
            let d: f64 = point(i + j).iter().zip(point(nn + j)).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
            if d > 0.0 {
                sum += d.ln();
                count += 1;
            }
        }
        if count == 0 {
            break;
        }
        curve.push(sum / count as f64);
    }
    curve
}

/// Short- and long-term Lyapunov exponents via Rosenstein (1993), with the fit
/// range picked from each window's own divergence curve.
///
/// `x` must already be z-score normalized. Neighbours closer than one mean
/// period (or the embedding span, if longer) are excluded, and the curve is
/// followed for `horizon` mean periods. See [`rise_fit`] for how `rise`
/// selects the fit. Returns `(short, long)` in units of 1/mean period; either
/// is `NaN` when the curve has no usable range for it.
pub fn rosenstein_lye(
    x: &[f64],
    fs: f64,
    tau: usize,
    dim: usize,
    mean_period: f64,
    horizon: f64,
    rise: [f64; 2],
) -> (f64, f64) {
    if dim == 0 || !(mean_period > 0.0) {
        return (f64::NAN, f64::NAN);
    }
    let band = ((dim - 1) * tau).max((mean_period * fs).ceil() as usize);
    let jmax = python_round(horizon * mean_period * fs) as usize;
    let curve = divergence_curve(x, tau, dim, band, jmax);
    rise_fit(&curve, fs, mean_period, rise)
}

/// Fits a divergence curve over a fraction of its rise.
///
/// The curve climbs from the initial neighbour separation and flattens once
/// the neighbours have spread across the attractor. Its rise is the climb
/// from the first step to the peak. The short exponent is the slope between
/// the first steps reaching `rise[0]` and `rise[1]` of that climb; the long
/// exponent is the slope from the `rise[1]` step to the end of the curve,
/// i.e. over the saturation.
pub fn rise_fit(curve: &[f64], fs: f64, mean_period: f64, rise: [f64; 2]) -> (f64, f64) {
    let nan = (f64::NAN, f64::NAN);
    if curve.len() < 2 {
        return nan;
    }
    let start = curve[0];
    let peak = curve.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let climb = peak - start;
    if !(climb > 0.0) {
        return nan;
    }
    let reach = |frac: f64, from: usize| (from..curve.len()).find(|&j| curve[j] >= start + frac * climb);
    // Both exist for fractions up to 1, since the peak itself reaches them,
    // and a step reaching `rise[1]` reaches `rise[0]` too, so `hi >= lo`.
    let (Some(lo), Some(hi)) = (reach(rise[0], 0), reach(rise[1], 0)) else {
        return nan;
    };

    let time: Vec<f64> = (0..curve.len()).map(|i| i as f64 / fs / mean_period).collect();
    let fit = |a: usize, b: usize| if b > a { linear_regression_slope(&time[a..=b], &curve[a..=b]) } else { f64::NAN };
    (fit(lo, hi), fit(hi, curve.len() - 1))
}

/// Rounds like Python 3's built-in `round()` (round-half-to-even), not
/// Rust's `f64::round()` (round-half-away-from-zero). The two disagree
/// exactly at `.5` boundaries -- e.g. `round(2.5)` is `2` in Python but
/// `3.0` in Rust -- which is precisely where the Rosenstein slope-window
/// bounds land with the default `slope`/`mean_period`/`fs` values. Assumes
/// a non-negative input, which is all this module ever rounds.
fn python_round(v: f64) -> f64 {
    let floor = v.floor();
    let diff = v - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Ordinary least-squares slope of y against x.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        num += (xi - mean_x) * (yi - mean_y);
        den += (xi - mean_x).powi(2);
    }
    if den == 0.0 { f64::NAN } else { num / den }
}

/// Sample Entropy, using Chebyshev (max-norm) distance between
/// `m`/`m+1`-length templates. `data` must already be z-score normalized;
/// `r` is applied as a multiple of the segment's own std.
pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n <= m + 1 {
        return f64::NAN;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let std = (data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    sample_entropy_tol(data, m, r * std)
}

/// Sample Entropy at an absolute tolerance `tol`, in the units of `data`.
/// With `tol` fixed for a whole run, windows are compared on one amplitude
/// scale instead of each being scaled by its own std.
pub fn sample_entropy_tol(data: &[f64], m: usize, tol: f64) -> f64 {
    let n = data.len();
    if n <= m + 1 || !(tol > 0.0) {
        return f64::NAN;
    }

    let l = n - m;
    let mut b_count: u64 = 0;
    let mut a_count: u64 = 0;

    // The Chebyshev distance is symmetric and the i == j case is skipped, so
    // the original ordered double loop visited each unordered pair twice and
    // got the same answer both times. Walking only j > i and incrementing by
    // two leaves both counts -- and therefore their ratio -- unchanged, for
    // half the work.
    for i in 0..l {
        for j in (i + 1)..l {
            let mut dist_m: f64 = 0.0;
            for k in 0..m {
                let d = (data[i + k] - data[j + k]).abs();
                if d > dist_m {
                    dist_m = d;
                }
            }
            if dist_m <= tol {
                b_count += 2;
                let dist_m1 = dist_m.max((data[i + m] - data[j + m]).abs());
                if dist_m1 <= tol {
                    a_count += 2;
                }
            }
        }
    }

    if a_count == 0 || b_count == 0 {
        return f64::NAN;
    }
    -((a_count as f64) / (b_count as f64)).ln()
}

thread_local! {
    static DISTS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The largest `q` with `q.sqrt() <= e`, for a positive finite `e`.
fn largest_square_below(e: f64) -> f64 {
    let mut q = e * e;
    while q > 0.0 && q.sqrt() > e {
        q = q.next_down();
    }
    while q.next_up().sqrt() <= e {
        q = q.next_up();
    }
    q
}

/// Correlation dimension estimate via a Grassberger-Procaccia style
/// correlation-integral slope fit, ported from the MATLAB `corrdim.m`
/// derived implementation used by the Python sibling project.
pub fn correlation_dimension(x: &[f64], tau: usize, de: usize) -> f64 {
    let n = x.len();
    if de == 0 || (de - 1) * tau >= n {
        return 0.0;
    }
    let sample_count = n - (de - 1) * tau;
    if sample_count == 0 {
        return 0.0;
    }

    // Row-major embedding, so point `idx` is the contiguous slice
    // `y[idx * de .. idx * de + de]`. The layout used to be column-major, which
    // meant reading one point had to gather `de` strided elements into a fresh
    // `Vec` -- inside a loop that runs once per *pair*, i.e. ~455k allocations
    // per component per window at the 200 s / 5 Hz configuration.
    let mut y = vec![0.0; de * sample_count];
    for j in 0..sample_count {
        for i in 0..de {
            y[j * de + i] = x[i * tau + j];
        }
    }
    let point = |idx: usize| -> &[f64] { &y[idx * de..idx * de + de] };

    let bins = 200usize;
    let k = de * tau;
    if k + 1 >= sample_count {
        return 0.0;
    }
    let pair_count = sample_count - k - 1;

    // Pass 1 keeps *squared* distances. `sqrt` is monotonic and correctly
    // rounded, so min/max commute with it -- taking the root of the extremes
    // gives bit-identical `eps1`/`eps2` while deferring 455k square roots to
    // the single pass that actually needs them.
    let mut total = 0usize;
    for i in 0..pair_count {
        total += sample_count - (i + k + 1);
    }

    // Reused across calls on the same thread: a fresh 3.6 MB buffer per call
    // cost a page fault for every 4 KiB of it.
    let mut dists = DISTS.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    dists.clear();
    dists.reserve(total);
    let mut max_sq = 0.0f64;
    let mut min_sq = f64::INFINITY;
    for i in 0..pair_count {
        let pi = point(i);
        for j in i + k + 1..sample_count {
            let pj = point(j);
            let mut acc = 0.0;
            for t in 0..de {
                let d = pi[t] - pj[t];
                acc += d * d;
            }
            dists.push(acc);
            if acc > max_sq {
                max_sq = acc;
            }
            if acc < min_sq {
                min_sq = acc;
            }
        }
    }
    let eps1 = max_sq.sqrt();
    let mut eps2 = min_sq.sqrt();
    if eps2 == 0.0 {
        eps2 = f64::EPSILON;
    }
    if eps1 <= eps2 {
        return 0.0;
    }

    let log_eps2 = eps2.ln();
    let log_eps1 = eps1.ln();
    let epsilon: Vec<f64> = (0..bins)
        .map(|b| (log_eps2 + (log_eps1 - log_eps2) * b as f64 / (bins - 1) as f64).exp())
        .collect();

    // `ci` is a *global* histogram: the old per-row loop accumulated
    // `count_b - count_{b-1}`, i.e. the number of that row's distances in
    // `(epsilon[b-1], epsilon[b]]`, into a shared `ci[b]`. The row structure
    // never affected the result, so each distance can be binned directly.
    // `partition_point(|&e| e < d)` returns the first `b` with
    // `epsilon[b] >= d` -- exactly the bin the sort-and-count assigned. A
    // return of `bins` means `d > epsilon[bins-1]`, which the old code
    // dropped implicitly (its per-row count never reached such a `d`); that
    // is reachable because `epsilon[bins-1]` is `exp(ln(eps1))`, which can
    // round just below `eps1` itself.
    //
    // That search, and a square root, ran once per pair and were 85% of this
    // function. The same bin is now found without either:
    //
    // * `sqrt` is correctly rounded and monotonic, so `epsilon[b] >= sqrt(sq)`
    //   holds exactly when `sq <= sq_edge[b]`, the largest square whose root
    //   does not exceed `epsilon[b]`.
    // * Positive doubles order like their bit patterns, so a table over the
    //   high bits of `sq` gives, for every `sq` in a cell, a bin at or below
    //   the answer: no edge before it can reach even the cell's smallest
    //   value. Stepping forward past the edges `sq` exceeds lands on the first
    //   `b` with `sq <= sq_edge[b]`, which is what the search returned. The
    //   bins are log-spaced and the cells nearly so, so that is a step or two.
    let sq_edge: Vec<f64> = epsilon.iter().map(|&e| largest_square_below(e)).collect();
    let base = sq_edge[0].to_bits();
    let span = max_sq.to_bits().saturating_sub(base);
    let shift = (64 - span.leading_zeros()).saturating_sub(13);
    let cells = (span >> shift) as usize + 1;
    let first_bin: Vec<u16> = (0..cells)
        .map(|c| {
            let lowest = f64::from_bits(base + ((c as u64) << shift));
            sq_edge.partition_point(|&e| e < lowest) as u16
        })
        .collect();

    let mut ci = vec![0.0f64; bins];
    for &sq in dists.iter() {
        let bits = sq.to_bits();
        let mut b = if bits <= base { 0 } else { first_bin[((bits - base) >> shift) as usize] as usize };
        while b < bins && sq > sq_edge[b] {
            b += 1;
        }
        if b < bins {
            ci[b] += 1.0;
        }
    }
    DISTS.with(|cell| *cell.borrow_mut() = dists);

    // Cumulative sum -> correlation integral -> log-log curve.
    let denom = (sample_count - k) as f64 * (sample_count - k) as f64;
    let mut cum = 0.0;
    let mut curve = vec![f64::NAN; bins];
    for b in 0..bins {
        cum += ci[b];
        if cum > 0.0 {
            curve[b] = (cum / denom).ln() + ((sample_count - k - 1) as f64).ln();
        }
    }
    let log_eps: Vec<f64> = epsilon.iter().map(|e| e.ln()).collect();

    let finite: Vec<(f64, f64)> = log_eps
        .iter()
        .zip(curve.iter())
        .filter(|(_, v)| v.is_finite())
        .map(|(&e, &v)| (e, v))
        .collect();
    if finite.is_empty() {
        return 0.0;
    }

    let lo = finite.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min);
    let hi = finite.iter().map(|&(_, v)| v).fold(f64::NEG_INFINITY, f64::max);
    let mid = (lo + hi) / 2.0;
    let q = (hi - lo) / 4.0;

    let selected: Vec<(f64, f64)> = finite
        .into_iter()
        .filter(|&(_, v)| v > mid && v < mid + q)
        .collect();
    if selected.len() < 2 {
        return 0.0;
    }

    let xs: Vec<f64> = selected.iter().map(|&(e, _)| e).collect();
    let ys: Vec<f64> = selected.iter().map(|&(_, v)| v).collect();
    linear_regression_slope(&xs, &ys)
}

/// Embedding delay as the first minimum of the average mutual information
/// between `x[t]` and `x[t + lag]` (Fraser & Swinney 1986), with histogram
/// bins from Scott's rule. Port of `ami_core` in the Python sibling project.
/// `None` if the AMI has no minimum below `max_lag`.
pub fn ami_tau(x: &[f64], max_lag: usize) -> Option<usize> {
    let n = x.len();
    if max_lag < 3 || n <= max_lag + 1 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let std = (x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let range = x.iter().copied().fold(f64::NEG_INFINITY, f64::max) - lo;
    if !(std > 0.0 && range > 0.0) {
        return None;
    }
    let bins = (range / (3.49 * std * (n as f64).powf(-1.0 / 3.0))).ceil() as usize;
    if bins < 2 {
        return None;
    }
    let width = range / (bins as f64 - f64::EPSILON);
    let y: Vec<usize> = x.iter().map(|&v| (((v - lo) / width) as usize).min(bins - 1)).collect();

    let overlap = n - max_lag;
    let p = 1.0 / overlap as f64;
    let mut pa = vec![0.0; bins];
    for &a in &y[..overlap] {
        pa[a] += p;
    }
    let mut ami = Vec::with_capacity(max_lag);
    let mut pb = vec![0.0; bins];
    let mut pab = vec![0.0; bins * bins];
    for lag in 0..max_lag {
        pb.fill(0.0);
        pab.fill(0.0);
        for (&a, &b) in y[..overlap].iter().zip(&y[lag..lag + overlap]) {
            pb[b] += p;
            pab[a * bins + b] += p;
        }
        let mut v = 0.0;
        for a in 0..bins {
            for b in 0..bins {
                let joint = pab[a * bins + b];
                if joint > 0.0 {
                    v += joint * (joint / (pa[a] * pb[b])).log2();
                }
            }
        }
        ami.push(v);
    }
    (1..max_lag - 1).find(|&i| ami[i - 1] >= ami[i] && ami[i] <= ami[i + 1])
}

/// Embedding dimension by False Nearest Neighbours (Kennel et al. 1992). Port
/// of `fnn_core` in the Python sibling project with its defaults (`rtol` 15,
/// `atol` 2, stopping at the first minimum), using an exhaustive neighbour
/// search in place of its kd-tree. `None` if `x` is too short to embed in
/// `max_dim` dimensions.
pub fn fnn_dim(x: &[f64], tau: usize, max_dim: usize) -> Option<usize> {
    const RTOL: f64 = 15.0;
    const ATOL: f64 = 2.0;
    if max_dim == 0 || tau * max_dim + 2 > x.len() {
        return None;
    }
    let n = x.len() - tau * max_dim;
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let ra = (x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    if !(ra > 0.0) {
        return None;
    }

    let mut false_frac: Vec<f64> = Vec::with_capacity(max_dim);
    for j in 0..max_dim {
        let d = j + 1;
        let mut false_count = 0usize;
        for i in 0..n {
            let mut best = f64::INFINITY;
            let mut nn = i;
            for k in 0..n {
                if k == i {
                    continue;
                }
                let mut acc = 0.0;
                for t in 0..d {
                    let e = x[i + t * tau] - x[k + t * tau];
                    acc += e * e;
                    if acc >= best {
                        break;
                    }
                }
                if acc < best {
                    best = acc;
                    nn = k;
                }
            }
            let dist = best.sqrt();
            let dz = x[i + d * tau] - x[nn + d * tau];
            if dz.abs() > dist * RTOL || (best + dz * dz).sqrt() / ra > ATOL {
                false_count += 1;
            }
        }
        false_frac.push(false_count as f64 / n as f64);

        let f = &false_frac;
        if j >= 2 && f[j - 2] > f[j - 1] && f[j - 1] < f[j] {
            return Some(j);
        }
        if j >= 1 && (f[j] - f[j - 1]).abs() <= 0.001 {
            return Some(j);
        }
        if f[j] == 0.0 {
            return Some(j + 1);
        }
    }
    Some(max_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic (RNG-free) 1000-sample signal, reproducible bit-for-bit
    /// in Python (`np.sin(i*0.13) + 0.5*np.sin(i*0.031)`) so the two sibling
    /// implementations can be checked against the same reference values
    /// without needing to match PRNGs across languages.
    fn reference_signal() -> Vec<f64> {
        let n = 1000;
        let raw: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.13).sin() + 0.5 * (i as f64 * 0.031).sin())
            .collect();
        let mean = raw.iter().sum::<f64>() / n as f64;
        let std = (raw.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
        raw.iter().map(|&v| (v - mean) / std).collect()
    }

    // Reference values computed by sismokaos/chaos_algorithms.py (the Python
    // sibling project) on the exact same `reference_signal()` above, with
    // fs=5.0, tau=5, dim=5, evolve=5, slope=[0.5,5.0,5.0,20.0],
    // mean_period=1.0, sample-entropy m=2/r=0.2.
    const REF_WOLF_LYE: f64 = 0.3035492535457496;
    const REF_ROS_SHORT: f64 = 0.38280772273173597;
    const REF_ROS_LONG: f64 = -0.029816027419789917;
    const REF_SAMP_ENT: f64 = 0.39819536425420143;
    const REF_CORR_DIM: f64 = 2.3299697367683927;

    fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: expected {expected}, got {actual} (tolerance {tol})"
        );
    }

    #[test]
    fn wolf_lye_matches_python_reference() {
        let x = reference_signal();
        let got = wolf_lye(&x, 5.0, 5, 5, 5);
        assert_close(got, REF_WOLF_LYE, 1e-10, "wolf_lye");
    }

    #[test]
    fn divergence_curve_matches_python_reference() {
        let x = reference_signal();
        let (short, long) = super::parity::fixed_window_lye(&x, 5.0, 5, 5, [0.5, 5.0, 5.0, 20.0], 1.0);
        assert_close(short, REF_ROS_SHORT, 1e-10, "ros_short");
        assert_close(long, REF_ROS_LONG, 1e-10, "ros_long");
    }

    #[test]
    fn rise_fit_reads_the_rise_between_the_fractions() {
        // Climbs one unit per step to 10, then stays flat.
        let curve: Vec<f64> = (0..=30).map(|j| (j as f64).min(10.0)).collect();
        let (short, long) = rise_fit(&curve, 1.0, 1.0, [0.1, 0.9]);
        assert_close(short, 1.0, 1e-12, "short");
        assert!(long > 0.0 && long < 0.2, "long {long}");
        // Halving the mean period halves the time axis' units.
        let (short, _) = rise_fit(&curve, 1.0, 0.5, [0.1, 0.9]);
        assert_close(short, 0.5, 1e-12, "short per mean period");
    }

    #[test]
    fn rise_fit_is_nan_without_a_rise() {
        let (s, l) = rise_fit(&[1.0; 20], 1.0, 1.0, [0.1, 0.9]);
        assert!(s.is_nan() && l.is_nan());
        let falling: Vec<f64> = (0..20).map(|j| -(j as f64)).collect();
        let (s, l) = rise_fit(&falling, 1.0, 1.0, [0.1, 0.9]);
        assert!(s.is_nan() && l.is_nan());
    }

    #[test]
    fn rosenstein_finds_the_logistic_map_chaotic() {
        let x = logistic_map();
        // One step per mean period, so the exponent is per iteration: ln 2.
        let (short, _) = rosenstein_lye(&x, 1.0, 1, 2, 1.0, 20.0, [0.1, 0.9]);
        assert!(short > 0.4 && short < 0.9, "short {short}");
    }

    #[test]
    fn sample_entropy_tol_matches_relative_r() {
        let x = reference_signal();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt();
        assert_eq!(sample_entropy_tol(&x, 2, 0.2 * sd).to_bits(), sample_entropy(&x, 2, 0.2).to_bits());
        // Scaling the signal at a fixed tolerance changes the answer; that is
        // the point of an absolute tolerance.
        let loud: Vec<f64> = x.iter().map(|v| v * 3.0).collect();
        assert_ne!(sample_entropy_tol(&loud, 2, 0.2 * sd), sample_entropy_tol(&x, 2, 0.2 * sd));
    }

    fn logistic_map() -> Vec<f64> {
        let mut v = 0.3;
        let raw: Vec<f64> = (0..1000)
            .map(|_| {
                v = 4.0 * v * (1.0 - v);
                v
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / raw.len() as f64;
        let sd = (raw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / raw.len() as f64).sqrt();
        raw.iter().map(|r| (r - mean) / sd).collect()
    }

    // `ami_core` and `fnn_core` of the Python sibling project on the same
    // signals. (A pure sine is no test: its exactly repeating samples give the
    // histogram AMI spurious early minima, in both implementations alike.)
    #[test]
    fn ami_tau_matches_python_reference() {
        let x = reference_signal();
        assert_eq!(ami_tau(&x, 20), Some(15));
        assert_eq!(ami_tau(&x, 40), Some(15));
        assert_eq!(ami_tau(&logistic_map(), 20), Some(8));
        assert_eq!(ami_tau(&[1.0; 100], 20), None);
    }

    #[test]
    fn fnn_dim_matches_python_reference() {
        let x = reference_signal();
        assert_eq!(fnn_dim(&x, 3, 10), Some(3));
        assert_eq!(fnn_dim(&x, 5, 10), Some(3));
        assert_eq!(fnn_dim(&x, 12, 10), Some(4));
        assert_eq!(fnn_dim(&logistic_map(), 1, 10), Some(1));
        assert_eq!(fnn_dim(&x[..20], 10, 8), None);
    }

    #[test]
    fn sample_entropy_matches_python_reference() {
        let x = reference_signal();
        let got = sample_entropy(&x, 2, 0.2);
        assert_close(got, REF_SAMP_ENT, 1e-9, "samp_ent");
    }

    #[test]
    fn correlation_dimension_matches_python_reference() {
        let x = reference_signal();
        let got = correlation_dimension(&x, 5, 5);
        assert_close(got, REF_CORR_DIM, 1e-10, "corr_dim");
    }

    #[test]
    fn functions_return_nan_or_zero_on_too_short_input() {
        let short = vec![1.0, 2.0, 3.0];
        assert!(wolf_lye(&short, 5.0, 5, 5, 5).is_nan());
        let (s, l) = rosenstein_lye(&short, 5.0, 5, 5, 1.0, 20.0, [0.1, 0.9]);
        assert!(s.is_nan() && l.is_nan());
        assert!(sample_entropy(&short, 2, 0.2).is_nan());
        assert_eq!(correlation_dimension(&short, 5, 5), 0.0);
    }
}

/// Bit-parity harness for the optimised `correlation_dimension` and the
/// Rosenstein divergence curve.
///
/// Both rewrites are performance-only: they must reproduce the pre-optimisation
/// output *exactly*, because the feature corpus is extracted incrementally and a
/// parquet written by an older binary has to stay comparable to one written by a
/// newer one. Approximate agreement is not sufficient for that, so these tests
/// hold the original implementations verbatim and compare raw bit patterns.
#[cfg(test)]
mod parity {
    #![allow(clippy::needless_range_loop)]
    use super::*;

    fn correlation_dimension_ref(x: &[f64], tau: usize, de: usize) -> f64 {
        let n = x.len();
        if de == 0 || (de - 1) * tau >= n {
            return 0.0;
        }
        let sample_count = n - (de - 1) * tau;
        if sample_count == 0 {
            return 0.0;
        }

        // Row-major embedding, so point `idx` is the contiguous slice
        // `y[idx * de .. idx * de + de]`. The layout used to be column-major, which
        // meant reading one point had to gather `de` strided elements into a fresh
        // `Vec` -- inside a loop that runs once per *pair*, i.e. ~455k allocations
        // per component per window at the 200 s / 5 Hz configuration.
        let mut y = vec![0.0; de * sample_count];
        for j in 0..sample_count {
            for i in 0..de {
                y[j * de + i] = x[i * tau + j];
            }
        }
        let point = |idx: usize| -> &[f64] { &y[idx * de..idx * de + de] };

        let bins = 200usize;
        let k = de * tau;
        if k + 1 >= sample_count {
            return 0.0;
        }
        let pair_count = sample_count - k - 1;

        // Every pair distance is used twice: once to find the eps range, once to
        // bin. Computing them into a flat triangular buffer trades ~3.6 MB per
        // call for halving the distance work and dropping the per-row allocation.
        let mut row_start = Vec::with_capacity(pair_count + 1);
        let mut total = 0usize;
        for i in 0..pair_count {
            row_start.push(total);
            total += sample_count - (i + k + 1);
        }
        row_start.push(total);

        let mut dists = vec![0.0f64; total];
        let mut eps1 = 0.0f64;
        let mut eps2 = f64::INFINITY;
        for i in 0..pair_count {
            let pi = point(i);
            let base = row_start[i];
            for (o, j) in (i + k + 1..sample_count).enumerate() {
                let pj = point(j);
                let mut acc = 0.0;
                for t in 0..de {
                    let d = pi[t] - pj[t];
                    acc += d * d;
                }
                let d = acc.sqrt();
                dists[base + o] = d;
                if d > eps1 {
                    eps1 = d;
                }
                if d < eps2 {
                    eps2 = d;
                }
            }
        }
        if eps2 == 0.0 {
            eps2 = f64::EPSILON;
        }
        if eps1 <= eps2 {
            return 0.0;
        }

        let log_eps2 = eps2.ln();
        let log_eps1 = eps1.ln();
        let epsilon: Vec<f64> = (0..bins)
            .map(|b| (log_eps2 + (log_eps1 - log_eps2) * b as f64 / (bins - 1) as f64).exp())
            .collect();

        let mut ci = vec![0.0f64; bins];
        for i in 0..pair_count {
            let sorted = &mut dists[row_start[i]..row_start[i + 1]];
            sorted.sort_by(f64::total_cmp);
            let mut prev_count = 0usize;
            for (b, &e) in epsilon.iter().enumerate() {
                let count = sorted.partition_point(|&d| d <= e);
                ci[b] += (count - prev_count) as f64;
                prev_count = count;
            }
        }

        // Cumulative sum -> correlation integral -> log-log curve.
        let denom = (sample_count - k) as f64 * (sample_count - k) as f64;
        let mut cum = 0.0;
        let mut curve = vec![f64::NAN; bins];
        for b in 0..bins {
            cum += ci[b];
            if cum > 0.0 {
                curve[b] = (cum / denom).ln() + ((sample_count - k - 1) as f64).ln();
            }
        }
        let log_eps: Vec<f64> = epsilon.iter().map(|e| e.ln()).collect();

        let finite: Vec<(f64, f64)> = log_eps
            .iter()
            .zip(curve.iter())
            .filter(|(_, v)| v.is_finite())
            .map(|(&e, &v)| (e, v))
            .collect();
        if finite.is_empty() {
            return 0.0;
        }

        let lo = finite.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min);
        let hi = finite.iter().map(|&(_, v)| v).fold(f64::NEG_INFINITY, f64::max);
        let mid = (lo + hi) / 2.0;
        let q = (hi - lo) / 4.0;

        let selected: Vec<(f64, f64)> = finite
            .into_iter()
            .filter(|&(_, v)| v > mid && v < mid + q)
            .collect();
        if selected.len() < 2 {
            return 0.0;
        }

        let xs: Vec<f64> = selected.iter().map(|&(e, _)| e).collect();
        let ys: Vec<f64> = selected.iter().map(|&(_, v)| v).collect();
        linear_regression_slope(&xs, &ys)
    }

    fn rosenstein_lye_ref(
        x: &[f64],
        fs: f64,
        tau: usize,
        dim: usize,
        slope: [f64; 4],
        mean_period: f64,
    ) -> (f64, f64) {
        let n = x.len();
        if dim == 0 || (dim - 1) * tau >= n {
            return (f64::NAN, f64::NAN);
        }
        let m = n - (dim - 1) * tau;
        if m < 2 {
            return (f64::NAN, f64::NAN);
        }

        let mut y = vec![0.0; m * dim];
        for j in 0..dim {
            for i in 0..m {
                y[i * dim + j] = x[j * tau + i];
            }
        }
        let point = |idx: usize| -> &[f64] { &y[idx * dim..idx * dim + dim] };

        let band = (dim - 1) * tau;

        // Nearest neighbour (excluding a temporal band around the diagonal) for
        // every embedded point.
        let mut ind2 = vec![0usize; m];
        for i in 0..m {
            let mut best_idx = usize::MAX;
            let mut best_d = f64::INFINITY;
            for k in 0..m {
                if k.abs_diff(i) <= band {
                    continue;
                }
                let d: f64 = point(i).iter().zip(point(k)).map(|(&a, &b)| (a - b).powi(2)).sum();
                if d < best_d {
                    best_d = d;
                    best_idx = k;
                }
            }
            ind2[i] = if best_idx == usize::MAX { i } else { best_idx };
        }

        // Average log divergence at each evolution step j: average, over all
        // starting points i whose trajectory and its neighbour's both survive j
        // steps, of ln(distance after j steps).
        let mut ave_ln_div = vec![0.0f64; m];
        for j in 0..m {
            let mut sum = 0.0;
            let mut count = 0usize;
            for i in 0..m {
                let nn = ind2[i];
                if i + j >= m || nn + j >= m {
                    continue;
                }
                let d: f64 = point(i + j).iter().zip(point(nn + j)).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
                if d > 0.0 {
                    sum += d.ln();
                    count += 1;
                }
            }
            ave_ln_div[j] = if count > 0 { sum / count as f64 } else { 0.0 };
            if count == 0 {
                break;
            }
        }

        let nz = ave_ln_div.iter().filter(|&&v| v != 0.0).count();
        let time: Vec<f64> = (0..m).map(|i| i as f64 / fs / mean_period).collect();

        let fit = |lo: usize, hi: usize| -> f64 {
            if hi >= ave_ln_div.len() || hi > nz || lo > hi {
                return f64::NAN;
            }
            linear_regression_slope(&time[lo..=hi], &ave_ln_div[lo..=hi])
        };

        let round_idx = |v: f64| -> usize { python_round(v * mean_period * fs).max(0.0) as usize };
        let s_lo = if slope[0] == 0.0 { 0 } else { round_idx(slope[0]) };
        let s_hi = round_idx(slope[1]);
        let l_lo = round_idx(slope[2]);
        let l_hi = round_idx(slope[3]);

        (fit(s_lo, s_hi), fit(l_lo, l_hi))
    }

    // `wolf_lye` and its neighbour search as they were before the search was
    // folded into one allocation-free sweep.
    #[allow(clippy::too_many_arguments)]
fn wolf_lye_ref(x: &[f64], fs: f64, tau: usize, dim: usize, evolve: usize) -> f64 {
        let n = x.len();
        if dim == 0 || (dim - 1) * tau + evolve >= n {
            return f64::NAN;
        }

        let min_v = x.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scale_mx = (max_v - min_v) / 10.0;
        let dt = 1.0 / fs;

        let m = n - (dim - 1) * tau;
        let mut y = vec![0.0; m * dim];
        for i in 0..dim {
            for j in 0..m {
                y[j * dim + i] = x[i * tau + j];
            }
        }

        let npt = n - (dim - 1) * tau - evolve;
        if npt == 0 {
            return f64::NAN;
        }
        let y_len = (npt + evolve).min(m);
        fn point(y: &[f64], dim: usize, idx: usize) -> &[f64] {
            &y[idx * dim..idx * dim + dim]
        }
        let dist = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b).map(|(&p, &q)| (p - q).powi(2)).sum::<f64>().sqrt()
        };

        // Initial neighbour pair: nearest point to index 0, excluding a small
        // temporal window around it.
        let mut current_pair = {
            let mut best_idx = usize::MAX;
            let mut best_dist = f64::INFINITY;
            for i in 0..npt.min(y_len) {
                if i <= 10 {
                    continue;
                }
                let d = dist(point(&y, dim, 0), point(&y, dim, i));
                if d > 0.0 && d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            if best_idx == usize::MAX {
                return f64::NAN;
            }
            best_idx
        };

        let mut its: u64 = 0;
        let mut dist_sum = 0.0;
        let mut lye = 0.0;
        let mut i = 0;
        while i < npt {
            let ep = i + evolve;
            if ep >= y_len {
                break;
            }
            let safe = current_pair + evolve < y_len;
            let pair_ep = if safe { current_pair + evolve } else { current_pair + evolve - 1 };
            if pair_ep >= y_len {
                break;
            }

            let start_dist = dist(point(&y, dim, i), point(&y, dim, current_pair));
            let end_dist = dist(point(&y, dim, ep), point(&y, dim, pair_ep));

            if start_dist > 0.0 && end_dist > 0.0 {
                dist_sum += (end_dist / start_dist).log2() / (evolve as f64 * dt);
                its += 1;
                lye = dist_sum / its as f64;
            }

            if end_dist < scale_mx {
                current_pair += evolve;
                if current_pair > npt {
                    current_pair -= evolve;
                    current_pair = wolf_next_point_ref(true, &y, dim, i, current_pair, npt, y_len, evolve, scale_mx);
                }
            } else {
                current_pair = wolf_next_point_ref(false, &y, dim, i, current_pair, npt, y_len, evolve, scale_mx);
            }

            i += evolve;
        }

        lye
    }

    fn wolf_next_point_ref(
        flag1: bool,
        y: &[f64],
        dim: usize,
        current_point: usize,
        current_point_pair: usize,
        npt: usize,
        y_len: usize,
        evolve: usize,
        scale_mx: f64,
    ) -> usize {
        const ANGLE_LIMIT_DEG: f64 = 30.0;

        let point = |idx: usize| -> &[f64] { &y[idx * dim..idx * dim + dim] };
        let ep = current_point + evolve;
        if ep >= y_len {
            return current_point_pair;
        }

        let safe = current_point_pair + evolve < y_len;
        let end_idx = if safe { current_point_pair + evolve } else { current_point_pair + evolve - 1 };
        if end_idx >= y_len {
            return current_point_pair;
        }

        let v_curr: Vec<f64> = point(ep).iter().zip(point(end_idx)).map(|(&a, &b)| a - b).collect();
        let end_dist: f64 = v_curr.iter().map(|v| v * v).sum::<f64>().sqrt();

        let bound = npt.min(y_len);
        let in_excl_window = |k: usize| k + 10 >= ep && k <= ep + 10;

        if !flag1 {
            let mut best: Option<(usize, f64)> = None;
            for k in 0..bound {
                if in_excl_window(k) {
                    continue;
                }
                let diff: Vec<f64> = point(ep).iter().zip(point(k)).map(|(&a, &b)| a - b).collect();
                let yd: f64 = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
                if yd <= 0.0 {
                    continue;
                }
                let dot: f64 = v_curr.iter().zip(diff.iter()).map(|(&a, &b)| a * b).sum();
                let denom = yd * end_dist;
                if denom <= 0.0 {
                    continue;
                }
                let cos_t = (dot / denom).abs().clamp(-1.0, 1.0);
                let theta = cos_t.acos();
                if theta < ANGLE_LIMIT_DEG.to_radians() && best.is_none_or(|(_, bd)| yd < bd) {
                    best = Some((k, yd));
                }
            }
            if let Some((idx, d)) = best {
                if d <= scale_mx {
                    return idx;
                }
            }
            // Falls through to the angle-agnostic nearest neighbour below,
            // exactly as the Python `next_pt == -1` fallback does.
        }

        let mut best_idx = usize::MAX;
        let mut best_d = f64::INFINITY;
        for k in 0..bound {
            if in_excl_window(k) {
                continue;
            }
            let diff: Vec<f64> = point(ep).iter().zip(point(k)).map(|(&a, &b)| a - b).collect();
            let yd: f64 = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
            if yd > 0.0 && yd < best_d {
                best_d = yd;
                best_idx = k;
            }
        }

        if best_idx == usize::MAX { current_point_pair } else { best_idx }
    }

    /// The original fixed-window Rosenstein estimate, rebuilt on top of
    /// `divergence_curve`. Matching `rosenstein_lye_ref` bit for bit proves the
    /// curve the rise fit reads is the one the original implementation fitted.
    pub(super) fn fixed_window_lye(
        x: &[f64],
        fs: f64,
        tau: usize,
        dim: usize,
        slope: [f64; 4],
        mean_period: f64,
    ) -> (f64, f64) {
        let n = x.len();
        if dim == 0 || (dim - 1) * tau >= n {
            return (f64::NAN, f64::NAN);
        }
        let m = n - (dim - 1) * tau;
        if m < 2 {
            return (f64::NAN, f64::NAN);
        }
        // The original left the steps after the curve's end at zero and
        // counted non-zeros, so its fit could reach one zero past the end.
        let curve = divergence_curve(x, tau, dim, (dim - 1) * tau, m - 1);
        let mut ave_ln_div = vec![0.0f64; m];
        ave_ln_div[..curve.len()].copy_from_slice(&curve);
        let nz = ave_ln_div.iter().filter(|&&v| v != 0.0).count();
        let time: Vec<f64> = (0..m).map(|i| i as f64 / fs / mean_period).collect();

        let fit = |lo: usize, hi: usize| -> f64 {
            if hi >= ave_ln_div.len() || hi > nz || lo > hi {
                return f64::NAN;
            }
            linear_regression_slope(&time[lo..=hi], &ave_ln_div[lo..=hi])
        };
        let round_idx = |v: f64| -> usize { python_round(v * mean_period * fs).max(0.0) as usize };
        let s_lo = if slope[0] == 0.0 { 0 } else { round_idx(slope[0]) };
        (fit(s_lo, round_idx(slope[1])), fit(round_idx(slope[2]), round_idx(slope[3])))
    }

    /// Bit-identical, treating any two NaNs as equal (the sign and payload of a
    /// NaN produced by a division is not part of the contract here).
    fn same(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    fn zscore(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        if sd > 0.0 {
            for x in v.iter_mut() {
                *x = (*x - mean) / sd;
            }
        }
        v
    }

    fn lcg(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect()
    }

    /// The inputs the running extraction actually produces, plus the degenerate
    /// shapes that exercise the guards the fast paths reorganise.
    fn cases() -> Vec<(&'static str, Vec<f64>)> {
        let mut out: Vec<(&'static str, Vec<f64>)> = Vec::new();
        out.push(("random_walk", zscore(lcg(12345, 1000))));
        out.push(("sine_mix", zscore((0..1000).map(|i| (i as f64 * 0.13).sin() + 0.5 * (i as f64 * 0.031).sin()).collect())));

        // A flat interpolated segment -- the gap-fill case from the handoff's
        // section 4.2, which is genuinely low-dimensional and is present in the
        // real corpus. It drives distances toward zero, exercising the
        // `eps2 == 0.0` and `d > 0.0` branches.
        let mut gappy = lcg(999, 1000);
        for i in 300..520 {
            gappy[i] = 0.0;
        }
        out.push(("interpolated_gap", zscore(gappy)));

        // Constant series: every pair distance is exactly zero.
        out.push(("constant", vec![0.0; 1000]));

        // Short series: trips `k + 1 >= sample_count` in correlation_dimension
        // and the `m < 2` / narrow-band guards in rosenstein_lye.
        out.push(("short_30", zscore(lcg(7, 30))));
        out.push(("short_26", zscore(lcg(8, 26))));
        out.push(("tiny_6", zscore(lcg(9, 6))));
        out
    }

    // The parameters `config_chaos_5hz.json` (and the compiled defaults) use,
    // which is what the corpus being written right now was extracted with.
    const TAU: usize = 5;
    const DIM: usize = 5;
    const FS: f64 = 5.0;
    const SLOPE: [f64; 4] = [0.5, 5.0, 5.0, 20.0];
    const MEAN_PERIOD: f64 = 1.0;

    #[test]
    fn correlation_dimension_matches_reference_bitwise() {
        for (name, x) in cases() {
            let got = correlation_dimension(&x, TAU, DIM);
            let want = correlation_dimension_ref(&x, TAU, DIM);
            assert!(same(got, want), "{name}: {got:?} != {want:?}");
        }
    }

    #[test]
    fn rosenstein_matches_reference_bitwise() {
        for (name, x) in cases() {
            let got = fixed_window_lye(&x, FS, TAU, DIM, SLOPE, MEAN_PERIOD);
            let want = rosenstein_lye_ref(&x, FS, TAU, DIM, SLOPE, MEAN_PERIOD);
            assert!(same(got.0, want.0) && same(got.1, want.1), "{name}: {got:?} != {want:?}");
        }
    }

    /// The fit indices scale with `fs` and `mean_period`, so rates that move
    /// them past the 5 Hz defaults must still agree.
    #[test]
    fn rosenstein_truncation_holds_across_rates() {
        let x = zscore(lcg(2024, 1000));
        for &(fs, mp) in &[(5.0, 1.0), (10.0, 1.0), (20.0, 1.0), (5.0, 0.5), (5.0, 2.0), (100.0, 1.0)] {
            let got = fixed_window_lye(&x, fs, TAU, DIM, SLOPE, mp);
            let want = rosenstein_lye_ref(&x, fs, TAU, DIM, SLOPE, mp);
            assert!(same(got.0, want.0) && same(got.1, want.1), "fs={fs} mp={mp}: {got:?} != {want:?}");
        }
    }

    /// Sweeps embedding parameters away from the defaults, since `tau`/`dim`
    /// set the exclusion band and the triangular pair count both rewrites index.
    #[test]
    fn parity_across_embedding_parameters() {
        let x = zscore(lcg(555, 600));
        for tau in 1..=6 {
            for dim in 2..=6 {
                let c = correlation_dimension(&x, tau, dim);
                let cr = correlation_dimension_ref(&x, tau, dim);
                assert!(same(c, cr), "corrdim tau={tau} dim={dim}: {c:?} != {cr:?}");
                let r = fixed_window_lye(&x, FS, tau, dim, SLOPE, MEAN_PERIOD);
                let rr = rosenstein_lye_ref(&x, FS, tau, dim, SLOPE, MEAN_PERIOD);
                assert!(same(r.0, rr.0) && same(r.1, rr.1), "ros tau={tau} dim={dim}: {r:?} != {rr:?}");
            }
        }
    }

    #[test]
    fn wolf_matches_reference_bitwise() {
        for (name, x) in cases() {
            for &(tau, dim) in &[(TAU, DIM), (2, 4), (1, 4), (3, 3)] {
                let got = wolf_lye(&x, FS, tau, dim, 5);
                let want = wolf_lye_ref(&x, FS, tau, dim, 5);
                assert!(same(got, want), "{name} tau={tau} dim={dim}: {got:?} != {want:?}");
            }
        }
        for seed in 0..40u64 {
            let x = zscore(lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(1), 400));
            assert!(same(wolf_lye(&x, FS, 2, 4, 5), wolf_lye_ref(&x, FS, 2, 4, 5)), "seed={seed}");
        }
    }

    /// Many independent random series, to catch a boundary the fixed cases miss.
    #[test]
    fn parity_over_many_seeds() {
        for seed in 0..40u64 {
            let x = zscore(lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(1), 400));
            let c = correlation_dimension(&x, TAU, DIM);
            let cr = correlation_dimension_ref(&x, TAU, DIM);
            assert!(same(c, cr), "corrdim seed={seed}: {c:?} != {cr:?}");
            let r = fixed_window_lye(&x, FS, TAU, DIM, SLOPE, MEAN_PERIOD);
            let rr = rosenstein_lye_ref(&x, FS, TAU, DIM, SLOPE, MEAN_PERIOD);
            assert!(same(r.0, rr.0) && same(r.1, rr.1), "ros seed={seed}: {r:?} != {rr:?}");
        }
    }
}
