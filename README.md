# Sismokaos CLI

Sismokaos CLI is a command-line interface tool for preprocessing and feature extraction of seismic data from miniSEED files.

## Features

- Parse and process miniSEED files from a local directory. `.zip` files in it
  are read too: their members are inflated to `<out-dir>/.unzipped-<pid>/` for
  the run and deleted afterwards, so expect that much free disk.
- Stream-safe preprocessing pipeline (gap fill, detrend, bandpass, decimation).
- Chaotic features (Wolf/Rosenstein Lyapunov exponents, Sample Entropy,
  Correlation Dimension) with parameters calibrated once per run on
  seismically quiet periods (see **Chaotic features** below).
- Several frequency bands per run, each with its own rate, windows and
  calibration (see **Frequency bands**).
- Full feature extraction pipeline (`run`) producing per-window engineered features.
- Preprocess-only pipeline (`preprocess`) producing filtered/decimated component streams without feature extraction.
- Configurable processing parameters via JSON config and CLI overrides.
- Output results with metadata (see **Output formats** below).

## Output formats

The two pipelines write different things, because they are consumed
differently.

### `preprocess` → flat `f32` + JSON sidecar

`<stem>_preprocessed.f32` is little-endian `float32`, interleaved
`E N Z E N Z …`, alongside `<stem>_preprocessed.f32.json`.

```python
import json, numpy as np
meta = json.load(open("out_preprocessed.f32.json"))
a = np.memmap("out_preprocessed.f32", dtype="<f4", mode="r").reshape(-1, 3)
```

This used to be zstd Parquet with five `f64` columns. It was replaced because
the Parquet was never read as Parquet — the model side loaded it once and
rewrote it as exactly this layout — and because it was *larger*: on the
718,848,001-sample 10 Hz archive, 16.41 GB of Parquet against 8.63 GB here.
Two of its five columns (`index`, `Zaman_Dk`) were pure functions of the row
number, 11.9% of the bytes, and the other three were stored at double the
precision the consumer keeps.

**Timing is piecewise.** The pipeline restarts its clock at every data gap, so
the sidecar carries a `segments` table of `(sample_offset, start_epoch)` rather
than one start time. Sample `i` sits at `start_epoch + (i - sample_offset) / fs`
for the last segment beginning at or before it. A single global start time
would place every sample after the first gap at the wrong moment.

Unfillable gaps stay `NaN`, which `f32` represents exactly; whether to
zero-fill is left to the consumer.

### `run` → Parquet

The feature table is small, heterogeneous and read whole, which is what Parquet
is for. It keeps `<name>` and `<name>_DEV` columns; the `_DEV` columns are
first differences, and the models consume them as features.

## Chaotic features

### Parameters are calibrated on quiet periods

The embedding delay `tau`, embedding dimension `dim`, Rosenstein mean period
and Sample Entropy tolerance are estimated once, before any window is
measured, and held fixed for the whole run. Estimating them per window would
let them change across an earthquake along with the features they set, which
makes windows incomparable.

1. Every event in the catalog within `max_dist_km` of the station, at or
   above `min_mag`, excludes `pre_sec` before its origin time to a
   magnitude-dependent coda after it. Gap-free windows touching no exclusion
   are quiet.
2. A first pass over the data reservoir-samples `max_windows` quiet windows
   from the whole run (seeded), and drops any louder than `loud_factor` times
   its component's median, since the catalog misses teleseisms and glitches.
3. Per window and component: `tau` is the first minimum of the average mutual
   information, the mean period is the reciprocal of the power-weighted mean
   frequency, and `dim` comes from False Nearest Neighbours at that `tau`.
   The run uses the medians. SampEn's tolerance becomes absolute:
   `chaos_sampent_r` times each component's median quiet std, so a louder
   window is not rescaled to look like a quiet one.

The result goes to `<stem>_calibration.json` next to the features, and the
values into `run_metadata.json`. `--calibration FILE` reuses one, for example
to measure another station with the same parameters. The catalog settings are
in the `chaos_calibration` block of the config (defaults: M>=0, 300 km, 300
windows); `--catalog` and `--station-coords` override the files.

The values `chaos_tau`, `chaos_dim` and `chaos_mean_period` in a config file
are overwritten by the calibration.

### Rosenstein fit range per window

Each window's mean log-divergence curve is followed for `chaos_ros_horizon`
mean periods (default 20). `ROS_SHORT` is the slope between the first steps
reaching 10% and 90% of the curve's rise from its first step to its peak
(`chaos_ros_rise`); `ROS_LONG` is the slope from that 90% step to the end,
over the saturation. Neighbours closer in time than one mean period (or the
embedding span, if longer) are excluded. Both are per mean period.

### Chaos only

`run --chaos-only` computes just the five chaotic features per component and
writes `<stem>_chaos_features.parquet`, so it never overwrites a full run in
the same directory. The plotter picks it up like any `*_features.parquet`;
keep chaos-only and full runs in separate directories, or a station's windows
are loaded twice.

## Frequency bands

A `bands` list in the config runs the pipeline once per band, each with its
own filter, output rate, windows, STA/LTA lengths and calibration, into
`<out-dir>/<band name>/`. `config_bands.json` defines four:

| Band | fs | Window / step | Dominated by |
|---|---|---|---|
| `0.02-0.1Hz` | 0.5 Hz | 2000 s / 500 s | primary microseism, tilt, pressure |
| `0.1-1Hz` | 2.5 Hz | 400 s / 100 s | secondary (ocean) microseism: the control band |
| `1-5Hz` | 25 Hz | 40 s / 50 s | local microseismicity, tremor, cultural noise |
| `5-20Hz` | 50 Hz | 20 s / 50 s | near-field microcracking, cultural noise |

Every window holds 1000 samples, and every rate is 2.5-5x the band's upper
edge, so the estimators see similar inputs in every band. Every rate divides
both 100 Hz (HH) and 50 Hz (BH), so the same list serves either kind of
archive. Steps keep each band near the default run's window count; the 1-5 and
5-20 Hz bands sample 40 s and 20 s of every 50 s.

```bash
sismokaos-cli --config config_bands.json run --data-dir ./data --out-dir ./output/elba
sismokaos-cli --config config_bands.json run --data-dir ./data --out-dir ./output/elba \
    --band 1-5Hz --band 5-20Hz --chaos-only
```

A band's `name` defaults to `<freqmin>-<freqmax>Hz`, and its
`spectral_split_hz` (the LOW/HIGH_FREQ_ENERGY boundary, 1 Hz for single-band
runs) to the band's geometric centre. `--fs`, `--freqmin`, `--freqmax` and
`--win-sec` are rejected alongside a band list, and every band is validated
before any work starts. A band that fails does not stop the others; the exit
code is 1 if any did.

## Filtering

The bandpass is an order-4 Butterworth (order 8 as a bandpass) applied forward
and backward, built as second-order sections. The `butterworth` crate it
replaced built one transfer-function polynomial, which lost accuracy at low
corners: at 100 Hz a 0.02-0.1 Hz band went unstable, 0.1-1 Hz came out 4% hot
at band centre, and 0.1-2 Hz 0.2%. Data are filtered in hour-long blocks, each
with `max(120 s, 5 / freqmin)` of real data on both sides that is filtered and
then dropped; before, the end of every block had no future context and was
distorted for about the filter's ringing time. Both changes alter output
slightly against runs made before them.

## Guards

Configurations whose damage would be invisible in the output are rejected
before any work starts:

- **`freqmax` at or above the output Nyquist** is a hard error (exit 2). The
  bandpass doubles as the anti-alias filter — decimation follows it with no
  separate low-pass — so a too-high `freqmax` folds everything above Nyquist
  back into the band with no way to detect it afterwards. Landing within 10%
  of Nyquist warns instead.
- **A requested `fs` that does not divide the native rate** warns: the
  decimation factor is an integer, so asking for 30 Hz of a 100 Hz archive
  yields 33.3 Hz, and window lengths derived from the requested rate would not
  match the data.

## Installation

Ensure you have Rust installed, then clone this repository and run:

```bash
cargo build --release
```

The binary will be located at `target/release/sismokaos-cli`.

## Usage

Initialize a default config:

```bash
sismokaos-cli init --out config.json
```

Run the full pipeline:

```bash
sismokaos-cli --config config.json run --data-dir ./data --out-dir ./output
```

Run preprocessing only (no feature extraction):

```bash
sismokaos-cli --config config.json preprocess --data-dir ./data --out-dir ./output
```

Dry-run (validate config/args only):

```bash
sismokaos-cli --config config.json preprocess --data-dir ./data --out-dir ./output --dry-run
```

## Plotting features

`plot/` is a small Python package that draws every feature of one or more
`run` outputs against the AFAD catalog: per station, per era (split at data
gaps over 30 days), one page per feature group plus one high-resolution
figure per feature and component, each with an event lane on top.

```bash
uv run --project plot sismokaos-plot output/van --out figs/van
uv run --project plot sismokaos-plot output/van --station TVAN \
    --around 2011-10-23T10:41:20 --days 7
uv run --project plot sismokaos-plot output/van --before-main 30 60 --no-eras
```

`--before-main` draws each era's lead-up to its main shock (the largest
M>=4.5 within 100 km).

A run with a `bands` config is drawn one band at a time, into
`<out>/<STATION>/<band>/`, with the band in every title; a folder input is
searched one level down, so the run's `--out-dir` is enough. `--band` picks
bands.

```bash
uv run --project plot sismokaos-plot output/focus_chaos --out figs/focus --before-main 30 60
```

### Threshold crossings against the catalog

`sismokaos-thresholds` puts lower and upper bounds at percentiles of each
feature and scores how often windows crossing them hold a catalog event:
precision, recall, and lift over chance. `--exclude-after-main` keeps the
aftershock sequence from dominating. `sismokaos-threshold-figs` draws the
summary figures from its CSVs, and `sismokaos-threshold-raw` draws every
feature's values with its bounds and crossings.

All three keep bands apart: each band is scored with its own window length
(from the run metadata; `--win-sec` overrides it), pooled across stations
only within a band, and drawn as `<figure>_<band>.png` or under
`<STA>/<band>/`. A crossing counts when an event lies in the window, so lift
measures detection of events, not anticipation of them.

```bash
uv run --project plot sismokaos-thresholds output/van --out figs/van/thresholds/all.csv
uv run --project plot sismokaos-thresholds output/van --exclude-after-main \
    --out figs/van/thresholds/pre_main.csv
uv run --project plot sismokaos-threshold-figs figs/van/thresholds --data output/van
uv run --project plot sismokaos-threshold-raw output/van --out figs/van/thresholds/raw
```

The catalog and station list default to the copies in `catalogs/`; see
`sismokaos-plot --help` for the event radius, line thresholds and output
options.

## Catalog

`catalogs/` holds the AFAD event catalog (gzipped; the CLI and the plotter
both read it compressed) and AFAD station list, copied from
`data_downloader/catalogs/` so the project runs on its own. To refresh, export
a new catalog there and copy it in under a new dated name:

```bash
gzip -9 -c ../data_downloader/catalogs/catalog_afad_full_<date>.csv \
    > catalogs/catalog_afad_<date>.csv.gz
```

then point the defaults at it (`CalibrationConfig::default` in
`src/config.rs`, `DEFAULT_CATALOG` in `plot/src/sismokaos_plot/cli.py`).
