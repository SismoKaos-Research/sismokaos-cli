"""Raw threshold figures: every feature's values with its percentile bounds.

One figure per station, era and feature column. The panel holds the raw
per-window values, the lower and upper bound for every q as labelled lines,
and the windows crossing the ``--mark-q`` bounds: filled red when an event
lies in the window (same matching as ``sismokaos-thresholds``), hollow when
none does. The lane above holds the events that count, and the title the
crossing count, precision, lift and recall at ``--mark-q``.

Bounds, events and matching are those of ``sismokaos-thresholds``, so the
numbers here equal its CSVs. Two sets are drawn, one per analysis:

    <out>/all/<STA>/<era>/<column>.png       whole eras, bounds from them
    <out>/pre_main/<STA>/<era>/<column>.png  only data before each era's
                                             main shock, bounds from that

A ``bands`` run gets one more level, ``<STA>/<band>/<era>/``, and each band
its own window length from the run metadata.

Example:
    sismokaos-threshold-raw output/van --out figs/van/thresholds/raw
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .catalog import events_near, load_catalog, load_station_coords  # noqa: E402
from .cli import DEFAULT_CATALOG, DEFAULT_STATIONS, find_parquets  # noqa: E402
from .features import (break_gaps, feature_columns, group_runs,  # noqa: E402
                       load_station, masked, split_eras, win_sec_of)
from .render import Style, pick_main, robust_limits  # noqa: E402
from .threshold_figs import (EVENT, GRID, INK, INK_2, SURFACE,  # noqa: E402
                             _style)
from .thresholds import detectable, finish, match_windows, score  # noqa: E402

RAW = "#9ec5f4"
MISS = "#52514e"
# One gray per q, lighter for looser bounds, so the tightest reads darkest.
BOUND_INKS = ["#c3c2b7", "#9a9993", "#6f6e69", "#3d3c39"]
MODES = {"all": "whole era", "pre_main": "before main shock"}


def fig_raw(era, column, events, main, qs, mark_q, reach, max_interp,
            station, mode, path, dpi):
    """Draws one figure; returns the scores at ``mark_q`` or None if skipped."""
    series = masked(era, column, max_interp)
    vals = series.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() < 100:
        return None
    finite = vals[valid]
    bounds = {q: np.percentile(finite, [q, 100 - q]) for q in qs}

    ends = era.index.to_numpy()
    det = detectable(events)
    origins = np.sort(det["time"].to_numpy())
    holds, first, last = match_windows(ends, origins, reach)
    lo, hi = bounds[mark_q]
    cross = ((vals < lo) | (vals > hi)) & valid
    result = finish(score(cross, valid, holds, first, last))
    sides = {name: finish(score(c & valid, valid, holds, first, last))
             for name, c in (("lower", vals < lo), ("upper", vals > hi))}

    fig, (lane, ax) = plt.subplots(
        2, 1, figsize=(12, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.06})
    t0, t1 = era.index[0], era.index[-1]

    shown = det[det["time"] >= t0]
    lane.vlines(shown["time"], 0, shown["mag"], color=EVENT, lw=0.6, alpha=0.6)
    lane.scatter(shown["time"], shown["mag"], s=6, color=EVENT, lw=0, zorder=3)
    lane.set_ylim(0, max(4.0, float(shown["mag"].max()) + 0.5) if len(shown) else 4.0)
    lane.set_ylabel("Mag")
    lane.grid(True, axis="y", color=GRID, lw=0.6)

    t, v = break_gaps(series, pd.Timedelta(np.median(np.diff(ends))) * 2.5)
    ax.plot(t, v, color=RAW, lw=0.4, rasterized=True, zorder=1)

    inks = dict(zip(sorted(qs, reverse=True), BOUND_INKS[-len(qs):]))
    for q, ink in inks.items():
        for b in bounds[q]:
            ax.axhline(b, color=ink, lw=1.1 if q == mark_q else 0.8,
                       ls="-" if q == mark_q else "--", zorder=2)

    hit, miss = cross & holds, cross & ~holds
    ax.scatter(series.index[miss], vals[miss], s=9, facecolor=SURFACE,
               edgecolor=MISS, lw=0.7, zorder=3, rasterized=True,
               label=f"crossing, no catalog event ({miss.sum()})")
    ax.scatter(series.index[hit], vals[hit], s=9, color=EVENT, lw=0,
               zorder=4, rasterized=True,
               label=f"crossing with an event ({hit.sum()})")

    if main is not None and t0 <= main["time"] <= t1:
        for a in (lane, ax):
            a.axvline(main["time"], color=EVENT, lw=1.4, zorder=5)
        lane.annotate(f"M{main['mag']:.1f}", (main["time"], 1.0),
                      xycoords=("data", "axes fraction"), xytext=(-4, -2),
                      textcoords="offset points", ha="right", va="top",
                      fontsize=8, color=EVENT)

    # Limits wide enough for the outermost bounds, on a log axis when the
    # values span decades (amplitude features).
    limits = robust_limits(vals, (0.01, 99.99))
    if limits is not None:
        ylo, yhi, log = limits
        outer = [b for q in qs for b in bounds[q]]
        ylo, yhi = min(ylo, min(outer)), max(yhi, max(outer))
        if log:
            ax.set_yscale("log")
        else:
            pad = 0.03 * (yhi - ylo)
            ylo, yhi = ylo - pad, yhi + pad
        ax.set_ylim(ylo, yhi)

    _label_bounds(ax, bounds, inks, mark_q)
    ax.set_ylabel(column)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for q, ink in inks.items():
        ax.plot([], [], color=ink, lw=1.1 if q == mark_q else 0.8,
                ls="-" if q == mark_q else "--", label=f"bounds at q = {q:g}%")
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6,
              ncol=2)
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_xlim(t0, t1)

    lane.set_title(f"{station}  |  {column}  |  {t0:%Y-%m-%d} to {t1:%Y-%m-%d}  "
                   f"({MODES[mode]})", fontsize=11, loc="left", color=INK)
    fig.text(0.01, 0.012,
             f"Bounds at q = {', '.join(f'{q:g}' for q in qs)}% per tail; "
             f"crossings marked at q = {mark_q:g}%: {result['crossings']} windows, "
             f"precision {result['precision']:.1%}, lift {result['lift']:.1f}, "
             f"recall {result['recall']:.0%} of {result['events']} events "
             f"(lower: precision {sides['lower']['precision']:.1%}, lift "
             f"{sides['lower']['lift']:.1f}; upper: precision "
             f"{sides['upper']['precision']:.1%}, lift {sides['upper']['lift']:.1f}).\n"
             "Events: any M within 100 km, M>=3 within 300 km (AFAD); a window "
             "holds one when an origin lies in it or up to 90 s before.",
             fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.94, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return result


def _label_bounds(ax, bounds, inks, mark_q, min_gap=0.03):
    """Labels each bound at the right edge, beside its own line. Where bounds
    bunch up (the low tail of a positive feature) the labels that would
    collide are left off; the legend still names every line."""
    to_axes = ax.transAxes.inverted()
    marks = []
    for q, (lo, hi) in bounds.items():
        for b in (lo, hi):
            y = to_axes.transform(ax.transData.transform((0, b)))[1]
            if 0 <= y <= 1:
                # The marked q is placed first so it is never the one dropped.
                marks.append((q != mark_q, y, q))
    placed = []
    for _, y, q in sorted(marks):
        if all(abs(y - other) >= min_gap for other in placed):
            placed.append(y)
            ax.text(1.004, y, f"{q:g}%", transform=ax.transAxes, va="center",
                    fontsize=7, color=INK if q == mark_q else inks[q])


def station_job(station, band, parquets, mode, opts):
    """All figures for one station, band and mode; runs in a worker process."""
    _style()
    catalog = load_catalog(opts["catalog"])
    coords = load_station_coords(opts["stations"])
    if station not in coords:
        return station, mode, 0, [f"[WARN] no coordinates for {station}; skipped"]
    df = load_station(parquets)
    events = events_near(catalog, coords[station], 300.0)
    win_sec = opts["win_sec"] or win_sec_of(parquets) or 200.0
    reach = np.timedelta64(int((win_sec + opts["travel_sec"]) * 1e9), "ns")
    name = f"{station} {band}" if band else station
    base = opts["out"] / mode / station / band if band else opts["out"] / mode / station
    made, notes = 0, []
    for era in split_eras(df, pd.Timedelta(days=opts["era_gap_days"])):
        t0, t1 = era.index[0], era.index[-1]
        ev = events[(events["time"] >= t0 - pd.Timedelta(reach))
                    & (events["time"] <= t1)]
        main = pick_main(ev, Style())
        if mode == "pre_main" and main is not None:
            era = era[era.index < main["time"]]
            ev = ev[ev["time"] < main["time"]]
        if era.empty:
            continue
        label = f"{era.index[0]:%Y-%m-%d}_{era.index[-1]:%Y-%m-%d}"
        folder = base / label
        for column in feature_columns(era.columns, dev=False):
            r = fig_raw(era, column, ev, main, opts["q"], opts["mark_q"], reach,
                        opts["max_interp"], name, mode, folder / f"{column}.png",
                        opts["dpi"])
            made += r is not None
        notes.append(f"{mode:8s} {name} {label}")
    return name, mode, made, notes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="sismokaos-threshold-raw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("thresholds_raw"))
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    ap.add_argument("--station", action="append", help="only these stations")
    ap.add_argument("--band", action="append", help="only these bands, e.g. 1-5Hz")
    ap.add_argument("--mode", choices=list(MODES), action="append",
                    help="which analysis to draw (default: both)")
    ap.add_argument("--q", type=float, nargs="+", default=[5, 1, 0.5, 0.1])
    ap.add_argument("--mark-q", type=float, default=1.0,
                    help="q whose crossings are marked (1)")
    ap.add_argument("--win-sec", type=float, default=None,
                    help="window length in s; default: each run's metadata, else 200")
    ap.add_argument("--travel-sec", type=float, default=90.0)
    ap.add_argument("--max-interp", type=float, default=0.5)
    ap.add_argument("--era-gap-days", type=float, default=30.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mark_q not in args.q:
        raise SystemExit(f"[ERROR] --mark-q {args.mark_q:g} is not among --q")
    runs = group_runs(find_parquets(args.inputs))
    if args.station:
        runs = {k: ps for k, ps in runs.items() if k[0] in args.station}
    if args.band:
        runs = {k: ps for k, ps in runs.items() if k[1] in args.band}
    opts = {k: getattr(args, k) for k in ("catalog", "stations", "q", "mark_q",
                                          "win_sec", "travel_sec", "max_interp",
                                          "era_gap_days", "dpi", "out")}
    jobs = [(s, b, ps, m, opts) for (s, b), ps in runs.items()
            for m in (args.mode or list(MODES))]
    total = 0
    # spawn, not fork: pyarrow has threads running by now, and forking a
    # threaded process can deadlock the child.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs))),
                             mp_context=ctx) as pool:
        futures = [pool.submit(station_job, *j) for j in jobs]
        for f in as_completed(futures):
            station, mode, made, notes = f.result()
            total += made
            for n in notes:
                print(f"  {n}")
            print(f"{station} {mode}: {made} figures")
    print(f"{total} figures -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
