"""How often a feature crossing a percentile bound coincides with an event.

For each station and era, each feature gets a lower bound at its q-th
percentile and an upper bound at its (100-q)-th. A window *crosses* when its
value falls outside a bound, and *matches* when a catalog event's origin lies
in the window or up to ``--travel-sec`` before its start, so the arrivals of
an event fall inside it.

Reported per feature, side (lower/upper/either) and q:
  precision  crossing windows that match an event
  base       all windows that match an event: precision of random picks
  lift       precision / base; 1 is chance
  recall     events with at least one crossing window

Lift is the number to read: during an aftershock sequence most windows hold
an event, so precision alone is high for anything.

Examples:
    sismokaos-thresholds output/van --out figs/van/thresholds.csv
    sismokaos-thresholds output/van --q 2 1 --exclude-after-main
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import events_near, load_catalog, load_station_coords
from .cli import DEFAULT_CATALOG, DEFAULT_STATIONS, find_parquets
from .features import feature_columns, load_station, masked, split_eras, station_of
from .render import Style, pick_main

# (upper distance km, min magnitude): events that can show on the trace,
# matching plot_rosenstein.py's per-distance-class floors.
DETECTABLE = [(100.0, 0.0), (300.0, 3.0)]
SIDES = ("lower", "upper", "either")


def detectable(events: pd.DataFrame) -> pd.DataFrame:
    keep = np.zeros(len(events), dtype=bool)
    lower = 0.0
    for upper, min_mag in DETECTABLE:
        keep |= ((events["dist_km"] >= lower) & (events["dist_km"] < upper)
                 & (events["mag"] >= min_mag)).to_numpy()
        lower = upper
    return events[keep]


def match_windows(ends: np.ndarray, origins: np.ndarray, reach: np.timedelta64):
    """(per-window: holds an event, per-event: [first, last) window index).

    A window ending at ``e`` holds an event with origin in ``[e - reach, e]``.
    """
    lo = np.searchsorted(origins, ends - reach, side="left")
    hi = np.searchsorted(origins, ends, side="right")
    holds = hi > lo
    first = np.searchsorted(ends, origins, side="left")
    last = np.searchsorted(ends, origins + reach, side="right")
    return holds, first, last


def score(cross: np.ndarray, valid: np.ndarray, holds: np.ndarray,
          first: np.ndarray, last: np.ndarray) -> dict:
    """Counts for one feature, side and q (summable across station-eras)."""
    cross = cross & valid
    # Prefix sums answer "any crossing window in [first, last)" per event.
    csum = np.concatenate([[0], np.cumsum(cross)])
    return {
        "windows": int(valid.sum()),
        "event_windows": int((holds & valid).sum()),
        "crossings": int(cross.sum()),
        "hits": int((cross & holds).sum()),
        "events": int(len(first)),
        "events_found": int(((csum[last] - csum[first]) > 0).sum()),
    }


def finish(counts: dict) -> dict:
    c = dict(counts)
    c["base"] = c["event_windows"] / c["windows"] if c["windows"] else np.nan
    c["precision"] = c["hits"] / c["crossings"] if c["crossings"] else np.nan
    c["lift"] = c["precision"] / c["base"] if c["base"] else np.nan
    c["recall"] = c["events_found"] / c["events"] if c["events"] else np.nan
    return c


def analyse(df, events, qs, max_interp, reach):
    """Rows of raw counts for one station-era."""
    ends = df.index.to_numpy()
    ev = detectable(events)
    origins = np.sort(ev["time"].to_numpy())
    holds, first, last = match_windows(ends, origins, reach)
    rows = []
    for column in feature_columns(df.columns, dev=False):
        values = masked(df, column, max_interp).to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 100:
            continue
        for q in qs:
            lo, hi = np.percentile(values[valid], [q, 100 - q])
            below, above = values < lo, values > hi
            for side, cross in zip(SIDES, (below, above, below | above)):
                rows.append({"feature": column, "side": side, "q": q,
                             **score(cross, valid, holds, first, last)})
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="sismokaos-thresholds", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("thresholds.csv"),
                    help="CSV of every station-era, feature, side and q")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    ap.add_argument("--q", type=float, nargs="+", default=[5, 1, 0.5, 0.1],
                    help="tail percentages to try (5 1 0.5 0.1)")
    ap.add_argument("--win-sec", type=float, default=200.0)
    ap.add_argument("--travel-sec", type=float, default=90.0,
                    help="S travel time allowance; 300 km at 3.5 km/s (90)")
    ap.add_argument("--max-interp", type=float, default=0.5)
    ap.add_argument("--era-gap-days", type=float, default=30.0)
    ap.add_argument("--exclude-after-main", action="store_true",
                    help="drop windows after each era's main shock, so the "
                         "aftershock sequence does not dominate")
    ap.add_argument("--top", type=int, default=10)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    catalog = load_catalog(args.catalog)
    coords = load_station_coords(args.stations)
    reach = np.timedelta64(int((args.win_sec + args.travel_sec) * 1e9), "ns")

    by_station = defaultdict(list)
    for p in find_parquets(args.inputs):
        by_station[station_of(p)].append(p)

    rows = []
    for station, parquets in sorted(by_station.items()):
        if station not in coords:
            print(f"[WARN] no coordinates for {station}; skipped")
            continue
        df = load_station(parquets)
        events = events_near(catalog, coords[station], 300.0)
        for era in split_eras(df, pd.Timedelta(days=args.era_gap_days)):
            t0, t1 = era.index[0], era.index[-1]
            ev = events[(events["time"] >= t0 - pd.Timedelta(reach))
                        & (events["time"] <= t1)]
            if args.exclude_after_main:
                main_ev = pick_main(ev, Style())
                if main_ev is not None:
                    era = era[era.index < main_ev["time"]]
                    ev = ev[ev["time"] < main_ev["time"]]
            label = f"{station} {t0:%Y-%m-%d}..{era.index[-1]:%Y-%m-%d}"
            n_ev = len(detectable(ev))
            print(f"{label}: {len(era)} windows, {n_ev} detectable events")
            for r in analyse(era, ev, args.q, args.max_interp, reach):
                rows.append({"station": station, "era": f"{t0:%Y}", **r})

    raw = pd.DataFrame(rows)
    table = pd.DataFrame([finish(r) for r in raw.to_dict("records")])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nPer station-era results -> {args.out}")

    # Pooled over every station-era: sums of counts, then the ratios.
    keys = ["feature", "side", "q"]
    count_cols = ["windows", "event_windows", "crossings", "hits", "events",
                  "events_found"]
    pooled = raw.groupby(keys, as_index=False)[count_cols].sum()
    pooled = pd.DataFrame([finish(r) for r in pooled.to_dict("records")])
    pooled_path = args.out.with_name(args.out.stem + "_pooled.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"Pooled over stations -> {pooled_path}")

    cols = ["feature", "side", "crossings", "precision", "base", "lift", "recall"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        for q in args.q:
            top = pooled[pooled["q"] == q].sort_values("lift", ascending=False)
            print(f"\n=== q = {q:g}% per tail: top {args.top} by lift (pooled) ===")
            print(top[cols].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
