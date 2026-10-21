"""Do percentile-bound crossings come *before* events?

``sismokaos-thresholds`` counts a crossing when an event lies inside the
window, so it measures detection. This asks the forecasting question: does
a window crossing a bound end within ``--lead-hours`` *before* a target
event's origin, more often than chance?

Windows that hold an event, or end during its coda, are dropped first, so a
crossing caused by an event's own waves cannot score; any event within
300 km counts, whatever its magnitude. The coda is that of the CLI's
calibration: ``max(300 s, 3 * 10 ** ((M + 0.87) / 2))`` after the window
could first see the event. Bounds are the q-th and (100-q)-th percentiles of
the remaining windows.

Targets are events within ``--target-dist`` km at or above ``--target-mag``.
``--quiet-hours H`` keeps only targets with no other catalog event within
that distance in the H hours before them: in a sequence, crossings left by
earlier events precede the later ones, which is clustering, not
anticipation.
By default only data before each era's main shock are used, the main shock
itself being a target: in an aftershock sequence nearly every window
precedes an event.

Reported per station-era-band, feature, side, q and lead, and pooled per
band:
  base       share of windows followed by a target within the lead
  precision  share of crossing windows followed by one
  lift       precision / base; 1 is chance
  recall     targets with a crossing window in the lead before them, of
             those with any usable window there
  p          share of day-shifted catalogs giving at least this lift

The null shifts the crossing series against the windows by whole days,
which keeps crossings' clustering and their alignment with the daily cycle.
Hundreds of combinations are tested, so the summary prints how many reach
p < 0.05 against how many chance would give.

Crossings left by an event's coda outlasting the formula, followed by the
next event of a cluster, still score: a clustered catalog flatters any
feature that stays disturbed after events.

``--figs DIR`` draws each target's lead-up and a lift summary per band
(see ``precursor_figs``).

Examples:
    sismokaos-precursors output/focus_chaos --out figs/focus/precursors/precursors.csv
    sismokaos-precursors output/focus_chaos --lead-hours 3 12 --target-mag 2.5 --remove-daily
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import events_near, load_catalog, load_station_coords
from .cli import DEFAULT_CATALOG, DEFAULT_STATIONS, find_parquets
from .precursor_figs import fig_summary, fig_target
from .features import (feature_columns, group_runs, load_station, masked,
                       remove_daily_cycle, split_eras, utc_offset_hours,
                       win_sec_of)
from .render import Style, pick_main
from .thresholds import SIDES

DAY_S = 86400.0
# Event influence radius: as in sismokaos-thresholds and the calibration.
NEAR_KM = 300.0
MIN_DAYS = 3


def coda_sec(mag) -> np.ndarray:
    """Seconds after an event's arrival that a window is not quiet, as the
    CLI's calibration has it (src/catalog.rs ``post_event_sec``)."""
    return np.maximum(300.0, 3.0 * 10.0 ** ((np.asarray(mag, dtype=float) + 0.87) / 2.0))


def in_intervals(t: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Whether each time lies in any [start, stop]; intervals may overlap."""
    if not len(starts):
        return np.zeros(len(t), dtype=bool)
    order = np.argsort(starts)
    s = starts[order]
    reach = np.maximum.accumulate(stops[order])
    i = np.searchsorted(s, t, side="right") - 1
    return (i >= 0) & (t <= reach[np.clip(i, 0, None)])


def contaminated(ends: np.ndarray, origins: np.ndarray, mags: np.ndarray,
                 reach_s: float) -> np.ndarray:
    """Windows that hold an event or end in its coda: an origin in
    ``[end - reach - coda, end]``."""
    stops = origins + ((reach_s + coda_sec(mags)) * 1e9).astype("timedelta64[ns]")
    return in_intervals(ends, origins, stops)


def followed(ends: np.ndarray, targets: np.ndarray, lead: np.timedelta64) -> np.ndarray:
    """Whether a target origin lies in ``(end, end + lead]`` of each window."""
    lo = np.searchsorted(targets, ends, side="right")
    hi = np.searchsorted(targets, ends + lead, side="right")
    return hi > lo


def lead_spans(ends: np.ndarray, targets: np.ndarray, lead: np.timedelta64):
    """Per target, the [first, last) windows ending in ``[origin - lead, origin)``."""
    return (np.searchsorted(ends, targets - lead, side="left"),
            np.searchsorted(ends, targets, side="left"))


def score_era(values: dict[str, np.ndarray], ends: np.ndarray, usable: np.ndarray,
              targets: np.ndarray, leads_h, qs, shifts: np.ndarray) -> list[dict]:
    """Raw counts, with null hit and crossing counts per shift, for one era.

    ``values`` maps column -> per-window values; ``usable`` marks windows
    clear of every event; ``shifts`` are circular shifts in windows.
    """
    n = len(ends)
    rows = []
    for lead_h in leads_h:
        lead = np.timedelta64(int(lead_h * 3600e9), "ns")
        pos = followed(ends, targets, lead)
        first, last = lead_spans(ends, targets, lead)
        usable_sum = np.concatenate([[0], np.cumsum(usable)])
        reachable = (usable_sum[last] - usable_sum[first]) > 0
        good = usable & pos
        for column, v in values.items():
            valid = usable & np.isfinite(v)
            if valid.sum() < 100:
                continue
            for q in qs:
                lo, hi = np.percentile(v[valid], [q, 100 - q])
                below, above = valid & (v < lo), valid & (v > hi)
                for side, cross in zip(SIDES, (below, above, below | above)):
                    csum = np.concatenate([[0], np.cumsum(cross)])
                    found = ((csum[last] - csum[first]) > 0) & reachable
                    idx = np.flatnonzero(cross)
                    moved = (idx[:, None] + shifts[None, :]) % n
                    rows.append({
                        "feature": column, "side": side, "q": q, "lead_h": lead_h,
                        "windows": int(valid.sum()),
                        "positives": int((valid & pos).sum()),
                        "crossings": int(len(idx)),
                        "hits": int(good[idx].sum()),
                        "targets": int(len(targets)),
                        "targets_reachable": int(reachable.sum()),
                        "found": int(found.sum()),
                        "null_hits": (good & valid)[moved].sum(axis=0),
                        "null_crossings": valid[moved].sum(axis=0),
                    })
    return rows


def finish(c: dict) -> dict:
    """Ratios and the shift p-value from summed counts; drops the null arrays."""
    out = {k: v for k, v in c.items() if not k.startswith("null_")}
    base = c["positives"] / c["windows"] if c["windows"] else np.nan
    precision = c["hits"] / c["crossings"] if c["crossings"] else np.nan
    out["base"] = base
    out["precision"] = precision
    out["lift"] = precision / base if base else np.nan
    out["recall"] = (c["found"] / c["targets_reachable"]
                     if c["targets_reachable"] else np.nan)
    nh, nc = np.asarray(c["null_hits"], float), np.asarray(c["null_crossings"], float)
    if len(nh) and np.isfinite(precision):
        null_prec = np.divide(nh, nc, out=np.zeros_like(nh), where=nc > 0)
        out["p"] = (1 + np.sum(null_prec >= precision - 1e-12)) / (len(nh) + 1)
    else:
        out["p"] = np.nan
    return out


def pool(rows: list[dict], keys) -> pd.DataFrame:
    """Rows summed over station-eras within ``keys``, nulls shift by shift
    (each era drew its own shifts), then finished. A group with an era too
    short for a null has none."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r[c] for c in keys)
        acc = groups.setdefault(k, {c: r[c] for c in keys})
        for c, v in r.items():
            if c in keys or c in ("station", "era"):
                continue
            if not c.startswith("null_"):
                acc[c] = acc.get(c, 0) + v
            elif c not in acc:
                acc[c] = np.asarray(v)
            else:
                acc[c] = acc[c] + v if len(acc[c]) == len(v) else np.zeros(0)
    return pd.DataFrame([finish(c) for c in groups.values()])


def isolated(targets: np.ndarray, others: np.ndarray, hours: float) -> np.ndarray:
    """Whether no time in ``others`` (sorted) lies in the ``hours`` before
    each target."""
    span = np.timedelta64(int(hours * 3600e9), "ns")
    lo = np.searchsorted(others, targets - span, side="left")
    hi = np.searchsorted(others, targets, side="left")
    return hi == lo


def day_shifts(n: int, step_s: float, rng, count: int) -> np.ndarray:
    """``count`` circular shifts of whole days, in windows, or none if the
    era is shorter than ``MIN_DAYS``."""
    per_day = DAY_S / step_s
    days = int(n / per_day)
    if days < MIN_DAYS:
        return np.zeros(0, dtype=int)
    return np.round(rng.integers(1, days, size=count) * per_day).astype(int)


def _draw_targets(args, name, station, band, ends, values, dirty, events, near,
                  targets):
    """Each target's lead-up, bounds as scored at ``--show-q``."""
    bounds = {}
    for c, v in values.items():
        ok = ~dirty & np.isfinite(v)
        if ok.sum() >= 100:
            bounds[c] = tuple(np.percentile(v[ok], [args.show_q, 100 - args.show_q]))
    shown = {c: values[c] for c in bounds}
    before, after = args.fig_hours
    for t in targets:
        e = near[near["time"] == t].iloc[0]
        when = pd.Timestamp(t)
        title = (f"{name}  |  target M{e['mag']:.1f} {e.get('location', '')}, "
                 f"{e['dist_km']:.0f} km, {when:%Y-%m-%d %H:%M:%S} UTC  |  "
                 f"bounds at q = {args.show_q:g}%")
        path = (args.figs / station / (band or "all")
                / f"target_{when:%Y%m%dT%H%M%S}_M{e['mag']:.1f}.png")
        fig_target(ends, shown, bounds, dirty, events, t, title, path, before, after)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="sismokaos-precursors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("precursors.csv"),
                    help="CSV of every station-era-band row; pooled per band beside it")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    ap.add_argument("--station", action="append", help="only these stations")
    ap.add_argument("--band", action="append", help="only these bands, e.g. 1-5Hz")
    ap.add_argument("--lead-hours", type=float, nargs="+", default=[1, 6, 24])
    ap.add_argument("--q", type=float, nargs="+", default=[5, 1, 0.5, 0.1],
                    help="tail percentages (5 1 0.5 0.1)")
    ap.add_argument("--target-mag", type=float, default=3.0,
                    help="targets are events at or above this (3.0)...")
    ap.add_argument("--target-dist", type=float, default=100.0,
                    help="...within this many km (100)")
    ap.add_argument("--quiet-hours", type=float, default=0.0,
                    help="only targets with no event within --target-dist in "
                         "this many hours before them (0: all)")
    ap.add_argument("--keep-aftershocks", action="store_true",
                    help="also use data after each era's main shock")
    ap.add_argument("--remove-daily", action="store_true",
                    help="take each feature's median by local hour out first")
    ap.add_argument("--win-sec", type=float, default=None,
                    help="window length in s; default: each run's metadata, else 200")
    ap.add_argument("--travel-sec", type=float, default=90.0)
    ap.add_argument("--max-interp", type=float, default=0.5)
    ap.add_argument("--era-gap-days", type=float, default=30.0)
    ap.add_argument("--shifts", type=int, default=200,
                    help="day-shifted nulls per era (200); p resolves to 1/(shifts+1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-q", type=float, default=1.0,
                    help="q of the printed top rows (1)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--figs", type=Path,
                    help="draw each target's lead-up and a summary per band here")
    ap.add_argument("--fig-hours", type=float, nargs=2, default=[24.0, 2.0],
                    metavar=("BEFORE", "AFTER"),
                    help="hours drawn around each target (24 2)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    catalog = load_catalog(args.catalog)
    coords = load_station_coords(args.stations)
    rng = np.random.default_rng(args.seed)
    gap = pd.Timedelta(days=args.era_gap_days)

    runs = group_runs(find_parquets(args.inputs))
    if args.station:
        runs = {k: ps for k, ps in runs.items() if k[0] in args.station}
    if args.band:
        runs = {k: ps for k, ps in runs.items() if k[1] in args.band}

    rows = []
    for (station, band), parquets in runs.items():
        name = f"{station}{f' {band}' if band else ''}"
        if station not in coords:
            print(f"[WARN] no coordinates for {station}; skipped")
            continue
        win_sec = args.win_sec or win_sec_of(parquets) or 200.0
        reach_s = win_sec + args.travel_sec
        df = load_station(parquets)
        columns = feature_columns(df.columns, dev=False)
        if args.remove_daily:
            df = remove_daily_cycle(df, columns, utc_offset_hours(coords[station][1]), gap)
        events = events_near(catalog, coords[station], NEAR_KM)
        for era in split_eras(df, gap):
            t0, t1 = era.index[0], era.index[-1]
            ev = events[(events["time"] >= t0 - pd.Timedelta(seconds=reach_s + 86400))
                        & (events["time"] <= t1 + pd.Timedelta(hours=max(args.lead_hours)))]
            if not args.keep_aftershocks:
                main_ev = pick_main(ev[(ev["time"] >= t0) & (ev["time"] <= t1)], Style())
                if main_ev is not None:
                    era = era[era.index < main_ev["time"]]
                    ev = ev[ev["time"] <= main_ev["time"]]
            if len(era) < 100:
                continue
            ends = era.index.to_numpy()
            dirty = contaminated(ends, ev["time"].to_numpy(), ev["mag"].to_numpy(), reach_s)
            near = ev[ev["dist_km"] <= args.target_dist]
            targets = np.sort(near.loc[near["mag"] >= args.target_mag, "time"].to_numpy())
            n_all = len(targets)
            if args.quiet_hours > 0:
                targets = targets[isolated(targets, np.sort(near["time"].to_numpy()),
                                           args.quiet_hours)]
            step_s = float(np.median(np.diff(ends)) / np.timedelta64(1, "s"))
            shifts = day_shifts(len(ends), step_s, rng, args.shifts)
            values = {c: masked(era, c, args.max_interp).to_numpy(dtype=float)
                      for c in columns}
            print(f"{name} {t0:%Y-%m-%d}..{era.index[-1]:%Y-%m-%d}: {len(era)} windows, "
                  f"{(~dirty).sum()} clear of events, {len(targets)} targets "
                  f"(M>={args.target_mag:g} within {args.target_dist:g} km"
                  + (f"; {n_all - len(targets)} dropped, not quiet for "
                     f"{args.quiet_hours:g} h before" if args.quiet_hours else "") + ")")
            if args.figs:
                _draw_targets(args, name, station, band, ends, values, dirty, ev,
                              near, targets)
            for r in score_era(values, ends, ~dirty, targets, args.lead_hours,
                               args.q, shifts):
                rows.append({"station": station, "band": band or "",
                             "era": f"{t0:%Y-%m-%d}", **r})

    if not rows:
        raise SystemExit("[ERROR] nothing to score")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([finish(r) for r in rows])
    table.to_csv(args.out, index=False)
    print(f"\nPer station-era results -> {args.out}")

    pooled = pool(rows, ("band", "feature", "side", "q", "lead_h"))
    if args.figs:
        for band in pooled["band"].unique():
            fig_summary(pooled, band, args.show_q,
                        args.figs / f"summary{f'_{band}' if band else ''}.png")
    pooled_path = args.out.with_name(args.out.stem + "_pooled.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"Pooled over stations, per band -> {pooled_path}")

    cols = ["feature", "side", "crossings", "precision", "base", "lift", "recall", "p"]
    sided = pooled[pooled["side"] != "either"]
    with pd.option_context("display.width", 150, "display.float_format", "{:.3f}".format):
        for band in pooled["band"].unique():
            b = sided[sided["band"] == band]
            tested = b["p"].notna().sum()
            sig = (b["p"] < 0.05).sum()
            print(f"\n##### {band or 'all'}: {sig} of {tested} feature/side/q/lead "
                  f"combinations at p < 0.05; chance alone gives about "
                  f"{0.05 * tested:.0f}")
            for lead_h in args.lead_hours:
                top = b[(b["q"] == args.show_q) & (b["lead_h"] == lead_h)]
                top = top.sort_values("lift", ascending=False).head(args.top)
                print(f"\n=== {band or 'all'}, crossing within {lead_h:g} h before "
                      f"a target, q = {args.show_q:g}% ===")
                print(top[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
