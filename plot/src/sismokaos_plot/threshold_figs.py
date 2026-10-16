"""Figures for the threshold-crossing analysis (``sismokaos-thresholds``).

Reads the per station-era CSVs that tool writes and draws:

  1. lift_heatmap_<name>.png   lift of every feature type x side x q
  2. tradeoff_<name>.png       precision against recall as q tightens
  3. stations_<name>.png       lift per station-era, for the strongest features
  4. pre_vs_all.png            lift with and without the aftershock sequence
  5. example_crossings.png     one feature's bounds and crossings over a few
                               days, matched crossings and events marked

Feature types pool their components (E/N/Z, or EN/EZ/NZ for
cross-correlation): their counts are summed and the ratios recomputed.

Example:
    sismokaos-threshold-figs figs/van/thresholds --data output/van --station VMUR
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .catalog import events_near, load_catalog, load_station_coords  # noqa: E402
from .cli import DEFAULT_CATALOG, DEFAULT_STATIONS, find_parquets  # noqa: E402
from .features import PAGES, load_station, masked, split_eras, station_of  # noqa: E402
from .render import Style, pick_main  # noqa: E402
from .thresholds import detectable, finish, match_windows  # noqa: E402

# Reference palette of the dataviz method (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
BLUE_ARM = ["#104281", "#256abf", "#5598e7", "#9ec5f4"]
RED_ARM = ["#f5c4c3", "#ec8c8b", "#e34948", "#a32f2e"]
NEUTRAL = "#f0efec"
EVENT = "#e34948"
BOUND = "#52514e"

COUNTS = ["windows", "event_windows", "crossings", "hits", "events",
          "events_found"]
_COMPONENT = re.compile(r"^(?:[ENZ]|EN|EZ|NZ)_")

# Diverging around lift = 1 on a log axis: blue worse than chance, red better.
LIFT_CMAP = LinearSegmentedColormap.from_list(
    "lift", BLUE_ARM + [NEUTRAL] + RED_ARM)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.grid": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def feature_type(feature: str) -> str:
    return _COMPONENT.sub("", feature)


def pooled_by_type(table: pd.DataFrame, by=("ftype", "side", "q")) -> pd.DataFrame:
    """Sums counts over components (and whatever else is not in ``by``)."""
    t = table.assign(ftype=table["feature"].map(feature_type))
    sums = t.groupby(list(by), as_index=False)[COUNTS].sum()
    return pd.DataFrame([finish(r) for r in sums.to_dict("records")])


def _type_order(types):
    order = [f for feats in PAGES.values() for f in feats] + ["CROSS_CORR"]
    return [f for f in order if f in types] + sorted(set(types) - set(order))


def fig_heatmap(table, name, out, title):
    p = pooled_by_type(table)
    p = p[p["side"] != "either"]
    qs = sorted(p["q"].unique(), reverse=True)
    types = _type_order(p["ftype"].unique())
    cols = [(side, q) for side in ("lower", "upper") for q in qs]
    grid = np.full((len(types), len(cols)), np.nan)
    for i, ft in enumerate(types):
        for j, (side, q) in enumerate(cols):
            v = p[(p["ftype"] == ft) & (p["side"] == side) & (p["q"] == q)]["lift"]
            if len(v):
                grid[i, j] = v.iloc[0]

    logv = np.log10(np.clip(grid, 1e-2, None))
    top = max(np.nanmax(logv), 0.5)
    norm = TwoSlopeNorm(vmin=-top / 2, vcenter=0.0, vmax=top)
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(types) + 2.2))
    im = ax.imshow(logv, cmap=LIFT_CMAP, norm=norm, aspect="auto")
    for i in range(len(types)):
        for j in range(len(cols)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            dark = abs(norm(logv[i, j]) - 0.5) > 0.32
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center",
                    va="center", fontsize=8, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), [f"{q:g}%" for _, q in cols])
    ax.set_yticks(range(len(types)), types, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # White gaps between cells and between the two sides.
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(types) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.axvline(len(qs) - 0.5, color=SURFACE, lw=6)
    ax.xaxis.set_ticks_position("top")
    ax.text((len(qs) - 1) / 2, -1.25, "LOWER bound crossed (value < q-th pct)",
            ha="center", fontsize=9, color=INK_2)
    ax.text(len(qs) + (len(qs) - 1) / 2, -1.25,
            "UPPER bound crossed (value > (100-q)-th pct)", ha="center",
            fontsize=9, color=INK_2)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300)
             if norm.vmin <= np.log10(t) <= norm.vmax]
    cb.set_ticks(np.log10(ticks), labels=[f"{t:g}" for t in ticks])
    cb.set_label("lift (1 = chance)", color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Lift = share of crossing windows holding an event / "
             "share of all windows holding one. Components and stations pooled.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = out / f"lift_heatmap_{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def best_pairs(table, k=5, q=1.0):
    """The k strongest (feature type, side) by pooled lift at ``q``."""
    p = pooled_by_type(table)
    p = p[(p["q"] == q) & (p["side"] != "either")]
    return list(p.sort_values("lift", ascending=False)[["ftype", "side"]]
                .head(k).itertuples(index=False, name=None))


def fig_tradeoff(table, name, out, title, pairs):
    p = pooled_by_type(table)
    base = p["base"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(base, color=INK_2, lw=1, ls="--")
    ax.text(1.0, base, f" chance ({base:.1%})", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color=INK_2)
    for i, (ft, side) in enumerate(pairs):
        s = p[(p["ftype"] == ft) & (p["side"] == side)].sort_values("q", ascending=False)
        ax.plot(s["recall"], s["precision"], color=SERIES[i], lw=2,
                marker=MARKERS[i], ms=8, mec=SURFACE, mew=1.5,
                label=f"{ft} ({side})")
        # Legend + marker shape identify the series; direct labels collide
        # where the curves converge at the tightest q.
        if i == 0:
            for _, r in s.iterrows():
                ax.annotate(f"q={r['q']:g}%", (r["recall"], r["precision"]),
                            xytext=(0, -14), textcoords="offset points",
                            fontsize=7, color=INK_2, ha="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("recall: share of events with a crossing window")
    ax.set_ylabel("precision: share of crossing windows holding an event")
    ax.set_xlim(0, 0.75)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    path = out / f"tradeoff_{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_stations(table, name, out, title, pairs, q=1.0):
    t = table.assign(ftype=table["feature"].map(feature_type))
    se = pooled_by_type(t.assign(se=t["station"] + " " + t["era"].astype(str)),
                        by=("ftype", "side", "q", "se"))
    se = se[se["q"] == q]
    pooled = pooled_by_type(table)
    pooled = pooled[pooled["q"] == q]
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(pairs) + 1.6))
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    labels = []
    for i, (ft, side) in enumerate(pairs):
        y = len(pairs) - 1 - i
        rows = se[(se["ftype"] == ft) & (se["side"] == side)]
        jitter = np.linspace(-0.15, 0.15, len(rows)) if len(rows) > 1 else [0]
        ax.scatter(rows["lift"], y + np.asarray(jitter), s=40, color=SERIES[0],
                   alpha=0.55, lw=0, zorder=3)
        # Name any station-era at or below chance: the outliers worth chasing.
        for (_, r), dy in zip(rows.iterrows(), jitter):
            if r["lift"] <= 1:
                ax.annotate(r["se"], (r["lift"], y + dy), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=8, color=INK_2)
        lift = pooled[(pooled["ftype"] == ft) & (pooled["side"] == side)]["lift"].iloc[0]
        ax.plot([lift, lift], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4)
        labels.append((y, f"{ft} ({side})"))
    ax.set_yticks([y for y, _ in labels], [l for _, l in labels], fontsize=9)
    ax.axvline(1, color=INK_2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(f"lift at q = {q:g}% (1 = chance)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.scatter([], [], s=40, color=SERIES[0], alpha=0.55, lw=0, label="one station-era")
    ax.plot([], [], color=INK, lw=2, label="pooled")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    path = out / f"stations_{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_pre_vs_all(pre, every, out, pairs):
    a = pooled_by_type(every)
    b = pooled_by_type(pre)
    qs = sorted(a["q"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs) + 0.6, 3.6),
                             sharey=True)
    for ax, (ft, side) in zip(np.atleast_1d(axes), pairs):
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for (frame, color, label) in ((a, SERIES[1], "whole era"),
                                      (b, SERIES[0], "before main shock")):
            s = frame[(frame["ftype"] == ft) & (frame["side"] == side)]
            s = s.set_index("q").reindex(qs)
            ax.plot(range(len(qs)), s["lift"], color=color, lw=2, marker="o",
                    ms=6, mec=SURFACE, label=label)
        ax.set_xticks(range(len(qs)), [f"{q:g}%" for q in qs], fontsize=8)
        ax.set_title(f"{ft}\n({side})", fontsize=9)
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].set_ylabel("lift (1 = chance)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Lift with and without the aftershock sequence, by q",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    path = out / "pre_vs_all.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_example(data, station, catalog_path, stations_path, column, side, q,
                days, out, win_sec=200.0, travel_sec=90.0):
    """A stretch before the main shock: the feature, its bound, crossings
    coloured by whether they hold an event, and the events themselves."""
    parquets = [p for p in find_parquets([data]) if station_of(p) == station]
    df = load_station(parquets)
    coords = load_station_coords(stations_path)[station]
    events = events_near(load_catalog(catalog_path), coords, 300.0)
    era = split_eras(df, pd.Timedelta(days=30))[0]
    ev_era = events[(events["time"] >= era.index[0]) & (events["time"] <= era.index[-1])]
    main = pick_main(ev_era, Style())
    if main is not None:
        era = era[era.index < main["time"]]
    det = detectable(ev_era[ev_era["time"] < era.index[-1]])

    series = masked(era, column, 0.5)
    vals = series.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    bound = np.percentile(finite, q if side == "lower" else 100 - q)

    # The span of `days` holding the most detectable events within 100 km.
    near = det[det["dist_km"] < 100]["time"]
    span = pd.Timedelta(days=days)
    starts = near if len(near) else pd.Series([era.index[0]])
    counts = [((near >= s) & (near < s + span)).sum() for s in starts]
    t0 = max(starts.iloc[int(np.argmax(counts))] - pd.Timedelta(hours=6), era.index[0])
    t1 = t0 + span
    w = series[(series.index >= t0) & (series.index < t1)]
    ev = det[(det["time"] >= t0) & (det["time"] < t1)]

    reach = np.timedelta64(int((win_sec + travel_sec) * 1e9), "ns")
    holds, _, _ = match_windows(w.index.to_numpy(), np.sort(ev["time"].to_numpy()), reach)
    cross = (w < bound) if side == "lower" else (w > bound)
    cross = cross.to_numpy()

    fig, (lane, ax) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    lane.vlines(ev["time"], 0, ev["mag"], color=EVENT, lw=1.2)
    lane.scatter(ev["time"], ev["mag"], s=18, color=EVENT, zorder=3)
    lane.set_ylabel("Mag")
    lane.set_ylim(0, max(4.0, ev["mag"].max() + 0.5) if len(ev) else 4.0)
    lane.grid(True, axis="y", color=GRID, lw=0.6)
    lane.set_title(f"{station}  {column}  ({side} bound at q = {q:g}%): "
                   f"{t0:%Y-%m-%d} to {t1:%Y-%m-%d}, before the main shock",
                   fontsize=11, loc="left")

    ax.plot(w.index, w.to_numpy(), color="#9ec5f4", lw=0.8, zorder=1)
    ax.axhline(bound, color=BOUND, lw=1.2, ls="--")
    ax.text(1.0, bound, f" {side} bound", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color=INK_2)
    hit, miss = cross & holds, cross & ~holds
    ax.scatter(w.index[hit], w.to_numpy()[hit], s=26, color=EVENT, zorder=3,
               label=f"crossing with an event ({hit.sum()})")
    ax.scatter(w.index[miss], w.to_numpy()[miss], s=26, facecolor=SURFACE,
               edgecolor=INK_2, lw=1.2, zorder=3,
               label=f"crossing, no catalog event ({miss.sum()})")
    for t in ev["time"]:
        ax.axvline(t, color=EVENT, lw=0.6, alpha=0.35, zorder=0)
    if side == "upper" or finite.min() > 0 and np.nanmax(w) / max(np.nanmin(w), 1e-12) > 100:
        ax.set_yscale("log")
    ax.set_ylabel(column)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.text(0.01, 0.005, "Events: any magnitude within 100 km, M>=3 within 300 km "
             "(AFAD). A crossing holds an event when an origin lies in its 200 s "
             "window or up to 90 s before.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out / f"example_crossings_{station}_{column}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="sismokaos-threshold-figs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path,
                    help="folder with pre_main.csv and all.csv from sismokaos-thresholds")
    ap.add_argument("--data", type=Path, help="feature parquets, for the example figure")
    ap.add_argument("--station", default="VMUR")
    ap.add_argument("--example", default="E_KURTOSIS:upper:1",
                    help="column:side:q for the example figure")
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _style()
    pre = pd.read_csv(args.folder / "pre_main.csv")
    every = pd.read_csv(args.folder / "all.csv")
    pairs = best_pairs(pre)
    made = [
        fig_heatmap(pre, "pre_main", args.folder,
                    "Lift of percentile-bound crossings: before each main shock"),
        fig_heatmap(every, "all", args.folder,
                    "Lift of percentile-bound crossings: whole eras, aftershocks included"),
        fig_tradeoff(pre, "pre_main", args.folder,
                     "Tighter bounds: more precise, fewer events caught (before main shock)",
                     pairs),
        fig_stations(pre, "pre_main", args.folder,
                     "Is the lift consistent across stations? (q = 1%, before main shock)",
                     pairs + [("CORR_DIM", "lower")]),
        fig_pre_vs_all(pre, every, args.folder, pairs),
    ]
    if args.data:
        column, side, q = args.example.split(":")
        made.append(fig_example(args.data, args.station, args.catalog, args.stations,
                                column, side, float(q), args.days, args.folder))
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
