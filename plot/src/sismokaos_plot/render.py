"""One figure per (station, era, page): an event lane over a grid of panels.

The event style follows ``chaos/scripts/plot_rosenstein.py`` (the Marmara
paper figures): markers on a magnitude axis coloured by magnitude and shaped
by distance class, and the main shock as a red vertical line.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from .features import Page, break_gaps, masked  # noqa: E402

MAIN_COLOR = "#d00000"
EVENT_LINE_COLOR = "#555555"
RAW_COLOR = "#9db4cf"
SMOOTH_COLOR = "#1f3f73"
CLIP_COLOR = "#c05a00"
MAG_MAX = 8.0
CMAP = "jet"
# Percentile range (hi / lo) beyond which a positive feature gets a log axis.
LOG_SPAN = 100.0

# (upper distance km, marker, legend text), as in plot_rosenstein.py.
DIST_CLASSES = [(100, "o", "0-100 km"), (300, "^", "100-300 km"),
                (500, "s", "300-500 km"), (1000, "*", "500-1000 km")]


@dataclass
class Style:
    """Drawing choices that the CLI exposes."""
    max_interp: float = 0.5
    smooth: str = "1h"
    clip_pct: tuple[float, float] = (0.1, 99.9)
    line_mag: float = 4.5
    line_dist_km: float = 100.0
    max_step: pd.Timedelta = pd.Timedelta(minutes=5)


def marker_size(mag):
    return 12.0 + 9.0 * np.clip(np.asarray(mag, dtype=float), 0, None) ** 1.6


def pick_main(events: pd.DataFrame, style: Style):
    """The largest event within ``line_dist_km`` at or above ``line_mag``."""
    near = events[(events["dist_km"] <= style.line_dist_km)
                  & (events["mag"] >= style.line_mag)]
    if near.empty:
        return None
    return near.loc[near["mag"].idxmax()]


def robust_limits(values: np.ndarray, pct: tuple[float, float]):
    """(lo, hi, log): percentile limits, on a log axis when the values are
    all positive and span more than ``LOG_SPAN`` -- amplitude features do,
    and on a linear axis their quiet stretches flatten to zero under the
    aftershocks."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    lo, hi = np.percentile(finite, pct)
    log = lo > 0 and hi / lo > LOG_SPAN
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
    if not hi > lo:
        span = abs(hi) * 0.05 or 1.0
        lo, hi = lo - span, hi + span
    else:
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    return (10 ** lo, 10 ** hi, True) if log else (lo, hi, False)


def _draw_lane(ax, events, main, t0, t1, legend):
    norm = Normalize(0.0, MAG_MAX)
    lower = 0.0
    shown = []
    for upper, marker, label in DIST_CLASSES:
        ev = events[(events["dist_km"] >= lower) & (events["dist_km"] < upper)]
        lower = upper
        if ev.empty:
            continue
        size = marker_size(ev["mag"]) * (2.2 if marker == "*" else 1.0) * 0.5
        ax.scatter(ev["time"], ev["mag"], c=ev["mag"], cmap=CMAP, norm=norm,
                   marker=marker, s=size, linewidths=0.3, edgecolors="black",
                   alpha=0.8, rasterized=True, zorder=3)
        shown.append(Line2D([], [], marker=marker, ls="", mfc="none",
                            mec="black", label=f"{label} ({len(ev)})"))
    if main is not None:
        ax.axvline(main["time"], color=MAIN_COLOR, lw=1.2, zorder=2)
    if main is not None and legend:
        # Text runs away from the nearer edge so it stays inside the axes.
        right_half = (main["time"] - t0) > (t1 - t0) / 2
        ax.annotate(
            f"M{main['mag']:.1f} {main.get('location', '')}\n"
            f"{main['time']:%Y-%m-%d %H:%M} UTC, {main['dist_km']:.0f} km",
            xy=(main["time"], 1.0), xycoords=("data", "axes fraction"),
            xytext=(-4 if right_half else 4, -2), textcoords="offset points",
            ha="right" if right_half else "left", va="top",
            fontsize=7, color=MAIN_COLOR)
    top = max(float(events["mag"].max()) + 0.5, 3.0) if len(events) else 3.0
    ax.set_ylim(0, top)
    ax.set_xlim(t0, t1)
    ax.grid(alpha=0.3, lw=0.5)
    if shown and legend:
        ax.set_ylabel("Mag")
        ax.legend(handles=shown, loc="upper left", fontsize=6, ncol=2,
                  frameon=True, framealpha=0.85)
    return plt.cm.ScalarMappable(norm=norm, cmap=CMAP)


def _prepared(df, column, style, cache):
    """(masked series, rolling median or None), reused across the page and
    single-graph figures of one span since the median is the slow part."""
    if cache is not None and column in cache:
        return cache[column]
    series = masked(df, column, style.max_interp)
    smooth = None
    if style.smooth:
        smooth = series.rolling(style.smooth, center=True, min_periods=3).median()
        smooth = smooth.where(series.notna())
    if cache is not None:
        cache[column] = (series, smooth)
    return series, smooth


def _event_marks(events, style):
    """(main shock or None, times of the other events that get a line)."""
    main = pick_main(events, style)
    lines = events.loc[(events["dist_km"] <= style.line_dist_km)
                       & (events["mag"] >= style.line_mag), "time"]
    if main is not None:
        lines = lines[lines != main["time"]]
    return main, lines


def _draw_panel(ax, df, column, style, lines, main, cache=None, fontsize=7,
                raw_lw=0.35, smooth_lw=0.8):
    series, smooth = _prepared(df, column, style, cache)
    t, v = break_gaps(series, style.max_step)
    ax.plot(t, v, color=RAW_COLOR, lw=raw_lw, rasterized=True, zorder=1)

    if smooth is not None:
        ts, vs = break_gaps(smooth, style.max_step)
        ax.plot(ts, vs, color=SMOOTH_COLOR, lw=smooth_lw, rasterized=True, zorder=2)

    limits = robust_limits(series.to_numpy(dtype=float), style.clip_pct)
    if limits is not None:
        lo, hi, log = limits
        if log:
            ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        # Values the percentile limits cut off: marked at the edge they cross,
        # so a spike at a large event stays visible instead of vanishing.
        edge = blended_transform_factory(ax.transData, ax.transAxes)
        vals = series.to_numpy(dtype=float)
        for mask, y, marker in ((vals > hi, 0.97, "^"), (vals < lo, 0.03, "v")):
            if mask.any():
                ax.scatter(series.index[mask], np.full(mask.sum(), y),
                           transform=edge, marker=marker, s=6, lw=0,
                           color=CLIP_COLOR, rasterized=True, zorder=4)

    for when in lines:
        ax.axvline(when, color=EVENT_LINE_COLOR, lw=0.5, ls="--", alpha=0.45,
                   zorder=0)
    if main is not None:
        ax.axvline(main["time"], color=MAIN_COLOR, lw=1.0, zorder=3)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)


def _date_axis(axes):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def render_single(df: pd.DataFrame, column: str, events: pd.DataFrame,
                  station: str, title_span: str, style: Style, cache=None):
    """One graph on its own: an event lane over a single feature panel."""
    t0, t1 = df.index[0], df.index[-1]
    main, lines = _event_marks(events, style)

    fig = plt.figure(figsize=(12.0, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.4], hspace=0.06,
                            left=0.08, right=0.92, top=0.92, bottom=0.09)
    lane = fig.add_subplot(grid[0])
    mappable = _draw_lane(lane, events, main, t0, t1, legend=True)
    lane.tick_params(labelbottom=False, labelsize=9)
    ax = fig.add_subplot(grid[1], sharex=lane)
    _draw_panel(ax, df, column, style, lines, main, cache, fontsize=9,
                raw_lw=0.4, smooth_lw=1.0)
    ax.set_ylabel(column, fontsize=10)
    _date_axis([lane, ax])

    box = lane.get_position()
    cax = fig.add_axes([0.935, box.y0, 0.01, box.height])
    fig.colorbar(mappable, cax=cax, label="Magnitude")
    cax.tick_params(labelsize=8)
    fig.suptitle(f"{station}  |  {column}  |  {title_span}  "
                 f"(window end time, UTC; {len(events)} events)", fontsize=12)
    return fig


def render_page(df: pd.DataFrame, page: Page, events: pd.DataFrame,
                station: str, title_span: str, style: Style, cache=None):
    """Returns the figure; the caller saves and closes it.

    ``events`` must already be cut to the era and carry ``dist_km``.
    """
    ncols = page.columns
    nrows = len(page.rows)
    t0, t1 = df.index[0], df.index[-1]
    main, lines = _event_marks(events, style)

    fig = plt.figure(figsize=(6.0 * ncols if ncols > 1 else 14.0,
                              1.8 + 1.9 * nrows))
    grid = fig.add_gridspec(nrows + 1, ncols,
                            height_ratios=[1.3] + [1.0] * nrows,
                            hspace=0.12, wspace=0.18,
                            left=0.06 if ncols > 1 else 0.08, right=0.94,
                            top=1 - 0.55 / fig.get_figheight(),
                            bottom=0.45 / fig.get_figheight())
    # One lane per column, on the same time axis as the panels under it, so
    # an event sits directly above its response in every column.
    lanes = []
    for c in range(ncols):
        lane = fig.add_subplot(grid[0, c], sharex=lanes[0] if lanes else None,
                               sharey=lanes[0] if lanes else None)
        mappable = _draw_lane(lane, events, main, t0, t1, legend=(c == 0))
        lane.tick_params(labelbottom=False, labelsize=7)
        if ncols > 1:
            lane.set_title(f"{'ENZ'[c]} component", fontsize=9)
        lanes.append(lane)

    axes = []
    for r, (feature, cells) in enumerate(page.rows):
        for c, column in enumerate(cells):
            ax = fig.add_subplot(grid[r + 1, c], sharex=lanes[0])
            axes.append(ax)
            if column is None:
                ax.text(0.5, 0.5, "no column", ha="center", va="center",
                        transform=ax.transAxes, color="0.5", fontsize=8)
            else:
                _draw_panel(ax, df, column, style, lines, main, cache)
            if c == 0:
                ax.set_ylabel(feature, fontsize=8)
            if r < nrows - 1:
                ax.tick_params(labelbottom=False)

    _date_axis([*lanes, *axes])

    box = lanes[-1].get_position()
    cax = fig.add_axes([0.952, box.y0, 0.008, box.height])
    fig.colorbar(mappable, cax=cax, label="Magnitude")
    cax.tick_params(labelsize=7)

    fig.suptitle(f"{station}  |  {page.name}  |  {title_span}  "
                 f"(window end time, UTC; {len(events)} events)",
                 fontsize=11, y=1 - 0.15 / fig.get_figheight(), va="top")
    return fig
