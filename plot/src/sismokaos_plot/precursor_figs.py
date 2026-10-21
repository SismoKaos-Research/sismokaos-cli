"""Figures for ``sismokaos-precursors --figs``.

  target_<STA>_<band>_<time>_M<mag>.png
      Every feature over the hours before one target: its bounds at the
      printed q, the crossing windows, and shaded where windows were dropped
      for holding an event or its coda; the catalog above.
  summary_<band>.png
      Pooled lift per feature against lead time and side, at the printed q,
      with the day-shift p-value.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .features import COMPONENTS, PAGES  # noqa: E402
from .threshold_figs import (BOUND, EVENT, GRID, INK, INK_2, LIFT_CMAP,  # noqa: E402
                             SURFACE, _style)

RAW = "#9ec5f4"
DROPPED = "#e4e3df"
LOW = "#256abf"
HIGH = "#e34948"


def _runs(mask: np.ndarray):
    """(start, stop) index pairs of the True runs in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return zip(edges[::2], edges[1::2])


def _feature_rows(columns):
    """(feature, [E, N, Z column or None]) in page order."""
    order = [f for feats in PAGES.values() for f in feats]
    feats = sorted({c.split("_", 1)[1] for c in columns},
                   key=lambda f: order.index(f) if f in order else len(order))
    return [(f, [f"{c}_{f}" if f"{c}_{f}" in columns else None for c in COMPONENTS])
            for f in feats]


def fig_target(ends, values, bounds, dirty, events, target, title, path,
               before_h=24.0, after_h=2.0):
    """One target's lead-up. ``values`` column -> per-window array over
    ``ends``; ``bounds`` column -> (lower, upper)."""
    _style()
    t = pd.Timestamp(target)
    t0, t1 = t - pd.Timedelta(hours=before_h), t + pd.Timedelta(hours=after_h)
    sel = (ends >= np.datetime64(t0)) & (ends <= np.datetime64(t1))
    if sel.sum() < 2:
        return None
    hours = (ends[sel] - np.datetime64(t)) / np.timedelta64(3600, "s")
    gone = dirty[sel]
    rows = _feature_rows(list(values))
    fig = plt.figure(figsize=(15, 1.6 + 1.55 * len(rows)))
    grid = fig.add_gridspec(len(rows) + 1, 3, height_ratios=[1.1] + [1] * len(rows),
                            hspace=0.12, wspace=0.16, left=0.07, right=0.99,
                            top=1 - 0.7 / fig.get_figheight(),
                            bottom=0.5 / fig.get_figheight())
    ev = events[(events["time"] >= t0) & (events["time"] <= t1)]
    ev_h = (ev["time"] - t) / pd.Timedelta(hours=1)
    lanes = []
    for c in range(3):
        lane = fig.add_subplot(grid[0, c], sharex=lanes[0] if lanes else None)
        lane.vlines(ev_h, 0, ev["mag"], color=INK_2, lw=0.8)
        lane.scatter(ev_h, ev["mag"], s=10 + 6 * ev["mag"].clip(lower=0) ** 1.6,
                     color=INK_2, zorder=3)
        lane.axvline(0, color=EVENT, lw=1.4)
        lane.set_ylim(0, max(3.5, float(ev["mag"].max()) + 0.5) if len(ev) else 3.5)
        lane.set_title(f"{COMPONENTS[c]} component", fontsize=9)
        lane.tick_params(labelbottom=False, labelsize=7)
        lane.grid(True, axis="y", color=GRID, lw=0.5)
        if c == 0:
            lane.set_ylabel("Mag", fontsize=8)
        lanes.append(lane)

    for r, (feat, cells) in enumerate(rows, 1):
        for c, col in enumerate(cells):
            ax = fig.add_subplot(grid[r, c], sharex=lanes[0])
            ax.tick_params(labelsize=7, labelbottom=(r == len(rows)))
            if c == 0:
                ax.set_ylabel(feat, fontsize=8)
            if col is None:
                continue
            v = values[col][sel]
            lo, hi = bounds[col]
            for a, b in _runs(gone):
                ax.axvspan(hours[a], hours[min(b, len(hours) - 1)], color=DROPPED,
                           lw=0, zorder=0)
            shown = np.where(gone, np.nan, v)
            ax.plot(hours, shown, color=RAW, lw=0.7, zorder=1)
            for mask, color in ((shown < lo, LOW), (shown > hi, HIGH)):
                ax.scatter(hours[mask], shown[mask], s=14, color=color, zorder=3, lw=0)
            for b in (lo, hi):
                ax.axhline(b, color=BOUND, lw=0.8, ls="--", zorder=2)
            ax.axvline(0, color=EVENT, lw=1.2, zorder=2)
            finite = shown[np.isfinite(shown)]
            if len(finite):
                span = [min(lo, finite.min()), max(hi, finite.max())]
                pad = 0.06 * (span[1] - span[0] or 1.0)
                ax.set_ylim(span[0] - pad, span[1] + pad)
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            if r == len(rows):
                ax.set_xlabel("hours from target origin", fontsize=8)
    lanes[0].set_xlim(-before_h, after_h)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left",
                 y=1 - 0.2 / fig.get_figheight(), va="top")
    fig.text(0.01, 0.005, "Dashed: the q-bounds of the scored windows; blue below, red "
             "above. Shaded: windows dropped for holding an event or its coda. "
             "Window end time.", fontsize=7.5, color=INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_summary(pooled: pd.DataFrame, band: str, q: float, path):
    """Lift per feature (rows) against side and lead (columns), at ``q``."""
    _style()
    t = pooled[(pooled["band"] == band) & (pooled["q"] == q) & (pooled["side"] != "either")]
    if t.empty:
        return None
    leads = sorted(t["lead_h"].unique())
    feats = [c for f, cells in _feature_rows(t["feature"].unique()) for c in cells if c]
    cols = [(side, lead) for side in ("lower", "upper") for lead in leads]
    lift = np.full((len(feats), len(cols)), np.nan)
    pval = np.full_like(lift, np.nan)
    for i, f in enumerate(feats):
        for j, (side, lead) in enumerate(cols):
            m = t[(t["feature"] == f) & (t["side"] == side) & (t["lead_h"] == lead)]
            if len(m):
                lift[i, j], pval[i, j] = m["lift"].iloc[0], m["p"].iloc[0]
    logv = np.log10(np.clip(lift, 1e-2, None))
    top = max(np.nanmax(np.abs(logv)) if np.isfinite(logv).any() else 0.5, 0.5)
    norm = TwoSlopeNorm(vmin=-top, vcenter=0.0, vmax=top)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3.2, 0.34 * len(feats) + 2.0))
    im = ax.imshow(logv, cmap=LIFT_CMAP, norm=norm, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(cols)):
            if np.isnan(lift[i, j]):
                continue
            sig = pval[i, j] < 0.05
            dark = abs(norm(logv[i, j]) - 0.5) > 0.32
            ax.text(j, i, f"{lift[i, j]:.1f}{'*' if sig else ''}", ha="center",
                    va="center", fontsize=8, fontweight="bold" if sig else "normal",
                    color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), [f"{lead:g} h" for _, lead in cols], fontsize=8)
    ax.set_yticks(range(len(feats)), feats, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(feats) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.axvline(len(leads) - 0.5, color=SURFACE, lw=6)
    for k, side in enumerate(("LOWER bound", "UPPER bound")):
        ax.text(k * len(leads) + (len(leads) - 1) / 2, -1.4, f"{side}, crossing within",
                ha="center", fontsize=8, color=INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ticks = [x for x in (0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30) if abs(np.log10(x)) <= top]
    cb.set_ticks(np.log10(ticks), labels=[f"{x:g}" for x in ticks])
    cb.set_label("lift (1 = chance)", color=INK_2)
    cb.outline.set_visible(False)
    n = int(t["targets"].iloc[0]) if "targets" in t else 0
    fig.suptitle(f"{band or 'all'}: crossings before targets, q = {q:g}% "
                 f"({n} targets pooled)", fontsize=11, x=0.02, ha="left")
    fig.text(0.02, 0.005, "* p < 0.05 against day-shifted crossings; with this many "
             "cells, about 1 in 20 is starred by chance.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
