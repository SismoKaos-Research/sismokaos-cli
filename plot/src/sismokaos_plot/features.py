"""Feature parquets: loading, era splitting, and page grouping."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COMPONENTS = ("E", "N", "Z")
CROSS_PAIRS = ("EN", "EZ", "NZ")

# Page name -> per-component feature names, in row order. A per-component
# feature not listed here still gets drawn, on the "other" page.
PAGES: dict[str, tuple[str, ...]] = {
    "amplitude": ("PEAK", "RMS", "SKEWNESS", "KURTOSIS", "ZCR"),
    "sta_lta": ("STA_LTA_Max", "STA_LTA_Mean", "STA_LTA_Median"),
    "spectral": ("DOMINANT_FREQ", "SPECTRAL_CENTROID", "LOW_FREQ_ENERGY",
                 "HIGH_FREQ_ENERGY"),
    "hjorth": ("HJORTH_ACTIVITY", "HJORTH_MOBILITY", "HJORTH_COMPLEXITY"),
    "chaos": ("WOLF_LYE", "ROS_SHORT", "ROS_LONG", "CORR_DIM", "SAMP_ENT",
              "PERMUTATION_ENTROPY"),
}
CROSS_PAGE = "cross_corr"
OTHER_PAGE = "other"

# Not features: identifiers, and the gap-fill fractions used for masking.
_BOOKKEEPING = re.compile(r"^(Pencere_ID|Zaman_Dk|[ENZ]_INTERP_FRAC)$")


@dataclass
class Page:
    name: str
    # Each row is one feature; each cell the column drawn there, or None
    # where that component has no such column.
    rows: list[tuple[str, list[str | None]]] = field(default_factory=list)

    @property
    def columns(self) -> int:
        return max((len(cells) for _, cells in self.rows), default=0)


def feature_columns(columns, dev: bool) -> list[str]:
    """Feature columns to draw: the ``_DEV`` ones when ``dev``, else the rest."""
    out = []
    for c in columns:
        if _BOOKKEEPING.match(c):
            continue
        if c.endswith("_DEV") == dev:
            out.append(c)
    return out


def build_pages(columns, dev: bool = False) -> list[Page]:
    """Lays every feature column out on a page; none is left undrawn."""
    suffix = "_DEV" if dev else ""
    remaining = set(feature_columns(columns, dev))
    pages = []

    def take(name):
        if name in remaining:
            remaining.discard(name)
            return name
        return None

    for page_name, features in PAGES.items():
        page = Page(page_name)
        for feat in features:
            cells = [take(f"{c}_{feat}{suffix}") for c in COMPONENTS]
            if any(cells):
                page.rows.append((feat, cells))
        if page.rows:
            pages.append(page)

    cross = Page(CROSS_PAGE)
    for pair in CROSS_PAIRS:
        col = take(f"{pair}_CROSS_CORR{suffix}")
        if col:
            cross.rows.append((f"{pair} cross-corr", [col]))
    if cross.rows:
        pages.append(cross)

    # Whatever is left: per-component features grouped into E/N/Z rows,
    # anything else one per row.
    other = Page(OTHER_PAGE)
    per_comp = re.compile(rf"^([ENZ])_(.+){suffix}$")
    leftover_feats = sorted({m.group(2) for c in remaining
                             if (m := per_comp.match(c))})
    for feat in leftover_feats:
        cells = [take(f"{c}_{feat}{suffix}") for c in COMPONENTS]
        other.rows.append((feat, cells))
    for col in sorted(remaining):
        remaining.discard(col)
        other.rows.append((col, [col]))
    if other.rows:
        pages.append(other)
    return pages


def _metadata(parquet: Path) -> dict:
    meta = parquet.with_suffix(".run_metadata.json")
    return json.loads(meta.read_text()) if meta.is_file() else {}


def station_of(parquet: Path) -> str:
    """The station recorded in the run's metadata, else the filename prefix."""
    station = _metadata(parquet).get("station")
    if station:
        return str(station)
    return parquet.name.split("_")[0]


def band_of(parquet: Path) -> str | None:
    """The frequency band a ``bands`` run recorded, else None (a single-band
    run, or one made before bands existed)."""
    band = _metadata(parquet).get("band")
    return str(band) if band else None


def band_sort_key(band: str | None):
    """Orders bands low to high by their lower edge; unparseable names last."""
    if band is None:
        return (-1.0, "")
    try:
        return (float(band.split("-", 1)[0]), band)
    except ValueError:
        return (float("inf"), band)


def group_runs(parquets) -> dict[tuple[str, str | None], list[Path]]:
    """(station, band) -> parquets, sorted by station then band. Bands are
    never merged: each has its own windows, rate and calibration."""
    runs: dict[tuple[str, str | None], list[Path]] = {}
    for p in parquets:
        runs.setdefault((station_of(p), band_of(p)), []).append(p)
    return dict(sorted(runs.items(), key=lambda kv: (kv[0][0], band_sort_key(kv[0][1]))))


def win_sec_of(parquets) -> float | None:
    """The window length the runs recorded, or None if none did."""
    for p in parquets:
        win = _metadata(p).get("win_sec")
        if win:
            return float(win)
    return None


def load_station(parquets: list[Path]) -> pd.DataFrame:
    """All of one station's windows, time-sorted, indexed by window end time.

    ``Zaman_Dk`` is the window's END in epoch minutes, so a window first
    contains an event at the event's own time on this axis.
    """
    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(np.round(df["Zaman_Dk"].to_numpy() * 60.0, 3),
                              unit="s")
    df.index.name = "time"
    df = df.sort_index()
    return df[~df.index.duplicated(keep="first")]


def split_eras(df: pd.DataFrame, max_gap: pd.Timedelta) -> list[pd.DataFrame]:
    """Splits wherever consecutive windows are more than ``max_gap`` apart."""
    if df.empty:
        return []
    gaps = np.flatnonzero(np.diff(df.index.to_numpy()) > max_gap.to_timedelta64())
    bounds = [0, *(gaps + 1), len(df)]
    return [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def masked(df: pd.DataFrame, column: str, max_interp: float) -> pd.Series:
    """``column`` with windows more than ``max_interp`` gap-filled set to NaN.

    A cross-correlation column is masked when either of its components is.
    """
    values = df[column].astype(float)
    head = column.split("_", 1)[0]
    comps = [c for c in head if c in COMPONENTS] if len(head) <= 2 else []
    for c in comps:
        frac = df.get(f"{c}_INTERP_FRAC")
        if frac is not None:
            values = values.where(~(frac.to_numpy() > max_interp))
    return values


def break_gaps(series: pd.Series, max_step: pd.Timedelta):
    """(times, values) with a NaN inserted inside every gap longer than
    ``max_step``, so a line plot breaks there instead of bridging it."""
    t = series.index.to_numpy()
    v = series.to_numpy(dtype=float)
    if len(t) < 2:
        return t, v
    gaps = np.flatnonzero(np.diff(t) > max_step.to_timedelta64())
    if not len(gaps):
        return t, v
    mid = t[gaps] + (t[gaps + 1] - t[gaps]) / 2
    return np.insert(t, gaps + 1, mid), np.insert(v, gaps + 1, np.nan)


def utc_offset_hours(lon: float) -> float:
    """Local mean solar time minus UTC at longitude ``lon``, in hours. Used
    instead of a time zone, which for Turkey changed in 2016 and had DST
    before."""
    return lon / 15.0


def local_hour(index: pd.DatetimeIndex, offset_h: float) -> np.ndarray:
    """Hour of day, 0 to 24, of each (UTC) time at ``offset_h`` from UTC."""
    h = index.hour + index.minute / 60.0 + index.second / 3600.0
    return (np.asarray(h, dtype=float) + offset_h) % 24.0


def _log_scaled(values: np.ndarray) -> bool:
    """Whether a feature varies by factors rather than steps: all positive,
    and spanning more than 100x (the plot's log-axis rule)."""
    finite = values[np.isfinite(values)]
    if not len(finite) or finite.min() <= 0:
        return False
    lo, hi = np.percentile(finite, (0.1, 99.9))
    return hi / lo > 100.0


def daily_profile(df: pd.DataFrame, columns, offset_h: float) -> pd.DataFrame:
    """25th, 50th and 75th percentile of each column by local hour (0-23),
    as a frame indexed by hour with (column, quantile) columns."""
    hours = np.floor(local_hour(df.index, offset_h)).astype(int)
    grouped = df[list(columns)].astype(float).groupby(hours)
    return grouped.quantile([0.25, 0.5, 0.75]).unstack()


def remove_daily_cycle(df: pd.DataFrame, columns, offset_h: float,
                       max_gap: pd.Timedelta) -> pd.DataFrame:
    """``df`` with each column's median by local hour taken out, era by era,
    and the era's median added back so the level stays readable. Features
    that vary by factors are adjusted by factors."""
    eras = []
    for era in split_eras(df, max_gap):
        era = era.copy()
        hours = np.floor(local_hour(era.index, offset_h)).astype(int)
        for column in columns:
            values = era[column].to_numpy(dtype=float)
            log = _log_scaled(values)
            work = pd.Series(np.log10(values) if log else values, index=era.index)
            hourly = work.groupby(hours).transform("median")
            fixed = (work - hourly + work.median()).to_numpy()
            era[column] = 10.0 ** fixed if log else fixed
        eras.append(era)
    return pd.concat(eras) if eras else df


def in_hours(df: pd.DataFrame, offset_h: float, start: float, end: float) -> pd.DataFrame:
    """The windows ending between local hours ``start`` and ``end``; the span
    wraps midnight when ``start > end`` (22-4 is 22:00 to 04:00)."""
    h = local_hour(df.index, offset_h)
    keep = (h >= start) & (h < end) if start <= end else (h >= start) | (h < end)
    return df[keep]
