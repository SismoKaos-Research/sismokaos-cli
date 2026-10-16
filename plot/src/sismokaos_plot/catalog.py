"""Earthquake catalog and station coordinates.

Both are the AFAD exports under ``data_downloader/catalogs``. The catalog's
``Date`` is day-first and already UTC: it lists the 2011 Van main shock at
23/10/2011 10:41:20, the international UTC origin time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CATALOG_TIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def load_catalog(path: Path) -> pd.DataFrame:
    """Returns ``time`` (naive UTC), ``mag``, ``lat``, ``lon``, ``location``."""
    # utf-8-sig: the AFAD export starts with a byte-order mark, which would
    # otherwise end up in the first column's name.
    raw = pd.read_csv(path, encoding="utf-8-sig")
    missing = {"Date", "Latitude", "Longitude", "Magnitude"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} lacks catalog column(s) {sorted(missing)}")
    events = pd.DataFrame({
        "time": pd.to_datetime(raw["Date"], format=CATALOG_TIME_FORMAT),
        "mag": pd.to_numeric(raw["Magnitude"], errors="coerce"),
        "lat": pd.to_numeric(raw["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(raw["Longitude"], errors="coerce"),
        "location": raw.get("Location", pd.Series("", index=raw.index)),
    })
    events = events.dropna(subset=["time", "mag", "lat", "lon"])
    return events.sort_values("time", ignore_index=True)


def load_station_coords(path: Path) -> dict[str, tuple[float, float]]:
    """Station code -> (latitude, longitude)."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(code).strip(): (float(lat), float(lon))
        for code, lat, lon in zip(raw["Code"], raw["Latitude"], raw["Longitude"])
    }


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def events_near(events: pd.DataFrame, coords: tuple[float, float],
                max_dist_km: float) -> pd.DataFrame:
    """Events within ``max_dist_km`` of ``coords``, with a ``dist_km`` column."""
    dist = haversine_km(coords[0], coords[1], events["lat"].to_numpy(),
                        events["lon"].to_numpy())
    near = events.assign(dist_km=dist)
    return near[near["dist_km"] <= max_dist_km].reset_index(drop=True)
