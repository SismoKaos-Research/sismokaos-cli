from pathlib import Path

import pandas as pd
import pytest

from sismokaos_plot.catalog import (events_near, haversine_km, load_catalog,
                                    load_station_coords)
from sismokaos_plot.cli import DEFAULT_CATALOG


def test_catalog_is_day_first_utc_and_sorted(catalog_csv):
    events = load_catalog(catalog_csv)
    assert list(events.columns[:4]) == ["time", "mag", "lat", "lon"]
    assert events["time"].is_monotonic_increasing
    main = events.loc[events["mag"].idxmax()]
    assert main["time"] == pd.Timestamp("2011-10-23 10:41:20")
    assert main["mag"] == 6.7


def test_station_coords_strip_bom_and_quotes(stations_csv):
    assert load_station_coords(stations_csv) == {"TVAN": (38.5235, 43.4044)}


def test_haversine_one_degree_of_latitude():
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.195, abs=0.01)


def test_events_near_filters_by_distance(catalog_csv, stations_csv):
    events = load_catalog(catalog_csv)
    near = events_near(events, load_station_coords(stations_csv)["TVAN"], 300)
    assert "Far away" not in set(near["location"])
    assert len(near) == 3
    assert near["dist_km"].between(0, 300).all()


@pytest.mark.skipif(not Path(DEFAULT_CATALOG).is_file(),
                    reason="AFAD catalog not on disk")
def test_real_catalog_has_van_main_shock_in_utc():
    events = load_catalog(DEFAULT_CATALOG)
    day = events[events["time"].dt.date == pd.Timestamp("2011-10-23").date()]
    main = day.loc[day["mag"].idxmax()]
    assert main["time"] == pd.Timestamp("2011-10-23 10:41:20")
    assert main["mag"] == pytest.approx(6.7)
