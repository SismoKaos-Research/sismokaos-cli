import numpy as np
import pandas as pd

from conftest import COLUMNS, FEATURES, make_features
from sismokaos_plot.features import (OTHER_PAGE, break_gaps, build_pages,
                                     load_station, masked, split_eras,
                                     station_of)


def _drawn(pages):
    return [col for p in pages for _, cells in p.rows for col in cells if col]


def test_every_feature_lands_on_a_named_page():
    pages = build_pages(COLUMNS)
    drawn = _drawn(pages)
    assert sorted(drawn) == sorted(FEATURES)
    assert len(drawn) == len(set(drawn)) == 66
    assert OTHER_PAGE not in [p.name for p in pages]


def test_dev_pages_draw_only_dev_columns():
    drawn = _drawn(build_pages(COLUMNS, dev=True))
    assert sorted(drawn) == sorted(f"{f}_DEV" for f in FEATURES)


def test_unknown_feature_goes_to_other_page_not_dropped():
    cols = COLUMNS + ["E_NEW_THING", "N_NEW_THING", "ODD"]
    pages = build_pages(cols)
    other = next(p for p in pages if p.name == OTHER_PAGE)
    assert ("NEW_THING", ["E_NEW_THING", "N_NEW_THING", None]) in other.rows
    assert ("ODD", ["ODD"]) in other.rows


def test_station_from_metadata_and_eras(run_parquet):
    parquets = sorted(run_parquet.glob("*.parquet"))
    assert {station_of(p) for p in parquets} == {"TVAN"}
    df = load_station(parquets)
    assert df.index.is_monotonic_increasing
    eras = split_eras(df, pd.Timedelta(days=30))
    assert [e.index[0].year for e in eras] == [2011, 2026]
    assert sum(len(e) for e in eras) == len(df) == 8000


def test_one_file_with_two_eras_is_split(tmp_path):
    df = pd.concat([make_features("2011-08-01", 100),
                    make_features("2026-01-04", 100)])
    path = tmp_path / "VMUR_features.parquet"
    df.to_parquet(path)
    assert len(split_eras(load_station([path]), pd.Timedelta(days=30))) == 2


def test_masked_blanks_gap_filled_windows():
    df = make_features("2011-10-22", 4)
    df["N_INTERP_FRAC"] = [0.0, 0.6, 0.5, 0.0]
    assert masked(df, "N_RMS", 0.5).isna().tolist() == [False, True, False, False]
    assert masked(df, "E_RMS", 0.5).notna().all()
    # Cross-correlation is lost when either of its components is.
    assert masked(df, "EN_CROSS_CORR", 0.5).isna().tolist() == [False, True, False, False]
    assert masked(df, "EZ_CROSS_CORR", 0.5).notna().all()


def test_break_gaps_inserts_nan_inside_long_gaps():
    idx = pd.to_datetime([0, 50, 100, 1000, 1050], unit="s")
    t, v = break_gaps(pd.Series([1.0, 2, 3, 4, 5], index=idx), pd.Timedelta(seconds=125))
    assert len(t) == 6
    assert np.isnan(v[3]) and t[2] < t[3] < t[4]
