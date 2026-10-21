import numpy as np
import pandas as pd

from sismokaos_plot import cli
from sismokaos_plot.features import (daily_profile, in_hours, local_hour,
                                     remove_daily_cycle, utc_offset_hours)

GAP = pd.Timedelta(days=30)


def _days(n_days, step_min=10, start="2020-01-01"):
    return pd.date_range(start, periods=int(n_days * 24 * 60 / step_min),
                         freq=f"{step_min}min")


def test_local_solar_hour_from_longitude():
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 22:30"])
    # 45 E is UTC+3; 22:30 UTC wraps to 01:30.
    assert local_hour(idx, utc_offset_hours(45.0)).tolist() == [3.0, 1.5]


def test_remove_daily_cycle_flattens_a_daily_wave_and_keeps_a_step():
    idx = _days(20)
    h = local_hour(idx, 3.0)
    wave = np.sin(2 * np.pi * h / 24)
    step = np.where(idx >= "2020-01-15", 1.0, 0.0)
    df = pd.DataFrame({"Z_X": 5 + wave + step}, index=idx)
    out = remove_daily_cycle(df, ["Z_X"], 3.0, GAP)["Z_X"]
    # The wave is gone; the step (a trend, not a daily cycle) survives, and
    # the level stays near the original median.
    resid = out - step
    assert resid.std() < 0.1 * wave.std()
    assert abs(out[idx >= "2020-01-16"].mean() - out[idx < "2020-01-14"].mean() - 1) < 0.1
    assert abs(out.median() - df["Z_X"].median()) < 0.3


def test_remove_daily_cycle_is_multiplicative_for_wide_positive_features():
    idx = _days(10)
    h = local_hour(idx, 0.0)
    rng = np.random.default_rng(0)
    amp = 10 ** (rng.uniform(0, 3, len(idx)) + np.sin(2 * np.pi * h / 24))
    out = remove_daily_cycle(pd.DataFrame({"E_RMS": amp}, index=idx),
                             ["E_RMS"], 0.0, GAP)["E_RMS"]
    assert (out > 0).all()
    prof = daily_profile(pd.DataFrame({"E_RMS": np.log10(out)}, index=idx),
                         ["E_RMS"], 0.0)[("E_RMS", 0.5)]
    assert prof.max() - prof.min() < 0.3


def test_in_hours_wraps_midnight():
    idx = _days(2, step_min=60)
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    kept = local_hour(in_hours(df, 0.0, 22, 4).index, 0.0)
    assert sorted(set(kept)) == [0.0, 1.0, 2.0, 3.0, 22.0, 23.0]


def test_cli_daily_options(bands_run, catalog_csv, stations_csv, tmp_path, capsys):
    out = tmp_path / "figs"
    common = [str(bands_run), "--out", str(out), "--catalog", str(catalog_csv),
              "--stations", str(stations_csv), "--no-single", "--no-pdf",
              "--band", "1-5Hz"]
    cli.main([*common, "--remove-daily"])
    cli.main([*common, "--night", "22-4"])
    band = out / "TVAN" / "1-5Hz"
    folders = sorted(p.name for p in band.iterdir())
    assert "daily_profile" in folders
    assert any(f.endswith("_nodaily") for f in folders)
    (night,) = [f for f in folders if f.endswith("_night22-4")]
    assert (band / "daily_profile" / "01_chaos.png").is_file()
    printed = capsys.readouterr().out
    assert "daily profile (local solar time)" in printed
    # 6 of 24 hours of a ~55 h run of 4000 windows: about a quarter.
    n = int(printed.split(f"{night[:-len('_night22-4')]}: ")[-1].split(" windows")[0])
    assert 800 < n < 1300
