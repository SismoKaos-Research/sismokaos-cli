import numpy as np
import pandas as pd

from conftest import BANDS
from sismokaos_plot import precursors
from sismokaos_plot.precursors import (contaminated, day_shifts, finish, followed,
                                       in_intervals, score_era)

S = np.timedelta64(1, "s")
T0 = np.datetime64("2020-01-01T00:00:00", "ns")


def test_in_intervals_handles_overlap_and_nesting():
    starts = T0 + np.array([0, 10, 100]) * S
    stops = T0 + np.array([50, 20, 110]) * S
    t = T0 + np.array([-1, 0, 15, 30, 50, 51, 105, 111]) * S
    assert in_intervals(t, starts, stops).tolist() == [
        False, True, True, True, True, False, True, False]


def test_contaminated_covers_the_window_and_the_coda():
    ends = T0 + np.arange(0, 3000, 50) * S
    # M2 coda: max(300, 3 * 10 ** 1.435) = 300 s; reach 130 s.
    dirty = contaminated(ends, np.array([T0 + 1000 * S]), np.array([2.0]), 130.0)
    hit = ((ends - T0) / S)[dirty]
    assert hit.min() == 1000 and hit.max() == 1400


def test_followed_means_strictly_after_the_window_end():
    ends = T0 + np.array([0, 100, 200]) * S
    pos = followed(ends, np.array([T0 + 100 * S]), np.timedelta64(150, "s"))
    # The window ending at the origin holds it; only the earlier one precedes.
    assert pos.tolist() == [True, False, False]


def _synthetic(seed=0, days=20, step=50):
    rng = np.random.default_rng(seed)
    ends = T0 + (np.arange(int(days * 86400 / step)) * step) * S
    targets = np.sort(T0 + rng.integers(86400, (days - 1) * 86400, 25) * S)
    return rng, ends, targets


def test_planted_precursor_scores_and_noise_does_not():
    rng, ends, targets = _synthetic()
    noise = rng.normal(size=len(ends))
    pre = rng.normal(size=len(ends))
    # Low in the 30 minutes before every target.
    before = followed(ends, targets, np.timedelta64(1800, "s"))
    pre[before] -= 10
    shifts = day_shifts(len(ends), 50.0, rng, 200)
    rows = score_era({"Z_PRE": pre, "Z_NOISE": noise}, ends,
                     np.ones(len(ends), dtype=bool), targets, [1], [1], shifts)
    got = {(r["feature"], r["side"]): finish(r) for r in rows}
    p = got[("Z_PRE", "lower")]
    assert p["lift"] > 10 and p["p"] < 0.01 and p["recall"] == 1.0
    assert got[("Z_NOISE", "lower")]["p"] > 0.05


def test_detection_alone_does_not_score_once_event_windows_are_dropped():
    rng, ends, targets = _synthetic(seed=1)
    det = rng.normal(size=len(ends))
    mags = np.full(len(targets), 3.5)
    dirty = contaminated(ends, targets, mags, 130.0)
    det[dirty] -= 10  # an event's own waves
    shifts = day_shifts(len(ends), 50.0, rng, 200)
    (row,) = [finish(r) for r in score_era({"Z_DET": det}, ends, ~dirty, targets,
                                           [1], [1], shifts) if r["side"] == "lower"]
    assert row["lift"] < 3 and row["p"] > 0.05


def test_cli_scores_every_band(bands_run, catalog_with_foreshock, stations_csv,
                               tmp_path, capsys):
    out = tmp_path / "pre" / "precursors.csv"
    precursors.main([str(bands_run), "--out", str(out), "--catalog",
                     str(catalog_with_foreshock), "--stations", str(stations_csv),
                     "--target-mag", "3", "--lead-hours", "1", "6", "--shifts", "20",
                     "--keep-aftershocks", "--figs", str(tmp_path / "pf")])
    table = pd.read_csv(out)
    assert sorted(table["band"].unique()) == sorted(BANDS)
    assert sorted(table["lead_h"].unique()) == [1, 6]
    pooled = pd.read_csv(out.with_name("precursors_pooled.csv"))
    assert {"lift", "p", "recall"} <= set(pooled.columns)
    assert "combinations at p < 0.05" in capsys.readouterr().out
    for band in BANDS:
        assert (tmp_path / "pf" / f"summary_{band}.png").is_file()
        assert list((tmp_path / "pf" / "TVAN" / band).glob("target_*.png"))


def test_isolated_drops_targets_with_a_recent_event():
    from sismokaos_plot.precursors import isolated
    targets = T0 + np.array([7200, 20000]) * S
    others = np.sort(T0 + np.array([5000, 7200, 20000]) * S)
    # 5000 s is within 1 h before 7200; nothing but itself precedes 20000.
    assert isolated(targets, others, 1.0).tolist() == [False, True]
