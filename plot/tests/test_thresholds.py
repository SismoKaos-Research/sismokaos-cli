import numpy as np
import pandas as pd

from sismokaos_plot.thresholds import detectable, finish, match_windows, score

S = np.timedelta64(1, "s")


def _t(seconds):
    return np.datetime64("2011-10-23T00:00:00", "ns") + np.asarray(seconds) * S


def test_window_holds_events_within_reach_before_its_end():
    ends = _t([200, 250, 300, 350, 1000])
    origins = _t([280])
    holds, first, last = match_windows(ends, origins, 290 * S)
    # Windows ending 300 and 350 contain origin 280 (ends 200 and 250 are
    # before it; 1000 - 290 = 710 is after it).
    assert holds.tolist() == [False, False, True, True, False]
    assert (first[0], last[0]) == (2, 4)


def test_score_counts_hits_recall_and_lift():
    ends = _t(np.arange(10) * 50 + 200)
    origins = _t([300, 640])
    holds, first, last = match_windows(ends, origins, 290 * S)
    cross = np.zeros(10, dtype=bool)
    # Window 0 (ends 200 s) holds nothing; window 2 (ends 300 s) holds the
    # first event; the second event (640 s) is only in window 9, not crossed.
    cross[[0, 2]] = True
    counts = score(cross, np.ones(10, dtype=bool), holds, first, last)
    assert counts["crossings"] == 2 and counts["hits"] == 1
    assert counts["events"] == 2 and counts["events_found"] == 1
    r = finish(counts)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["lift"] == r["precision"] / r["base"]


def test_detectable_uses_distance_class_floors():
    ev = pd.DataFrame({"time": pd.to_datetime(["2011-10-23"] * 4),
                       "mag": [1.0, 2.9, 3.0, 5.0],
                       "dist_km": [50.0, 150.0, 150.0, 400.0]})
    assert detectable(ev)["mag"].tolist() == [1.0, 3.0]
