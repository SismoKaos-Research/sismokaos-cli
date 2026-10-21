import json

import numpy as np
import pandas as pd
import pytest

# The 66 feature columns sismokaos-cli writes (src/features/mod.rs), as found
# in output/van/VMUR_features.parquet. Every one must land on a named page.
PER_COMPONENT = [
    "CORR_DIM", "DOMINANT_FREQ", "HIGH_FREQ_ENERGY", "HJORTH_ACTIVITY",
    "HJORTH_COMPLEXITY", "HJORTH_MOBILITY", "KURTOSIS", "LOW_FREQ_ENERGY",
    "PEAK", "PERMUTATION_ENTROPY", "RMS", "ROS_LONG", "ROS_SHORT", "SAMP_ENT",
    "SKEWNESS", "SPECTRAL_CENTROID", "STA_LTA_Max", "STA_LTA_Mean",
    "STA_LTA_Median", "WOLF_LYE", "ZCR",
]
FEATURES = ([f"{c}_{f}" for c in "ENZ" for f in PER_COMPONENT]
            + ["EN_CROSS_CORR", "EZ_CROSS_CORR", "NZ_CROSS_CORR"])
COLUMNS = (["Pencere_ID", "Zaman_Dk"] + FEATURES
           + [f"{f}_DEV" for f in FEATURES]
           + ["E_INTERP_FRAC", "N_INTERP_FRAC", "Z_INTERP_FRAC"])

CATALOG_CSV = (
    "﻿Date,Longitude,Latitude,Depth,Rms,Type,Magnitude,Location,EventID\n"
    "23/10/2011 10:56:48,43.3633,38.7825,19.92,0.77,Ml,5.8,Tuşba (Van),141937\n"
    "23/10/2011 10:41:20,43.4657,38.689,19.02,0.28,Ml,6.7,Tuşba (Van),141933\n"
    "24/10/2011 03:00:00,43.40,38.70,10.0,0.5,Ml,3.1,Tuşba (Van),1\n"
    "24/10/2011 04:00:00,30.00,40.00,10.0,0.5,Ml,4.0,Far away,2\n"
)
STATIONS_CSV = (
    "﻿Network,Code,Longitude,Latitude,Height,Province,District\n"
    '"TU","TVAN",43.4044,38.5235,1970,"Van","Tuşba"\n'
)


@pytest.fixture
def catalog_csv(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(CATALOG_CSV, encoding="utf-8")
    return path


@pytest.fixture
def stations_csv(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text(STATIONS_CSV, encoding="utf-8")
    return path


def make_features(start, periods, step_s=50, seed=0):
    """A feature frame shaped like sismokaos-cli's output."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(start).value / 1e9 + np.arange(periods) * step_s
    df = pd.DataFrame(rng.normal(size=(periods, len(COLUMNS))), columns=COLUMNS)
    df["Pencere_ID"] = [f"W{i}" for i in range(periods)]
    df["Zaman_Dk"] = t / 60.0
    for c in "ENZ":
        df[f"{c}_INTERP_FRAC"] = 0.0
    return df


@pytest.fixture
def run_parquet(tmp_path):
    """Two eras of TVAN, written as the CLI writes them."""
    out = tmp_path / "out"
    out.mkdir()
    for name, start in (("TVAN_2011_BH", "2011-10-22"), ("TVAN_2026_HH", "2026-01-04")):
        make_features(start, 4000).to_parquet(out / f"{name}_features.parquet")
        (out / f"{name}_features.run_metadata.json").write_text(
            json.dumps({"station": "TVAN", "step_sec": 50}))
    return out


CHAOS = ["WOLF_LYE", "ROS_SHORT", "ROS_LONG", "CORR_DIM", "SAMP_ENT"]
# Band name -> (window s, step s, windows): about 55 h each, from 2011-10-22,
# so the M6.7 and the events after it fall inside.
BANDS = {"0.1-1Hz": (400, 100, 2000), "1-5Hz": (40, 50, 4000)}


@pytest.fixture
def bands_run(tmp_path):
    """A chaos-only bands run of TVAN as the CLI writes it: <out-dir>/<band>/."""
    out = tmp_path / "bands_out"
    keep = ["Pencere_ID", "Zaman_Dk", *(f"{c}_INTERP_FRAC" for c in "ENZ"),
            *(f"{c}_{f}" for c in "ENZ" for f in CHAOS)]
    for band, (win, step, n) in BANDS.items():
        folder = out / band
        folder.mkdir(parents=True)
        df = make_features("2011-10-22", n, step_s=step)
        df[keep].to_parquet(folder / "TVAN_2011_BH_chaos_features.parquet")
        (folder / "TVAN_2011_BH_chaos_features.run_metadata.json").write_text(
            json.dumps({"station": "TVAN", "band": band, "win_sec": win,
                        "step_sec": step}))
    return out


@pytest.fixture
def catalog_with_foreshock(tmp_path):
    """The catalog plus an event the day before the M6.7, so the pre-main
    analysis has something to match."""
    path = tmp_path / "catalog_foreshock.csv"
    path.write_text(CATALOG_CSV
                    + "22/10/2011 12:00:00,43.40,38.70,10.0,0.5,Ml,3.5,Tuşba (Van),3\n",
                    encoding="utf-8")
    return path
