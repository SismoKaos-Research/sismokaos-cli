import pandas as pd

from sismokaos_plot.threshold_figs import feature_type, pooled_by_type


def test_feature_type_strips_component_prefix():
    assert feature_type("E_KURTOSIS") == "KURTOSIS"
    assert feature_type("Z_STA_LTA_Max") == "STA_LTA_Max"
    assert feature_type("EN_CROSS_CORR") == "CROSS_CORR"


def test_components_are_pooled_by_summing_counts_not_averaging_ratios():
    rows = [
        {"feature": "E_PEAK", "side": "upper", "q": 1, "windows": 100,
         "event_windows": 10, "crossings": 10, "hits": 9, "events": 4,
         "events_found": 3},
        {"feature": "N_PEAK", "side": "upper", "q": 1, "windows": 100,
         "event_windows": 10, "crossings": 30, "hits": 3, "events": 4,
         "events_found": 1},
    ]
    p = pooled_by_type(pd.DataFrame(rows)).iloc[0]
    assert p["ftype"] == "PEAK" and p["crossings"] == 40 and p["hits"] == 12
    assert p["precision"] == 12 / 40  # not the mean of 0.9 and 0.1
    assert p["lift"] == (12 / 40) / (20 / 200)


def test_raw_figures_one_per_column_and_mode(run_parquet, catalog_csv,
                                             stations_csv, tmp_path):
    from conftest import FEATURES
    from sismokaos_plot.threshold_raw import main

    out = tmp_path / "raw"
    main([str(run_parquet), "--out", str(out), "--catalog", str(catalog_csv),
          "--stations", str(stations_csv), "--dpi", "30", "--jobs", "1"])
    for mode in ("all", "pre_main"):
        eras = sorted(p.name for p in (out / mode / "TVAN").iterdir())
        assert len(eras) == 2, (mode, eras)
        for era in eras:
            names = sorted(p.stem for p in (out / mode / "TVAN" / era).glob("*.png"))
            assert names == sorted(FEATURES)
