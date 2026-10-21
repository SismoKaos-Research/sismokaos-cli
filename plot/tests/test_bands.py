"""Runs made with a ``bands`` config: every tool keeps the bands apart."""

import pandas as pd

from conftest import BANDS
from sismokaos_plot import cli, threshold_figs, threshold_raw, thresholds


def test_plot_draws_bands_separately(bands_run, catalog_csv, stations_csv,
                                     tmp_path, capsys):
    out = tmp_path / "figs"
    cli.main([str(bands_run), "--out", str(out), "--catalog", str(catalog_csv),
              "--stations", str(stations_csv), "--single-dpi", "40"])

    station = out / "TVAN"
    assert sorted(p.name for p in station.iterdir()) == ["0.1-1Hz", "1-5Hz"]
    for band in BANDS:
        (era,) = [p for p in (station / band).iterdir() if p.is_dir()]
        assert [p.name for p in era.glob("*.png")] == ["01_chaos.png"]
        assert len(list((era / "single").glob("*.png"))) == 15
        assert (station / band / f"TVAN_{band}_features.pdf").is_file()
    printed = capsys.readouterr().out
    # Low band first, and each band's windows counted on their own.
    assert printed.index("TVAN 0.1-1Hz") < printed.index("TVAN 1-5Hz")
    assert ": 2000 windows" in printed and ": 4000 windows" in printed


def test_plot_band_filter(bands_run, catalog_csv, stations_csv, tmp_path):
    out = tmp_path / "figs"
    cli.main([str(bands_run), "--out", str(out), "--catalog", str(catalog_csv),
              "--stations", str(stations_csv), "--no-single", "--no-pdf",
              "--band", "1-5Hz"])
    assert sorted(p.name for p in (out / "TVAN").iterdir()) == ["1-5Hz"]


def _score(bands_run, catalog, stations, csv, *extra):
    thresholds.main([str(bands_run), "--out", str(csv), "--catalog", str(catalog),
                     "--stations", str(stations), *extra])
    return pd.read_csv(csv)


def test_thresholds_score_each_band_with_its_own_window(
        bands_run, catalog_csv, stations_csv, tmp_path):
    table = _score(bands_run, catalog_csv, stations_csv, tmp_path / "all.csv")
    row = table[(table["feature"] == "Z_SAMP_ENT") & (table["side"] == "upper")
                & (table["q"] == 1)].set_index("band")
    assert sorted(row.index) == sorted(BANDS)
    assert row.loc["0.1-1Hz", "windows"] == 2000
    assert row.loc["1-5Hz", "windows"] == 4000
    # An event is held by the windows ending within window + 90 s after it:
    # 490 s at a 100 s step is 4-5 windows, 130 s at a 50 s step 2-3. A
    # shared 200 s default would give 2-3 and 5-6.
    per_event = row["event_windows"] / row["events"]
    assert per_event["0.1-1Hz"] >= 4
    assert per_event["1-5Hz"] <= 3

    pooled = pd.read_csv(tmp_path / "all_pooled.csv")
    assert sorted(pooled["band"].unique()) == sorted(BANDS)


def test_threshold_figs_per_band(bands_run, catalog_with_foreshock, stations_csv,
                                 tmp_path):
    folder = tmp_path / "thr"
    _score(bands_run, catalog_with_foreshock, stations_csv, folder / "all.csv")
    _score(bands_run, catalog_with_foreshock, stations_csv, folder / "pre_main.csv",
           "--exclude-after-main")
    threshold_figs.main([str(folder), "--data", str(bands_run), "--station", "TVAN",
                         "--example", "Z_SAMP_ENT:upper:1",
                         "--catalog", str(catalog_with_foreshock),
                         "--stations", str(stations_csv)])
    for band in BANDS:
        for name in (f"lift_heatmap_pre_main_{band}", f"lift_heatmap_all_{band}",
                     f"tradeoff_pre_main_{band}", f"stations_pre_main_{band}",
                     f"pre_vs_all_{band}", f"example_crossings_TVAN_{band}_Z_SAMP_ENT"):
            assert (folder / f"{name}.png").is_file(), name
    assert not (folder / "lift_heatmap_pre_main.png").exists()


def test_threshold_raw_writes_a_folder_per_band(bands_run, catalog_csv,
                                                stations_csv, tmp_path):
    out = tmp_path / "raw"
    threshold_raw.main([str(bands_run), "--out", str(out), "--catalog", str(catalog_csv),
                        "--stations", str(stations_csv), "--mode", "all",
                        "--band", "1-5Hz", "--dpi", "30", "--jobs", "1"])
    assert sorted(p.name for p in (out / "all" / "TVAN").iterdir()) == ["1-5Hz"]
    (era,) = list((out / "all" / "TVAN" / "1-5Hz").iterdir())
    assert len(list(era.glob("*.png"))) == 15

