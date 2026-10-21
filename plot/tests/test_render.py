from conftest import FEATURES
from sismokaos_plot.cli import main


def test_renders_every_page_for_every_era(run_parquet, catalog_csv,
                                          stations_csv, tmp_path):
    out = tmp_path / "figs"
    main([str(run_parquet), "--out", str(out), "--catalog", str(catalog_csv),
          "--stations", str(stations_csv), "--single-dpi", "40",
          "--around", "2011-10-23T10:41:20", "--days", "1"])

    station = out / "TVAN"
    folders = sorted(p.name for p in station.iterdir() if p.is_dir())
    assert len(folders) == 3  # two eras and the zoom
    assert any(f.startswith("2011-10-22_") for f in folders)
    assert any(f.startswith("2026-01-04_") for f in folders)
    assert "zoom_20111023T1041_1d" in folders
    for f in folders:
        names = sorted(p.name for p in (station / f).glob("*.png"))
        assert names == ["01_amplitude.png", "02_sta_lta.png", "03_spectral.png",
                         "04_hjorth.png", "05_chaos.png", "06_cross_corr.png"]
        singles = sorted(p.stem for p in (station / f / "single").glob("*.png"))
        assert singles == sorted(FEATURES)
    assert (station / "TVAN_features.pdf").stat().st_size > 0


def test_single_graphs_can_be_skipped(run_parquet, catalog_csv, stations_csv,
                                      tmp_path):
    out = tmp_path / "figs"
    main([str(run_parquet), "--out", str(out), "--catalog", str(catalog_csv),
          "--stations", str(stations_csv), "--no-single", "--no-pdf"])
    assert not list(out.rglob("single"))


def test_main_shock_lead_up_zooms(run_parquet, catalog_csv, stations_csv,
                                  tmp_path, capsys):
    out = tmp_path / "figs"
    main([str(run_parquet), "--out", str(out), "--catalog", str(catalog_csv),
          "--stations", str(stations_csv), "--single-dpi", "40",
          "--before-main", "1", "--no-eras"])

    station = out / "TVAN"
    folders = sorted(p.name for p in station.iterdir() if p.is_dir())
    # The 2011 era holds the M6.7; the 2026 era has no qualifying event.
    assert folders == ["main_M6.7_20111023T1041_minus1d"]
    assert len(list((station / folders[0]).glob("*.png"))) == 6
    assert len(list((station / folders[0] / "single").glob("*.png"))) == 66
    assert (station / "TVAN_mainshock_zooms.pdf").is_file()
    # --no-eras drew no era pages, so it must not clobber the era PDF.
    assert not (station / "TVAN_features.pdf").exists()
    assert "no main-shock zoom" in capsys.readouterr().out

