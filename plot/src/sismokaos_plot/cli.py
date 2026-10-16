"""Plots every feature of sismokaos-cli runs with catalog events marked.

Each station's parquets are merged, split into eras wherever the data stop
for longer than ``--era-gap-days``, and drawn one page per feature group per
era. Pages go to ``<out>/<STATION>/<era start>_<era end>/`` as PNGs and, all
together, into ``<out>/<STATION>/<STATION>_features.pdf``. Every graph is also
drawn on its own, at high resolution, into that era folder's ``single/``.

Examples:
    sismokaos-plot output/van --out figs/van
    sismokaos-plot output/van/TVAN_*.parquet --around 2011-10-23T10:41:20 --days 7
    sismokaos-plot output/van --dev --station VMUR
"""

import argparse
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .catalog import events_near, load_catalog, load_station_coords
from .features import build_pages, load_station, split_eras, station_of
from .render import Style, render_page, render_single

# plot/src/sismokaos_plot/cli.py -> the directory holding the sibling repos.
_PROJECTS = Path(__file__).resolve().parents[4]
DEFAULT_CATALOG = _PROJECTS / "data_downloader/catalogs/catalog_current.csv"
DEFAULT_STATIONS = _PROJECTS / "data_downloader/catalogs/istasyon_katalog.csv"


def find_parquets(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found.extend(sorted(p.glob("*_features.parquet")))
        elif p.is_file():
            found.append(p)
        else:
            raise SystemExit(f"[ERROR] not found: {p}")
    if not found:
        raise SystemExit("[ERROR] no *_features.parquet among the inputs")
    return found


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="sismokaos-plot", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="feature parquets, or folders holding *_features.parquet")
    ap.add_argument("--out", type=Path, default=Path("figs"))
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--stations", type=Path, default=DEFAULT_STATIONS,
                    help="station coordinates CSV (Code, Latitude, Longitude)")
    ap.add_argument("--station", action="append",
                    help="only these stations (repeatable)")
    ap.add_argument("--max-dist", type=float, default=300.0,
                    help="km; events further away are not drawn (default 300)")
    ap.add_argument("--line-mag", type=float, default=4.5,
                    help="events at or above this get a vertical line (4.5)")
    ap.add_argument("--line-dist", type=float, default=100.0,
                    help="...if within this many km (100)")
    ap.add_argument("--max-interp", type=float, default=0.5,
                    help="blank windows more gap-filled than this fraction (0.5)")
    ap.add_argument("--smooth", default="1h",
                    help="rolling-median width as a pandas offset, or 'none'")
    ap.add_argument("--era-gap-days", type=float, default=30.0)
    ap.add_argument("--dev", action="store_true",
                    help="plot the _DEV (first difference) columns instead")
    ap.add_argument("--around", action="append", default=[],
                    help="UTC time to add zoomed pages around (repeatable)")
    ap.add_argument("--days", type=float, default=7.0,
                    help="width of each --around zoom in days (7)")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-single", action="store_true",
                    help="skip the one-figure-per-graph output")
    ap.add_argument("--single-dpi", type=int, default=300,
                    help="resolution of the single-graph PNGs (300)")
    ap.add_argument("--single-format", choices=("png", "pdf", "svg"),
                    default="png", help="file type of the single graphs (png)")
    return ap.parse_args(argv)


def _save(fig, png: Path, pdf):
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    if pdf is not None:
        pdf.savefig(fig, dpi=110)
    plt.close(fig)


def _render_span(df, events, station, label, folder, style, args, pdf):
    t0, t1 = df.index[0], df.index[-1]
    ev = events[(events["time"] >= t0) & (events["time"] <= t1)]
    span = f"{t0:%Y-%m-%d} to {t1:%Y-%m-%d}"
    pages = build_pages(df.columns, dev=args.dev)
    cache = {}
    for i, page in enumerate(pages, 1):
        fig = render_page(df, page, ev, station, span, style, cache)
        _save(fig, folder / f"{i:02d}_{page.name}.png", pdf)
    singles = 0
    if not args.no_single:
        (folder / "single").mkdir(parents=True, exist_ok=True)
        for page in pages:
            for _, cells in page.rows:
                for column in filter(None, cells):
                    fig = render_single(df, column, ev, station, span, style, cache)
                    fig.savefig(folder / "single" / f"{column}.{args.single_format}",
                                dpi=args.single_dpi)
                    plt.close(fig)
                    singles += 1
    print(f"  {label}: {len(df)} windows, {len(ev)} events, "
          f"{len(pages)} pages, {singles} single graphs -> {folder}")


def main(argv=None):
    args = parse_args(argv)
    by_station = defaultdict(list)
    for p in find_parquets(args.inputs):
        by_station[station_of(p)].append(p)
    if args.station:
        by_station = {s: ps for s, ps in by_station.items() if s in args.station}
        if not by_station:
            raise SystemExit(f"[ERROR] none of {args.station} among the inputs")

    for path, what in ((args.catalog, "catalog"), (args.stations, "station list")):
        if not path.is_file():
            raise SystemExit(f"[ERROR] {what} not found: {path}")
    print(f"Catalog: {args.catalog}")
    catalog = load_catalog(args.catalog)
    coords = load_station_coords(args.stations)

    base = Style(max_interp=args.max_interp,
                 smooth=None if args.smooth.lower() == "none" else args.smooth,
                 line_mag=args.line_mag, line_dist_km=args.line_dist)
    suffix = "_dev" if args.dev else ""

    for station, parquets in sorted(by_station.items()):
        print(f"{station}: {', '.join(p.name for p in parquets)}")
        df = load_station(parquets)
        if station in coords:
            events = events_near(catalog, coords[station], args.max_dist)
        else:
            print(f"  [WARN] no coordinates for {station} in {args.stations}; "
                  "no events drawn")
            events = catalog.iloc[0:0].assign(dist_km=[])

        # Lines break at gaps over 2.5 window steps.
        step = pd.Timedelta(np.median(np.diff(df.index.to_numpy())))
        style = dataclasses.replace(base, max_step=step * 2.5)

        out = args.out / station
        out.mkdir(parents=True, exist_ok=True)
        pdf = None if args.no_pdf else PdfPages(out / f"{station}_features{suffix}.pdf")
        try:
            for era in split_eras(df, pd.Timedelta(days=args.era_gap_days)):
                label = f"{era.index[0]:%Y-%m-%d}_{era.index[-1]:%Y-%m-%d}"
                _render_span(era, events, station, label, out / f"{label}{suffix}",
                             style, args, pdf)
            for when in args.around:
                t = pd.Timestamp(when)
                t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
                half = pd.Timedelta(days=args.days / 2)
                zoom = df[(df.index >= t - half) & (df.index <= t + half)]
                if zoom.empty:
                    print(f"  [WARN] no {station} data within {args.days:g} days "
                          f"around {t}")
                    continue
                label = f"zoom_{t:%Y%m%dT%H%M}_{args.days:g}d"
                _render_span(zoom, events, station, label, out / f"{label}{suffix}",
                             style, args, pdf)
        finally:
            if pdf is not None:
                pdf.close()
                print(f"  PDF -> {out / f'{station}_features{suffix}.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
