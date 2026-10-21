"""Plots every feature of sismokaos-cli runs with catalog events marked.

Each station's parquets are merged, split into eras wherever the data stop
for longer than ``--era-gap-days``, and drawn one page per feature group per
era. Pages go to ``<out>/<STATION>/<era start>_<era end>/`` as PNGs and, all
together, into ``<out>/<STATION>/<STATION>_features.pdf``. Every graph is also
drawn on its own, at high resolution, into that era folder's ``single/``.
``--before-main`` adds the lead-up to each era's main shock, into
``main_M<mag>_<time>_minus<N>d/`` and ``<STATION>_mainshock_zooms.pdf``.

Runs made with a ``bands`` config are drawn one band at a time: each band's
figures go to ``<out>/<STATION>/<band>/`` with the band in every title and
PDF name. A folder input is searched one level down too, so the run's
``--out-dir`` covers all its band folders.

Daily cycles (cultural noise, thermal) can swamp slow trends. ``--remove-daily``
takes each feature's median by hour of day out of it, era by era;
``--night 22-4`` keeps only windows ending in those hours. Hours are local
mean solar time at the station (``--utc-offset`` to use a clock instead).
Either one, or ``--daily-profile`` on its own, also draws each feature's
median by hour into ``daily_profile/``. Their figures go to folders and PDFs
suffixed ``_nodaily`` or ``_night<start>-<end>``.

Examples:
    sismokaos-plot output/van --out figs/van
    sismokaos-plot output/van/TVAN_*.parquet --around 2011-10-23T10:41:20 --days 7
    sismokaos-plot output/van --dev --station VMUR
    sismokaos-plot output/van --before-main 30 60 --no-eras
    sismokaos-plot output/focus_chaos --out figs/focus --band 1-5Hz --band 5-20Hz
    sismokaos-plot output/focus_chaos --out figs/focus --remove-daily --before-main 30
"""

import argparse
import contextlib
import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .catalog import events_near, load_catalog, load_station_coords
from .features import (build_pages, daily_profile, feature_columns, group_runs,
                       in_hours, load_station, remove_daily_cycle, split_eras,
                       utc_offset_hours)
from .render import (Style, pick_main, render_daily_profile, render_page,
                     render_single)

# plot/src/sismokaos_plot/cli.py -> the repo's own copy of the AFAD exports.
_CATALOGS = Path(__file__).resolve().parents[3] / "catalogs"
DEFAULT_CATALOG = _CATALOGS / "catalog_afad_2026-08-30.csv.gz"
DEFAULT_STATIONS = _CATALOGS / "istasyon_katalog.csv"


def find_parquets(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            # One level down as well: a bands run writes <out-dir>/<band>/.
            found.extend(sorted(p.glob("*_features.parquet")))
            found.extend(sorted(p.glob("*/*_features.parquet")))
        elif p.is_file():
            found.append(p)
        else:
            raise SystemExit(f"[ERROR] not found: {p}")
    if not found:
        raise SystemExit("[ERROR] no *_features.parquet among the inputs")
    return found


def _hours(text: str) -> tuple[float, float]:
    try:
        start, end = (float(x) for x in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START-END hours, e.g. 22-4: {text!r}")
    if not (0 <= start < 24 and 0 <= end <= 24) or start == end:
        raise argparse.ArgumentTypeError(f"hours must be 0-24 and differ: {text!r}")
    return start, end


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
    ap.add_argument("--band", action="append",
                    help="only these bands, by name, e.g. 1-5Hz (repeatable)")
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
    daily = ap.add_mutually_exclusive_group()
    daily.add_argument("--remove-daily", action="store_true",
                       help="take each feature's median by hour of day out of it")
    daily.add_argument("--night", type=_hours, metavar="START-END",
                       help="keep only windows ending in these local hours, e.g. 22-4")
    ap.add_argument("--daily-profile", action="store_true",
                    help="draw each feature's median by hour of day "
                         "(implied by --remove-daily and --night)")
    ap.add_argument("--utc-offset", type=float, metavar="HOURS",
                    help="local hours are UTC plus this; default: mean solar "
                         "time from the station's longitude")
    ap.add_argument("--around", action="append", default=[],
                    help="UTC time to add zoomed pages around (repeatable)")
    ap.add_argument("--days", type=float, default=7.0,
                    help="width of each --around zoom in days (7)")
    ap.add_argument("--before-main", type=float, nargs="+", default=[],
                    metavar="DAYS",
                    help="also draw each era's main shock from DAYS before it, "
                         "e.g. --before-main 30 60")
    ap.add_argument("--after-main", type=float, default=1.0, metavar="DAYS",
                    help="...up to this many days after it (1)")
    ap.add_argument("--no-eras", action="store_true",
                    help="skip the whole-era figures (draw only zooms)")
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


def _draw_profile(df, columns, offset, clock, title, out, args):
    """Each feature's median by hour of day, one figure per page, into
    ``<out>/daily_profile[_dev]/``. Drawn from the whole run, before any
    daily-cycle option changes it."""
    profile = daily_profile(df, columns, offset)
    folder = out / f"daily_profile{'_dev' if args.dev else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    span = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"
    pages = build_pages(df.columns, dev=args.dev)
    for i, page in enumerate(pages, 1):
        fig = render_daily_profile(profile, page, title, span, clock)
        fig.savefig(folder / f"{i:02d}_{page.name}.png", dpi=130)
        plt.close(fig)
    print(f"  daily profile ({clock}): {len(pages)} pages -> {folder}")


def main(argv=None):
    args = parse_args(argv)
    by_run = group_runs(find_parquets(args.inputs))
    if args.station:
        by_run = {k: ps for k, ps in by_run.items() if k[0] in args.station}
        if not by_run:
            raise SystemExit(f"[ERROR] none of {args.station} among the inputs")
    if args.band:
        by_run = {k: ps for k, ps in by_run.items() if k[1] in args.band}
        if not by_run:
            raise SystemExit(f"[ERROR] none of the bands {args.band} among the inputs")

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
    if args.remove_daily:
        suffix += "_nodaily"
    elif args.night:
        suffix += f"_night{args.night[0]:g}-{args.night[1]:g}"
    daily = args.remove_daily or args.night or args.daily_profile

    for (station, band), parquets in by_run.items():
        print(f"{station}{f' {band}' if band else ''}: "
              f"{', '.join(p.name for p in parquets)}")
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

        out = args.out / station / band if band else args.out / station
        out.mkdir(parents=True, exist_ok=True)
        # Titles and PDF names carry the band, so a figure says which it is
        # once it leaves its folder.
        prefix = f"{station}_{band}" if band else station
        title = f"{station}  |  {band}" if band else station

        if daily:
            if args.utc_offset is not None:
                offset, clock = args.utc_offset, f"UTC{args.utc_offset:+g}"
            elif station in coords:
                offset, clock = utc_offset_hours(coords[station][1]), "local solar time"
            else:
                offset, clock = 0.0, "UTC"
                print(f"  [WARN] no coordinates for {station}; hours are UTC")
            columns = feature_columns(df.columns, args.dev)
            _draw_profile(df, columns, offset, clock, title, out, args)
            if args.remove_daily:
                df = remove_daily_cycle(df, columns, offset,
                                        pd.Timedelta(days=args.era_gap_days))
                title += "  |  daily cycle removed"
            elif args.night:
                df = in_hours(df, offset, *args.night)
                title += f"  |  {args.night[0]:g}-{args.night[1]:g} h {clock}"
                if df.empty:
                    print("  [WARN] no windows in those hours; skipped")
                    continue
        eras = split_eras(df, pd.Timedelta(days=args.era_gap_days))

        with contextlib.ExitStack() as stack:
            pdfs = {}

            def pdf(name):
                """The PDF ``name``, opened on first use so a run that draws
                none of its pages leaves an existing one untouched."""
                if args.no_pdf:
                    return None
                if name not in pdfs:
                    path = out / f"{prefix}_{name}{suffix}.pdf"
                    pdfs[name] = stack.enter_context(PdfPages(path))
                    stack.callback(print, f"  PDF -> {path}")
                return pdfs[name]

            if not args.no_eras:
                for era in eras:
                    label = f"{era.index[0]:%Y-%m-%d}_{era.index[-1]:%Y-%m-%d}"
                    _render_span(era, events, title, label,
                                 out / f"{label}{suffix}", style, args,
                                 pdf("features"))

            for when in args.around:
                t = pd.Timestamp(when)
                t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
                half = pd.Timedelta(days=args.days / 2)
                zoom = df[(df.index >= t - half) & (df.index <= t + half)]
                if zoom.empty:
                    print(f"  [WARN] no {title} data within {args.days:g} days "
                          f"around {t}")
                    continue
                label = f"zoom_{t:%Y%m%dT%H%M}_{args.days:g}d"
                _render_span(zoom, events, title, label, out / f"{label}{suffix}",
                             style, args, pdf("features"))

            # Lead-ups to each era's main shock -- the same event the red
            # line marks -- ending --after-main days past it.
            after = pd.Timedelta(days=args.after_main)
            for era in eras if args.before_main else []:
                in_era = events[(events["time"] >= era.index[0])
                                & (events["time"] <= era.index[-1])]
                main_ev = pick_main(in_era, style)
                if main_ev is None:
                    print(f"  [WARN] no M>={style.line_mag:g} within "
                          f"{style.line_dist_km:g} km in {era.index[0]:%Y-%m-%d}"
                          f"..{era.index[-1]:%Y-%m-%d}; no main-shock zoom")
                    continue
                t = main_ev["time"]
                for days in args.before_main:
                    lead = era[(era.index >= t - pd.Timedelta(days=days))
                               & (era.index <= t + after)]
                    if lead.empty:
                        print(f"  [WARN] no {title} data in the {days:g} days "
                              f"before {t}")
                        continue
                    if lead.index[0] > t - pd.Timedelta(days=days) + pd.Timedelta(days=1):
                        print(f"  [NOTE] data start {lead.index[0]:%Y-%m-%d}, "
                              f"short of {days:g} days before {t:%Y-%m-%d}")
                    label = (f"main_M{main_ev['mag']:.1f}_{t:%Y%m%dT%H%M}"
                             f"_minus{days:g}d")
                    _render_span(lead, events, title, label,
                                 out / f"{label}{suffix}", style, args,
                                 pdf("mainshock_zooms"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
