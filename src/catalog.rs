//! Earthquake catalog and station coordinates, used to find the seismically
//! quiet stretches the chaos parameters are calibrated on.
//!
//! Both are AFAD exports, copied into `catalogs/`. The catalog's `Date` is
//! day-first and already UTC: it lists the 2011 Van main shock at
//! 23/10/2011 10:41:20, the international UTC origin time.

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::config::CalibrationConfig;

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub epoch: f64,
    pub mag: f64,
    pub lat: f64,
    pub lon: f64,
}

fn open(path: &Path) -> Result<csv::Reader<Box<dyn Read>>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(csv::ReaderBuilder::new().flexible(true).from_reader(reader))
}

/// Index of the first header matching one of `names`, ignoring case and a
/// leading byte-order mark.
fn column(headers: &csv::StringRecord, names: &[&str]) -> Option<usize> {
    headers.iter().position(|h| {
        let h = h.trim_start_matches('\u{feff}').trim();
        names.iter().any(|n| h.eq_ignore_ascii_case(n))
    })
}

fn number(record: &csv::StringRecord, idx: usize) -> Option<f64> {
    record.get(idx)?.trim().replace(',', ".").parse().ok()
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's
/// `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parses `dd/mm/YYYY HH:MM:SS[.fff]` as UTC epoch seconds.
pub fn parse_afad_time(s: &str) -> Option<f64> {
    let (date, clock) = s.trim().split_once(' ')?;
    let mut d = date.split('/').map(|p| p.parse::<i64>());
    let (day, month, year) = (d.next()?.ok()?, d.next()?.ok()?, d.next()?.ok()?);
    let mut c = clock.trim().split(':');
    let (h, mi): (i64, i64) = (c.next()?.parse().ok()?, c.next()?.parse().ok()?);
    let sec: f64 = c.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some((days_from_civil(year, month, day) * 86400 + h * 3600 + mi * 60) as f64 + sec)
}

/// Reads every event with a parseable time, magnitude and location.
pub fn load_events(path: &Path) -> Result<Vec<Event>, String> {
    let mut rdr = open(path)?;
    let headers = rdr.headers().map_err(|e| format!("{:?}: {}", path, e))?.clone();
    let need = |names: &[&str]| {
        column(&headers, names).ok_or_else(|| {
            format!("{:?} has no {} column; its columns are {:?}", path, names[0], headers)
        })
    };
    let (t, mag, lat, lon) = (
        need(&["Date"])?,
        need(&["Magnitude"])?,
        need(&["Latitude"])?,
        need(&["Longitude"])?,
    );

    let mut events = Vec::new();
    for record in rdr.records() {
        let record = record.map_err(|e| format!("{:?}: {}", path, e))?;
        let event = (|| {
            Some(Event {
                epoch: parse_afad_time(record.get(t)?)?,
                mag: number(&record, mag)?,
                lat: number(&record, lat)?,
                lon: number(&record, lon)?,
            })
        })();
        events.extend(event);
    }
    if events.is_empty() {
        return Err(format!("{:?} holds no readable events", path));
    }
    Ok(events)
}

/// Looks a station up by code. A code listed at more than one place (AFAD
/// reuses codes across networks) is an error rather than a guess.
pub fn station_coords(path: &Path, station: &str) -> Result<Option<(f64, f64)>, String> {
    let mut rdr = open(path)?;
    let headers = rdr.headers().map_err(|e| format!("{:?}: {}", path, e))?.clone();
    let (code, lat, lon) = match (
        column(&headers, &["Code", "station", "sta"]),
        column(&headers, &["Latitude", "lat"]),
        column(&headers, &["Longitude", "lon"]),
    ) {
        (Some(c), Some(a), Some(o)) => (c, a, o),
        _ => {
            return Err(format!(
                "{:?} needs code, latitude and longitude columns; its columns are {:?}",
                path, headers
            ));
        }
    };

    let mut found: Vec<(f64, f64)> = Vec::new();
    for record in rdr.records() {
        let record = record.map_err(|e| format!("{:?}: {}", path, e))?;
        if !record.get(code).is_some_and(|c| c.trim().eq_ignore_ascii_case(station)) {
            continue;
        }
        if let (Some(a), Some(o)) = (number(&record, lat), number(&record, lon)) {
            let same = |&(pa, po): &(f64, f64)| (pa - a).abs() < 1e-3 && (po - o).abs() < 1e-3;
            if !found.iter().any(same) {
                found.push((a, o));
            }
        }
    }
    if found.len() > 1 {
        return Err(format!(
            "station {} is listed at several places in {:?}: {:?}. Set \
             chaos_calibration.station_lat/station_lon in the config",
            station, path, found
        ));
    }
    Ok(found.first().copied())
}

pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * 6371.0 * a.sqrt().asin()
}

/// Seconds after an origin time during which the data are not quiet.
///
/// Inverts the duration magnitude of Lee et al. (1972), `Md = 2 log10(T) -
/// 0.87`, with the distance term dropped, and scales the coda duration by a
/// safety factor. M2 gives ~27 s of coda, M4 ~4.5 min, M6.2 ~57 min.
pub fn post_event_sec(mag: f64, opts: &CalibrationConfig) -> f64 {
    let coda = 10f64.powf((mag + 0.87) / 2.0);
    opts.min_post_sec.max(opts.coda_factor * coda)
}

/// Merged, sorted time intervals during which a station is not quiet.
#[derive(Debug, Default)]
pub struct Exclusions {
    starts: Vec<f64>,
    ends: Vec<f64>,
}

impl Exclusions {
    pub fn new(events: &[Event], station: (f64, f64), opts: &CalibrationConfig) -> Self {
        let mut spans: Vec<(f64, f64)> = events
            .iter()
            .filter(|e| e.mag >= opts.min_mag)
            .filter(|e| haversine_km(station.0, station.1, e.lat, e.lon) <= opts.max_dist_km)
            .map(|e| (e.epoch - opts.pre_sec, e.epoch + post_event_sec(e.mag, opts)))
            .collect();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut out = Exclusions::default();
        for (s, e) in spans {
            match out.ends.last_mut() {
                Some(last) if s <= *last => *last = last.max(e),
                _ => {
                    out.starts.push(s);
                    out.ends.push(e);
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// True if `[a, b)` touches any interval. The intervals are disjoint and
    /// sorted, so only the last one starting before `b` can reach past `a`.
    pub fn overlaps(&self, a: f64, b: f64) -> bool {
        let i = self.starts.partition_point(|&s| s < b);
        i > 0 && self.ends[i - 1] > a
    }

    /// Number of intervals that start inside `[a, b]`, to check that the
    /// catalog covers the data at all.
    pub fn count_within(&self, a: f64, b: f64) -> usize {
        self.starts.partition_point(|&s| s <= b) - self.starts.partition_point(|&s| s < a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_afad_dates_as_utc() {
        // 2011 Van main shock.
        assert_eq!(parse_afad_time("23/10/2011 10:41:20"), Some(1319366480.0));
        assert_eq!(parse_afad_time("01/01/1970 00:00:00"), Some(0.0));
        assert_eq!(parse_afad_time("2011-10-23 10:41:20"), None);
    }

    #[test]
    fn exclusions_merge_and_overlap() {
        let opts = CalibrationConfig {
            pre_sec: 10.0,
            min_post_sec: 100.0,
            coda_factor: 0.0,
            ..CalibrationConfig::default()
        };
        let ev = |epoch| Event { epoch, mag: 1.0, lat: 0.0, lon: 0.0 };
        let ex = Exclusions::new(&[ev(1000.0), ev(1050.0), ev(5000.0)], (0.0, 0.0), &opts);
        assert_eq!(ex.len(), 2);
        assert!(ex.overlaps(1140.0, 1200.0));
        assert!(!ex.overlaps(1150.0, 4990.0));
        assert!(ex.overlaps(0.0, 991.0));
        assert!(!ex.overlaps(0.0, 990.0));
        assert_eq!(ex.count_within(0.0, 2000.0), 1);
    }

    #[test]
    fn distant_and_small_events_are_ignored() {
        let opts = CalibrationConfig { min_mag: 2.0, max_dist_km: 100.0, ..CalibrationConfig::default() };
        let events = [
            Event { epoch: 0.0, mag: 3.0, lat: 0.0, lon: 0.0 },
            Event { epoch: 0.0, mag: 1.0, lat: 0.0, lon: 0.0 },
            Event { epoch: 0.0, mag: 5.0, lat: 10.0, lon: 0.0 },
        ];
        assert_eq!(Exclusions::new(&events, (0.0, 0.0), &opts).len(), 1);
    }
}
