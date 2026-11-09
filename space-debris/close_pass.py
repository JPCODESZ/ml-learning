# Closest passes between the International Space Station and a debris cloud.
#
# A TLE (two-line element set) is a small text snapshot of an orbit.
# It is not a full history. It is "where this object was, and how it was
# moving, at one moment called the epoch."
#
# Propagation means using that snapshot to estimate where the object will
# be later. The sgp4 library does that with a standard orbit model.
# The estimate gets worse as the TLE gets older, which is why we save TLE age.
#
# Distance alone is not the whole story. Two objects can pass 5 km apart
# while barely moving relative to each other, or 5 km apart at 10 km per
# second. The relative speed says how fast that gap is changing and how
# hard a hit would be. A later model will use both.

import csv
import math
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sgp4.api import Satrec, jday

# Public orbit files from CelesTrak. stations includes the ISS.
# cosmos-2251-debris is the debris cloud from a 2009 satellite collision.
# iridium-33-debris is another real cloud if you want to switch the name below.
STATION_GROUP = "stations"
DEBRIS_GROUP = "cosmos-2251-debris"
ISS_CATALOG_NUMBER = 25544

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
OUTPUT_CSV = HERE / "close_passes.csv"
# Reuse a download for 12 hours so we do not hammer CelesTrak.
CACHE_HOURS = 12
# Look ahead one day, checking once per minute.
HOURS_AHEAD = 24
STEP_MINUTES = 1
# Rough Earth radius in kilometers. Altitude is height above this, not
# distance from the center of the Earth.
EARTH_RADIUS_KM = 6378.137


def download_group(group):
    """Return TLE text for one CelesTrak group, using a local cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{group}.tle"
    now = datetime.now(timezone.utc)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        modified = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc)
        age = now - modified
        if age < timedelta(hours=CACHE_HOURS):
            print(f"Using cached {group} data ({age.total_seconds() / 3600:.1f} hours old).")
            return cache_path.read_text(encoding="utf-8")

    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    print(f"Downloading {group} from CelesTrak...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ml-learning-space-debris-lesson/1.0 (educational; cached)"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_tles(text):
    """Turn a TLE file into a list of name, catalog number, and the two lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    satellites = []
    index = 0
    while index < len(lines):
        # CelesTrak sends three lines: name, line 1, line 2.
        if (
            index + 2 < len(lines)
            and lines[index + 1].startswith("1 ")
            and lines[index + 2].startswith("2 ")
        ):
            line1 = lines[index + 1]
            line2 = lines[index + 2]
            satellites.append(
                {
                    "name": lines[index],
                    "catalog": int(line1[2:7]),
                    "line1": line1,
                    "line2": line2,
                }
            )
            index += 3
            continue
        # Some files skip the name and only have the two lines.
        if (
            index + 1 < len(lines)
            and lines[index].startswith("1 ")
            and lines[index + 1].startswith("2 ")
        ):
            line1 = lines[index]
            satellites.append(
                {
                    "name": f"CATALOG {int(line1[2:7])}",
                    "catalog": int(line1[2:7]),
                    "line1": line1,
                    "line2": lines[index + 1],
                }
            )
            index += 2
            continue
        index += 1
    return satellites


def epoch_datetime(line1):
    """Read the TLE epoch (the moment the snapshot was taken) as a UTC time."""
    # Columns 19-20 are a two-digit year. Columns 21-32 are the day of that year.
    year = int(line1[18:20])
    year += 2000 if year < 57 else 1900
    day_of_year = float(line1[20:32])
    january_first = datetime(year, 1, 1, tzinfo=timezone.utc)
    return january_first + timedelta(days=day_of_year - 1)


def vector_subtract(left, right):
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def vector_length(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def propagate(satellite, times):
    """Estimate position (km) and velocity (km/s) at each time.

    Propagation can fail for a badly shaped orbit. Those minutes are None.
    """
    states = []
    for moment in times:
        julian_day, fraction = jday(
            moment.year,
            moment.month,
            moment.day,
            moment.hour,
            moment.minute,
            moment.second,
        )
        error, position, velocity = satellite.sgp4(julian_day, fraction)
        if error == 0:
            states.append((position, velocity))
        else:
            states.append(None)
    return states


def closest_pass(iss_states, debris_states, times, debris):
    """Find the minute in this window when the debris is nearest the ISS."""
    best_distance = None
    best = None
    epoch = epoch_datetime(debris["line1"])
    for moment, iss_state, debris_state in zip(times, iss_states, debris_states):
        if iss_state is None or debris_state is None:
            continue
        iss_position, iss_velocity = iss_state
        debris_position, debris_velocity = debris_state
        gap = vector_subtract(debris_position, iss_position)
        distance_km = vector_length(gap)
        if best_distance is None or distance_km < best_distance:
            relative_velocity = vector_subtract(debris_velocity, iss_velocity)
            # Altitude is how high the debris is above Earth's surface.
            altitude_km = vector_length(debris_position) - EARTH_RADIUS_KM
            tle_age_days = (moment - epoch).total_seconds() / 86400
            best_distance = distance_km
            best = {
                "name": debris["name"],
                "catalog": debris["catalog"],
                "time_utc": moment.strftime("%Y-%m-%d %H:%M:%S"),
                "miss_distance_km": distance_km,
                "relative_speed_km_s": vector_length(relative_velocity),
                "altitude_km": altitude_km,
                "tle_age_days": tle_age_days,
            }
    return best


def load_satellite(record):
    return Satrec.twoline2rv(record["line1"], record["line2"])


def main():
    station_text = download_group(STATION_GROUP)
    debris_text = download_group(DEBRIS_GROUP)
    stations = parse_tles(station_text)
    debris_objects = parse_tles(debris_text)

    iss_record = None
    for record in stations:
        if record["catalog"] == ISS_CATALOG_NUMBER:
            iss_record = record
            break
    if iss_record is None:
        raise SystemExit("Could not find the ISS (catalog 25544) in the stations file.")

    # Start at the current minute and step forward for 24 hours.
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    step_count = int(HOURS_AHEAD * 60 / STEP_MINUTES) + 1
    times = [start + timedelta(minutes=STEP_MINUTES * step) for step in range(step_count)]

    print(f"Propagating the ISS and {len(debris_objects)} debris objects...")
    print("This is a 1-minute grid. The true closest moment can fall between steps.")
    iss_states = propagate(load_satellite(iss_record), times)

    passes = []
    for record in debris_objects:
        if record["catalog"] == ISS_CATALOG_NUMBER:
            continue
        try:
            debris_sat = load_satellite(record)
        except ValueError:
            # A broken TLE line cannot be propagated. Skip it and keep going.
            continue
        approach = closest_pass(iss_states, propagate(debris_sat, times), times, record)
        if approach is not None:
            passes.append(approach)

    passes.sort(key=lambda item: item["miss_distance_km"])

    print()
    print("Top 10 closest ISS passes in the next 24 hours")
    print(f"Debris group: {DEBRIS_GROUP}")
    print("name | time (UTC) | miss distance (km) | relative speed (km/s)")
    for rank, approach in enumerate(passes[:10], start=1):
        print(
            f"{rank:2}. {approach['name']} ({approach['catalog']}) | "
            f"{approach['time_utc']} | "
            f"{approach['miss_distance_km']:.3f} | "
            f"{approach['relative_speed_km_s']:.3f}"
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "catalog",
                "time_utc",
                "miss_distance_km",
                "relative_speed_km_s",
                "altitude_km",
                "tle_age_days",
            ],
        )
        writer.writeheader()
        for approach in passes:
            writer.writerow(
                {
                    "name": approach["name"],
                    "catalog": approach["catalog"],
                    "time_utc": approach["time_utc"],
                    "miss_distance_km": f"{approach['miss_distance_km']:.6f}",
                    "relative_speed_km_s": f"{approach['relative_speed_km_s']:.6f}",
                    "altitude_km": f"{approach['altitude_km']:.3f}",
                    "tle_age_days": f"{approach['tle_age_days']:.6f}",
                }
            )
    print()
    print(f"Saved {len(passes)} passes to {OUTPUT_CSV.name}")
    print("Each row is the closest minute for one debris object, not a danger label.")


if __name__ == "__main__":
    main()
