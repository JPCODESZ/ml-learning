# Space debris close passes

This starter looks at a real question: when the International Space Station and a piece of debris are both in orbit, how close do they come in the next day, and how fast is that pass?

The eventual machine-learning model should predict which warnings turn dangerous. The useful goal is to keep recall near 100% (almost never miss a truly dangerous pass) while raising precision (fewer false alarms).

## What the script does

`close_pass.py` downloads public orbit snapshots from CelesTrak:

- the stations list, which includes the ISS
- the Cosmos 2251 debris cloud (`GROUP=cosmos-2251-debris`)

A TLE (two-line element set) is a short text snapshot of an orbit at one moment, the epoch. Propagation means using that snapshot, through the `sgp4` library, to estimate later positions. The script steps through the next 24 hours once per minute and, for each debris object, keeps the minute with the smallest distance to the ISS.

It prints the 10 closest passes: object name, time in UTC, miss distance in km, and relative speed in km/s. Distance alone is not enough. A slow pass and a very fast pass can have the same miss distance. Relative speed is how fast one object is moving compared with the other at that minute.

## Run it

From the repository root:

```bash
pip install -r space-debris/requirements.txt
python3 space-debris/close_pass.py
```

Downloads are saved in `space-debris/cache/` and reused for 12 hours. That folder is gitignored. A normal run downloads each group once. Please do not delete the cache and immediately re-run in a loop. CelesTrak is a public service.

To write somewhere else, pass a path:

```bash
python3 space-debris/close_pass.py --output space-debris/history/close_passes_YYYY-MM-DD.csv
```

With no `--output`, the script still writes `space-debris/close_passes.csv`.

## Daily history

GitHub Actions runs this once a day and commits that day's CSV under `space-debris/history/`. See [history/README.md](history/README.md). Those files are how a later model can watch the same pass change as newer orbit snapshots arrive.

## Starter features

Each CSV has one row per debris object, for its closest minute in the window:

- `run_date` and `snapshot_time_utc` (when this file was produced)
- `miss_distance_km`
- `relative_speed_km_s`
- `altitude_km` (debris height above Earth's surface at that minute)
- `tle_epoch_utc` (the debris snapshot's epoch) and `iss_tle_epoch_utc`
- `tle_age_days` (how old the debris snapshot is at that minute)

These numbers are a practice set, not a collision warning.

## Honest limits

Real collisions are rare. This CSV has no "this one hit" label. A later lesson can use a proxy label, for example "the miss distance on the final orbit update was under 1 km." A proxy is a stand-in. It is not the same thing as a confirmed collision.

A 1-minute grid can miss the true closest moment, which may fall between two steps. A later pass can zoom in around the best minute and check a finer step, such as every second.
