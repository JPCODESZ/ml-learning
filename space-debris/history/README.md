# Daily close-pass history

This folder collects one CSV per UTC day. The file name is `close_passes_YYYY-MM-DD.csv`.

Each file is a snapshot from that day's orbit data. Every row is one debris object and its closest minute to the ISS in the following 24 hours. The same catalog number shows up again the next day.

That repeat is the point. A predicted miss distance can move when a newer TLE arrives. `tle_epoch_utc` is the time of the debris snapshot used for that row. `iss_tle_epoch_utc` is the ISS snapshot used for the whole file. `run_date` and `snapshot_time_utc` say when this file was made.

Later, the model can learn from how the miss distance for the same pass trends across these daily updates. A pass that stays close as the snapshots get newer is different from a pass that only looked close on an old snapshot.

These files are practice data, not collision warnings. The raw TLE downloads are cached locally and are not stored here.
