"""Clean a client table and write down every change."""

import pandas as pd

from jobkit.detect import column_kinds, id_columns, leakage_suspects


class ChangeLog:
    def __init__(self):
        self.entries = []

    def add(self, action, detail):
        self.entries.append({"action": action, "detail": detail})

    def as_dicts(self):
        return list(self.entries)


def _positive_labels(series):
    text = series.astype(str).str.strip().str.lower()
    known = {"yes", "y", "true", "1", "no", "n", "false", "0"}
    if set(text.dropna().unique()).issubset(known):
        return text.isin({"yes", "y", "true", "1"}).astype(int), "yes"
    numeric = pd.to_numeric(series, errors="coerce")
    values = set(numeric.dropna().unique())
    if values.issubset({0, 1}):
        return numeric.fillna(0).astype(int), "1"
    minority = numeric.value_counts().idxmin()
    return (numeric == minority).astype(int), str(minority)


def load_table(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def prepare(frame, task, target, date_column=None):
    """Return a modeling table, the columns to keep, and a log of changes."""
    log = ChangeLog()
    if target not in frame.columns:
        raise SystemExit(f"Target column {target!r} is not in the file.")
    working = frame.copy()
    kinds = column_kinds(working)
    log.add("detected columns", ", ".join(f"{name} ({kind})" for name, kind in kinds.items()))

    before = len(working)
    working = working.drop_duplicates()
    removed = before - len(working)
    if removed:
        log.add("dropped duplicate rows", f"{removed} exact copies removed")
    else:
        log.add("checked duplicates", "no exact duplicate rows")

    if task == "forecast":
        if not date_column:
            date_candidates = [name for name, kind in kinds.items() if kind == "date" and name != target]
            if len(date_candidates) == 1:
                date_column = date_candidates[0]
                log.add("guessed the date column", date_column)
            else:
                raise SystemExit("A forecast needs --date, or exactly one date column.")
        parsed = pd.to_datetime(working[date_column], errors="coerce", format="mixed")
        bad_dates = int(parsed.isna().sum())
        working = working.loc[parsed.notna()].copy()
        working[date_column] = parsed.loc[parsed.notna()]
        if bad_dates:
            log.add("dropped bad dates", f"{bad_dates} rows had a date that could not be read")
        else:
            log.add("checked dates", "every date could be read")
        working[target] = pd.to_numeric(working[target], errors="coerce")
        missing_target = int(working[target].isna().sum())
        working = working.dropna(subset=[target])
        if missing_target:
            log.add("dropped blank target values", f"{missing_target} rows")
    else:
        missing_target = int(working[target].isna().sum())
        working = working.dropna(subset=[target])
        if missing_target:
            log.add("dropped blank target values", f"{missing_target} rows")
        encoded, positive_name = _positive_labels(working[target])
        working[target] = encoded
        log.add("encoded the target", f"the event being scored is {positive_name!r}")
        rate = float(working[target].mean())
        log.add(
            "class balance",
            f"{rate:.1%} of rows are the event" + (" — the classes are uneven" if rate < 0.2 or rate > 0.8 else ""),
        )

    kinds = column_kinds(working)
    identifiers = id_columns(working, kinds, target)
    if identifiers:
        log.add("held out id columns", ", ".join(identifiers))
    suspects, reasons = leakage_suspects(working, kinds, target, task)
    for name, reason in reasons.items():
        log.add("held out a possible leakage column", f"{name}: {reason}")
    if not suspects:
        log.add("checked leakage", "no column was an obvious copy of the answer")
    if "day_of_week" in working.columns:
        day_numbers = pd.to_numeric(working["day_of_week"], errors="coerce")
        if day_numbers.notna().any() and float(day_numbers.max()) > 7:
            log.add(
                "suspicious column name",
                "day_of_week has values above 7, so it may be the day of the month. The kit left the name unchanged.",
            )

    feature_columns = [
        name
        for name in working.columns
        if name not in set(identifiers + suspects + [target, date_column])
    ]
    for name in feature_columns:
        if kinds.get(name) == "number":
            numeric = pd.to_numeric(working[name], errors="coerce")
            blanks = int(numeric.isna().sum())
            if blanks and task == "classify":
                fill = float(numeric.median())
                working[name] = numeric.fillna(fill)
                log.add("filled missing numbers", f"{name}: {blanks} blanks set to the median {fill:.4g}")
            else:
                working[name] = numeric
        elif kinds.get(name) == "category":
            blanks = int(working[name].isna().sum())
            working[name] = working[name].astype("object").where(working[name].notna(), "unknown")
            if blanks:
                log.add("filled missing categories", f"{name}: {blanks} blanks set to 'unknown'")

    meta = {
        "kinds": kinds,
        "id_columns": identifiers,
        "leakage_suspects": suspects,
        "leakage_reasons": reasons,
        "feature_columns": feature_columns,
        "date_column": date_column,
        "changes": log.as_dicts(),
    }
    return working, meta
