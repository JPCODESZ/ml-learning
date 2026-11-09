"""Guess column roles. Jon still has to confirm the guesses."""

import re

import pandas as pd

ID_NAME = re.compile(r"(^id$|_id$|^uuid$|email|phone|customer)", re.IGNORECASE)
LEAK_NAME = re.compile(
    r"(duration|leakage|leaked|future_|_future|after_outcome|churn_date|label_copy)",
    re.IGNORECASE,
)


def _looks_like_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    usable = series.notna().sum()
    if usable == 0:
        return False
    if parsed.notna().sum() / usable < 0.8:
        return False
    # Month names such as "may" parse as dates, but they are categories.
    text = series.dropna().astype(str).str.strip()
    has_digit = text.str.contains(r"\d").mean()
    if has_digit < 0.5 and parsed.dropna().nunique() <= 12:
        return False
    return True


def column_kinds(frame):
    """Return a role for every column: date, number, or category."""
    kinds = {}
    for name in frame.columns:
        series = frame[name]
        if _looks_like_dates(series):
            kinds[name] = "date"
        elif pd.api.types.is_numeric_dtype(series):
            kinds[name] = "number"
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if series.notna().sum() and numeric.notna().sum() / series.notna().sum() >= 0.9:
                kinds[name] = "number"
            else:
                kinds[name] = "category"
    return kinds


def id_columns(frame, kinds, target):
    """Columns that identify a row and should not be used as predictors."""
    found = []
    rows = max(len(frame), 1)
    for name in frame.columns:
        if name == target:
            continue
        unique = frame[name].nunique(dropna=True)
        named_like_id = bool(ID_NAME.search(str(name)))
        # A sales amount can be unique without being an id. Only non-numeric
        # columns get the "almost every value is different" test.
        almost_unique = unique / rows >= 0.98 and unique >= 20 and kinds.get(name) == "category"
        if named_like_id or almost_unique:
            found.append(name)
    return found


def leakage_suspects(frame, kinds, target, task):
    """Columns that may contain the answer. These are held out unless Jon puts them back."""
    suspects = []
    reasons = {}
    target_values = frame[target]
    for name in frame.columns:
        if name == target:
            continue
        reason = None
        if LEAK_NAME.search(str(name)):
            reason = "the column name looks like an after-the-fact result"
        elif task == "forecast" and kinds.get(name) == "number":
            numeric_target = pd.to_numeric(target_values, errors="coerce")
            numeric_column = pd.to_numeric(frame[name], errors="coerce")
            pair = pd.DataFrame({"column": numeric_column, "target": numeric_target}).dropna()
            future = pd.DataFrame({"column": numeric_column, "future": numeric_target.shift(-1)}).dropna()
            if len(pair) >= 12 and pair["column"].corr(pair["target"]) > 0.98:
                reason = "the values match the target"
            elif len(future) >= 12 and future["column"].corr(future["future"]) > 0.98:
                reason = "the values match next period's target"
        elif task == "classify" and kinds.get(name) == "number":
            numeric_column = pd.to_numeric(frame[name], errors="coerce")
            numeric_target = pd.to_numeric(target_values, errors="coerce")
            if numeric_target.dropna().nunique() > 2:
                encoded = target_values.astype(str).str.lower().isin(["yes", "y", "true", "1"])
                numeric_target = encoded.astype(float)
            both = pd.DataFrame({"column": numeric_column, "target": numeric_target}).dropna()
            if len(both) >= 30 and abs(both["column"].corr(both["target"])) >= 0.95:
                reason = "the values almost match the target"
        if reason:
            suspects.append(name)
            reasons[name] = reason
    return suspects, reasons
