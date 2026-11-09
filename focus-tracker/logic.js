// Focus score, from 0 to 100. This file has no page elements, so tests can run it.
//
// The check has two parts, each worth up to 50 points.
//
// Reaction (0 to 50):
//   There are 8 trials. Each saved time is the milliseconds from the
//   screen turning yellow to the tap.
//   A tap that comes too early is saved as 1000 ms.
//   If the yellow screen is not tapped within 2000 ms, that trial is saved as 2000 ms.
//   Line the 8 times up from fastest to slowest and take the middle
//   (the average of the 4th and 5th). That middle time is the median.
//   250 ms or faster = 50 points. 700 ms or slower = 0 points.
//   Between 250 and 700, the points fall in a straight line.
//
// Memory (0 to 50):
//   Digit span starts at 3 digits and grows by one after each correct try.
//   The test stops after 2 misses, or after a correct span of 9.
//   Max span is the longest sequence typed back correctly.
//   Span 3 = 0 points. Span 9 = 50 points.
//   Between 3 and 9, the points fall in a straight line.
//   A span below 3 is 0 points.
//
// Focus score = reaction points + memory points, rounded to the nearest
// whole number, and kept between 0 and 100.

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.FocusLogic = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const FAST_MS = 250;
  const SLOW_MS = 700;
  const EARLY_TAP_MS = 1000;
  const NO_TAP_MS = 2000;
  const REACTION_TRIALS = 8;
  const SPAN_START = 3;
  const SPAN_MAX = 9;
  const MEMORY_MISSES = 2;
  const SPAN_LOW = 3;
  const SPAN_HIGH = 9;
  const WEEKDAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
  ];
  const COLUMNS = [
    "timestamp",
    "hour",
    "weekday",
    "sleep_hours",
    "ate_last_3h",
    "caffeine_last_3h",
    "adhd_meds",
    "hours_since_meds",
    "workout_today",
    "self_rated_focus",
    "median_reaction_ms",
    "false_starts",
    "max_digit_span",
    "focus_score",
    "reaction_times_ms",
  ];

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function median(values) {
    if (!values.length) {
      throw new Error("Need at least one reaction time.");
    }
    const sorted = values.slice().sort(function (a, b) {
      return a - b;
    });
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) {
      return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function reactionPoints(timesMs) {
    const middle = median(timesMs);
    const ratio = (SLOW_MS - middle) / (SLOW_MS - FAST_MS);
    return 50 * clamp(ratio, 0, 1);
  }

  function memoryPoints(maxSpan) {
    const ratio = (maxSpan - SPAN_LOW) / (SPAN_HIGH - SPAN_LOW);
    return 50 * clamp(ratio, 0, 1);
  }

  function focusScore(timesMs, maxSpan) {
    const total = reactionPoints(timesMs) + memoryPoints(maxSpan);
    return clamp(Math.round(total), 0, 100);
  }

  function summarize(timesMs, maxSpan) {
    return {
      medianReactionMs: median(timesMs),
      reactionPoints: reactionPoints(timesMs),
      memoryPoints: memoryPoints(maxSpan),
      focusScore: focusScore(timesMs, maxSpan),
    };
  }

  function scoresByHour(entries) {
    const buckets = new Map();
    entries.forEach(function (entry) {
      if (!buckets.has(entry.hour)) {
        buckets.set(entry.hour, []);
      }
      buckets.get(entry.hour).push(entry.focusScore);
    });
    const rows = [];
    buckets.forEach(function (scores, hour) {
      const total = scores.reduce(function (sum, score) {
        return sum + score;
      }, 0);
      rows.push({
        hour: hour,
        average: Math.round((total / scores.length) * 10) / 10,
        count: scores.length,
      });
    });
    rows.sort(function (a, b) {
      return a.hour - b.hour;
    });
    return rows;
  }

  function csvEscape(value) {
    const text = value === null || value === undefined ? "" : String(value);
    if (/[",\n\r]/.test(text)) {
      return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
  }

  function entryToRow(entry) {
    return [
      entry.timestamp,
      entry.hour,
      entry.weekday,
      entry.sleepHours,
      entry.ateLast3h,
      entry.caffeineLast3h,
      entry.adhdMeds,
      entry.hoursSinceMeds === null ? "" : entry.hoursSinceMeds,
      entry.workoutToday,
      entry.selfRatedFocus === null ? "" : entry.selfRatedFocus,
      entry.medianReactionMs,
      entry.falseStarts,
      entry.maxDigitSpan,
      entry.focusScore,
      entry.reactionTimesMs.join("|"),
    ];
  }

  function entriesToCsv(entries) {
    const lines = [COLUMNS.join(",")];
    entries.forEach(function (entry) {
      lines.push(entryToRow(entry).map(csvEscape).join(","));
    });
    return lines.join("\n") + "\n";
  }

  function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;
    const src = String(text).replace(/^\uFEFF/, "");
    for (let i = 0; i < src.length; i++) {
      const char = src[i];
      if (inQuotes) {
        if (char === '"') {
          if (src[i + 1] === '"') {
            field += '"';
            i++;
          } else {
            inQuotes = false;
          }
        } else {
          field += char;
        }
      } else if (char === '"') {
        inQuotes = true;
      } else if (char === ",") {
        row.push(field);
        field = "";
      } else if (char === "\n") {
        row.push(field);
        field = "";
        if (row.some(function (cell) { return cell !== ""; })) {
          rows.push(row);
        }
        row = [];
      } else if (char !== "\r") {
        field += char;
      }
    }
    if (field.length || row.length) {
      row.push(field);
      if (row.some(function (cell) { return cell !== ""; })) {
        rows.push(row);
      }
    }
    return rows;
  }

  function requireEnum(value, allowed, label) {
    if (allowed.indexOf(value) === -1) {
      throw new Error(label + " must be " + allowed.join(", ") + ".");
    }
    return value;
  }

  function optionalNumber(value, label) {
    if (value === "") {
      return null;
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      throw new Error(label + " is not a number.");
    }
    return number;
  }

  function requiredText(value, label) {
    if (!value) {
      throw new Error(label + " is missing.");
    }
    return value;
  }

  function requiredNumber(value, label) {
    const number = optionalNumber(value, label);
    if (number === null) {
      throw new Error(label + " is missing.");
    }
    return number;
  }

  function rowToEntry(cells) {
    if (cells.length !== COLUMNS.length) {
      throw new Error("Each row needs " + COLUMNS.length + " columns.");
    }
    const value = {};
    COLUMNS.forEach(function (name, index) {
      value[name] = cells[index];
    });
    const times = value.reaction_times_ms === ""
      ? []
      : value.reaction_times_ms.split("|").map(function (part) {
        return requiredNumber(part, "reaction time");
      });
    return {
      timestamp: value.timestamp,
      hour: requiredNumber(value.hour, "hour"),
      weekday: requiredText(value.weekday, "weekday"),
      sleepHours: requiredNumber(value.sleep_hours, "sleep hours"),
      ateLast3h: requireEnum(value.ate_last_3h, ["yes", "no"], "ate"),
      caffeineLast3h: requireEnum(value.caffeine_last_3h, ["yes", "no"], "caffeine"),
      adhdMeds: requireEnum(value.adhd_meds, ["yes", "no", "na"], "ADHD meds"),
      hoursSinceMeds: optionalNumber(value.hours_since_meds, "hours since meds"),
      workoutToday: requireEnum(value.workout_today, ["yes", "no"], "workout"),
      selfRatedFocus: optionalNumber(value.self_rated_focus, "self-rated focus"),
      medianReactionMs: requiredNumber(value.median_reaction_ms, "median reaction"),
      falseStarts: requiredNumber(value.false_starts, "false starts"),
      maxDigitSpan: requiredNumber(value.max_digit_span, "digit span"),
      focusScore: requiredNumber(value.focus_score, "focus score"),
      reactionTimesMs: times,
    };
  }

  function csvToEntries(text) {
    const rows = parseCsv(text);
    if (!rows.length) {
      throw new Error("That CSV is empty.");
    }
    const header = rows[0].map(function (cell) { return cell.trim(); });
    if (header.join(",") !== COLUMNS.join(",")) {
      throw new Error("That CSV does not have the focus-tracker columns.");
    }
    return rows.slice(1).map(rowToEntry);
  }

  function mergeEntries(existing, incoming) {
    const byTime = new Map();
    existing.forEach(function (entry) {
      byTime.set(entry.timestamp, entry);
    });
    incoming.forEach(function (entry) {
      byTime.set(entry.timestamp, entry);
    });
    return Array.from(byTime.values()).sort(function (a, b) {
      if (a.timestamp === b.timestamp) {
        return 0;
      }
      return a.timestamp < b.timestamp ? 1 : -1;
    });
  }

  return {
    FAST_MS: FAST_MS,
    SLOW_MS: SLOW_MS,
    EARLY_TAP_MS: EARLY_TAP_MS,
    NO_TAP_MS: NO_TAP_MS,
    REACTION_TRIALS: REACTION_TRIALS,
    SPAN_START: SPAN_START,
    SPAN_MAX: SPAN_MAX,
    MEMORY_MISSES: MEMORY_MISSES,
    SPAN_LOW: SPAN_LOW,
    SPAN_HIGH: SPAN_HIGH,
    WEEKDAYS: WEEKDAYS,
    COLUMNS: COLUMNS,
    clamp: clamp,
    median: median,
    reactionPoints: reactionPoints,
    memoryPoints: memoryPoints,
    focusScore: focusScore,
    summarize: summarize,
    scoresByHour: scoresByHour,
    entriesToCsv: entriesToCsv,
    csvToEntries: csvToEntries,
    mergeEntries: mergeEntries,
  };
});
