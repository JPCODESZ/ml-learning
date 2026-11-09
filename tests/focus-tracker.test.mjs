import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const logic = require("../focus-tracker/logic.js");

function sampleEntry(overrides) {
  const entry = {
    timestamp: "2026-11-09T15:04:05.123",
    hour: 15,
    weekday: "Monday",
    sleepHours: 7.5,
    ateLast3h: "yes",
    caffeineLast3h: "no",
    adhdMeds: "yes",
    hoursSinceMeds: 3,
    workoutToday: "no",
    selfRatedFocus: 6,
    medianReactionMs: 312.5,
    falseStarts: 1,
    maxDigitSpan: 6,
    focusScore: 50,
    reactionTimesMs: [280, 300, 340, 290, 1000, 310, 330, 360],
  };
  return Object.assign(entry, overrides);
}

test("median uses the middle of an even list, not the mean", function () {
  assert.equal(logic.median([200, 200, 200, 200, 200, 200, 200, 900]), 200);
  assert.equal(logic.median([1000, 1000, 1000, 1000, 250, 250, 250, 250]), 625);
  assert.equal(logic.median([400]), 400);
});

test("reaction points fall from 50 at 250 ms to 0 at 700 ms", function () {
  assert.equal(logic.reactionPoints([250, 250, 250, 250, 250, 250, 250, 250]), 50);
  assert.equal(logic.reactionPoints([475, 475, 475, 475, 475, 475, 475, 475]), 25);
  assert.equal(logic.reactionPoints([700, 700, 700, 700, 700, 700, 700, 700]), 0);
  assert.equal(logic.reactionPoints([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]), 0);
  const mixed = logic.reactionPoints([1000, 1000, 1000, 1000, 250, 250, 250, 250]);
  assert.ok(Math.abs(mixed - (50 / 6)) < 1e-9);
});

test("memory points fall from 0 at span 3 to 50 at span 9", function () {
  assert.equal(logic.memoryPoints(0), 0);
  assert.equal(logic.memoryPoints(3), 0);
  assert.equal(logic.memoryPoints(6), 25);
  assert.equal(logic.memoryPoints(9), 50);
  assert.equal(logic.memoryPoints(12), 50);
});

test("focus score adds the two halves and stays inside 0 to 100", function () {
  const fast = [250, 250, 250, 250, 250, 250, 250, 250];
  const slow = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000];
  assert.equal(logic.focusScore(fast, 9), 100);
  assert.equal(logic.focusScore(slow, 0), 0);
  assert.equal(logic.focusScore([475, 475, 475, 475, 475, 475, 475, 475], 6), 50);
  const summary = logic.summarize([475, 475, 475, 475, 475, 475, 475, 475], 6);
  assert.equal(summary.medianReactionMs, 475);
  assert.equal(summary.reactionPoints, 25);
  assert.equal(summary.memoryPoints, 25);
  assert.equal(summary.focusScore, 50);
});

test("csv export and import round trip, including empty optional fields", function () {
  const entries = [
    sampleEntry({}),
    sampleEntry({
      timestamp: "2026-11-09T08:15:00.000",
      hour: 8,
      weekday: "Sunday",
      adhdMeds: "na",
      hoursSinceMeds: null,
      selfRatedFocus: null,
      caffeineLast3h: "yes",
      reactionTimesMs: [200, 210, 220, 230, 240, 250, 260, 270],
      medianReactionMs: 235,
      falseStarts: 0,
      maxDigitSpan: 9,
      focusScore: 100,
    }),
  ];
  const csv = logic.entriesToCsv(entries);
  assert.equal(csv.split("\n")[0], logic.COLUMNS.join(","));
  const loaded = logic.csvToEntries(csv);
  assert.deepEqual(loaded, entries);
  assert.equal(logic.entriesToCsv(loaded), csv);
});

test("csv parser keeps a comma that was inside quotes", function () {
  const header = logic.COLUMNS.join(",");
  const row = [
    "2026-11-09T09:00:00.000",
    "9",
    '"Monday, early"',
    "8",
    "no",
    "no",
    "no",
    "",
    "yes",
    "",
    "400",
    "0",
    "5",
    "42",
    "400|400|400|400|400|400|400|400",
  ].join(",");
  const loaded = logic.csvToEntries(header + "\n" + row + "\n");
  assert.equal(loaded[0].weekday, "Monday, early");
  assert.equal(loaded[0].hoursSinceMeds, null);
  assert.equal(loaded[0].selfRatedFocus, null);
  assert.deepEqual(loaded[0].reactionTimesMs, [400, 400, 400, 400, 400, 400, 400, 400]);
});

test("import merges by timestamp and scores group by hour", function () {
  const morning = sampleEntry({ timestamp: "2026-11-09T08:00:00.000", hour: 8, focusScore: 40 });
  const later = sampleEntry({ timestamp: "2026-11-09T08:30:00.000", hour: 8, focusScore: 80 });
  const night = sampleEntry({ timestamp: "2026-11-09T21:00:00.000", hour: 21, focusScore: 60 });
  const replaced = sampleEntry({
    timestamp: morning.timestamp,
    hour: 8,
    focusScore: 50,
    sleepHours: 9,
  });
  const merged = logic.mergeEntries([morning, night], [later, replaced]);
  assert.equal(merged.length, 3);
  assert.equal(merged[0].timestamp, night.timestamp);
  const keptMorning = merged.find(function (entry) {
    return entry.timestamp === morning.timestamp;
  });
  assert.equal(keptMorning.sleepHours, 9);
  assert.equal(keptMorning.focusScore, 50);
  assert.deepEqual(logic.scoresByHour(merged), [
    { hour: 8, average: 65, count: 2 },
    { hour: 21, average: 60, count: 1 },
  ]);
});

test("a CSV with the wrong header is rejected", function () {
  assert.throws(function () {
    logic.csvToEntries("a,b\n1,2\n");
  }, /focus-tracker columns/);
});
