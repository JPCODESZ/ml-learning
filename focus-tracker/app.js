(function () {
  const logic = window.FocusLogic;
  const STORAGE_KEY = "focus-tracker-entries";
  const SHOW_MS = 700;
  const GAP_MS = 250;

  const screens = {
    home: document.getElementById("screen-home"),
    context: document.getElementById("screen-context"),
    reactionIntro: document.getElementById("screen-reaction-intro"),
    reaction: document.getElementById("screen-reaction"),
    memory: document.getElementById("screen-memory"),
    result: document.getElementById("screen-result"),
    history: document.getElementById("screen-history"),
  };

  const statusLine = document.getElementById("status");
  const backButton = document.getElementById("back-button");
  const form = document.getElementById("context-form");
  const formError = document.getElementById("form-error");
  const medsHoursWrap = document.getElementById("meds-hours-wrap");
  const reactionSurface = document.getElementById("reaction-surface");
  const reactionCount = document.getElementById("reaction-count");
  const reactionStatus = document.getElementById("reaction-status");
  const memoryTitle = document.getElementById("memory-title");
  const memoryDigit = document.getElementById("memory-digit");
  const memoryTyped = document.getElementById("memory-typed");
  const memoryPad = document.getElementById("memory-pad");
  const memoryProgress = document.getElementById("memory-progress");

  const choices = {
    ate: null,
    caffeine: null,
    meds: null,
    workout: null,
  };
  let selfRatedFocus = null;
  let context = null;

  const reaction = {
    times: [],
    falseStarts: 0,
    phase: "idle",
    timer: null,
    colorAt: 0,
  };

  const memory = {
    length: logic.SPAN_START,
    misses: 0,
    best: 0,
    sequence: [],
    typed: [],
    timers: [],
    accepting: false,
  };

  function show(name) {
    Object.keys(screens).forEach(function (key) {
      screens[key].hidden = key !== name;
    });
    backButton.hidden = name === "home";
    statusLine.textContent = "";
    window.scrollTo(0, 0);
  }

  function setStatus(message) {
    statusLine.textContent = message;
  }

  function loadEntries() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function saveEntries(entries) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  }

  function pad(number) {
    return String(number).padStart(2, "0");
  }

  function localTimestamp(date) {
    return (
      date.getFullYear() +
      "-" +
      pad(date.getMonth() + 1) +
      "-" +
      pad(date.getDate()) +
      "T" +
      pad(date.getHours()) +
      ":" +
      pad(date.getMinutes()) +
      ":" +
      pad(date.getSeconds()) +
      "." +
      String(date.getMilliseconds()).padStart(3, "0")
    );
  }

  function hourLabel(hour) {
    if (hour === 0) {
      return "12 AM";
    }
    if (hour < 12) {
      return hour + " AM";
    }
    if (hour === 12) {
      return "12 PM";
    }
    return hour - 12 + " PM";
  }

  function clearReactionTimer() {
    if (reaction.timer !== null) {
      clearTimeout(reaction.timer);
      reaction.timer = null;
    }
  }

  function clearMemoryTimers() {
    memory.timers.forEach(function (timer) {
      clearTimeout(timer);
    });
    memory.timers = [];
  }

  function setPressed(group, value) {
    choices[group] = value;
    document.querySelectorAll('[data-choice-group="' + group + '"] button').forEach(function (button) {
      button.setAttribute("aria-pressed", button.dataset.value === value ? "true" : "false");
    });
    if (group === "meds") {
      medsHoursWrap.hidden = value !== "yes";
    }
  }

  function resetForm() {
    form.reset();
    Object.keys(choices).forEach(function (group) {
      setPressed(group, null);
      choices[group] = null;
    });
    selfRatedFocus = null;
    document.querySelectorAll("#rating-choices button").forEach(function (button) {
      button.setAttribute("aria-pressed", "false");
    });
    medsHoursWrap.hidden = true;
    formError.hidden = true;
    formError.textContent = "";
  }

  function readContext() {
    const sleepRaw = document.getElementById("sleep-hours").value.trim();
    const sleep = Number(sleepRaw);
    if (sleepRaw === "" || !Number.isFinite(sleep) || sleep < 0 || sleep > 16) {
      return "Enter hours of sleep from 0 to 16.";
    }
    if (!choices.ate) {
      return "Choose yes or no for eating.";
    }
    if (!choices.caffeine) {
      return "Choose yes or no for caffeine.";
    }
    if (!choices.meds) {
      return "Choose yes, no, or N/A for ADHD meds.";
    }
    let hoursSinceMeds = null;
    if (choices.meds === "yes") {
      const hoursRaw = document.getElementById("hours-since-meds").value.trim();
      hoursSinceMeds = Number(hoursRaw);
      if (hoursRaw === "" || !Number.isFinite(hoursSinceMeds) || hoursSinceMeds < 0 || hoursSinceMeds > 24) {
        return "Enter hours since the meds, from 0 to 24.";
      }
    }
    if (!choices.workout) {
      return "Choose yes or no for practice or workout.";
    }
    context = {
      sleepHours: sleep,
      ateLast3h: choices.ate,
      caffeineLast3h: choices.caffeine,
      adhdMeds: choices.meds,
      hoursSinceMeds: hoursSinceMeds,
      workoutToday: choices.workout,
      selfRatedFocus: selfRatedFocus,
    };
    return "";
  }

  function startReaction() {
    reaction.times = [];
    reaction.falseStarts = 0;
    reaction.phase = "idle";
    show("reactionIntro");
  }

  function beginTrial() {
    clearReactionTimer();
    if (reaction.times.length >= logic.REACTION_TRIALS) {
      startMemory();
      return;
    }
    reaction.phase = "waiting";
    reactionSurface.className = "reaction-surface";
    reactionCount.textContent = "Trial " + (reaction.times.length + 1) + " of " + logic.REACTION_TRIALS;
    reactionStatus.textContent = "Wait for yellow";
    const delay = 1000 + Math.random() * 2000;
    reaction.timer = setTimeout(showYellow, delay);
  }

  function showYellow() {
    reaction.phase = "go";
    reaction.colorAt = performance.now();
    reactionSurface.className = "reaction-surface is-go";
    reactionStatus.textContent = "Tap";
    reaction.timer = setTimeout(function () {
      finishTrial(logic.NO_TAP_MS, "Too slow");
    }, logic.NO_TAP_MS);
  }

  function finishTrial(ms, label) {
    clearReactionTimer();
    reaction.phase = "feedback";
    reaction.times.push(ms);
    reactionSurface.className = label === "Too soon" ? "reaction-surface is-early" : "reaction-surface";
    reactionStatus.textContent = label;
    reaction.timer = setTimeout(function () {
      if (reaction.times.length >= logic.REACTION_TRIALS) {
        startMemory();
      } else {
        beginTrial();
      }
    }, 700);
  }

  function onReactionPointer(event) {
    event.preventDefault();
    if (reaction.phase === "waiting") {
      reaction.falseStarts += 1;
      finishTrial(logic.EARLY_TAP_MS, "Too soon");
      return;
    }
    if (reaction.phase === "go") {
      const elapsed = Math.round(performance.now() - reaction.colorAt);
      finishTrial(elapsed, elapsed + " ms");
    }
  }

  function randomDigits(length) {
    const digits = [];
    for (let i = 0; i < length; i++) {
      digits.push(Math.floor(Math.random() * 10));
    }
    return digits;
  }

  function updateMemoryProgress() {
    memoryProgress.textContent = "Length " + memory.length + " · misses " + memory.misses;
  }

  function startMemory() {
    clearReactionTimer();
    memory.length = logic.SPAN_START;
    memory.misses = 0;
    memory.best = 0;
    memory.typed = [];
    show("memory");
    showSequence();
  }

  function showSequence() {
    clearMemoryTimers();
    memory.accepting = false;
    memory.sequence = randomDigits(memory.length);
    memory.typed = [];
    memoryPad.hidden = true;
    memoryTyped.hidden = true;
    memoryTyped.textContent = "";
    memoryTitle.textContent = "Watch";
    memoryDigit.textContent = "";
    updateMemoryProgress();
    memory.sequence.forEach(function (digit, index) {
      const showAt = index * (SHOW_MS + GAP_MS);
      memory.timers.push(setTimeout(function () {
        memoryDigit.textContent = String(digit);
      }, showAt));
      memory.timers.push(setTimeout(function () {
        memoryDigit.textContent = "";
      }, showAt + SHOW_MS));
    });
    const doneAt = memory.sequence.length * (SHOW_MS + GAP_MS);
    memory.timers.push(setTimeout(openMemoryPad, doneAt));
  }

  function openMemoryPad() {
    memory.accepting = true;
    memoryTitle.textContent = "Type them back";
    memoryDigit.textContent = "";
    memoryTyped.hidden = false;
    memoryTyped.textContent = "";
    memoryPad.hidden = false;
  }

  function renderTyped() {
    memoryTyped.textContent = memory.typed.join(" ");
  }

  function checkSequence() {
    memory.accepting = false;
    const gotIt = memory.typed.join("") === memory.sequence.join("");
    if (gotIt) {
      memory.best = Math.max(memory.best, memory.length);
      memoryTitle.textContent = "Got it";
      if (memory.length >= logic.SPAN_MAX) {
        memory.timers.push(setTimeout(finishCheck, 600));
        return;
      }
      memory.length += 1;
      memory.timers.push(setTimeout(showSequence, 600));
      return;
    }
    memory.misses += 1;
    memoryTitle.textContent = "Miss";
    updateMemoryProgress();
    if (memory.misses >= logic.MEMORY_MISSES) {
      memory.timers.push(setTimeout(finishCheck, 700));
      return;
    }
    memory.timers.push(setTimeout(showSequence, 700));
  }

  function onDigit(digit) {
    if (!memory.accepting || memory.typed.length >= memory.sequence.length) {
      return;
    }
    memory.typed.push(digit);
    renderTyped();
    if (memory.typed.length === memory.sequence.length) {
      checkSequence();
    }
  }

  function finishCheck() {
    clearMemoryTimers();
    const summary = logic.summarize(reaction.times, memory.best);
    const now = new Date();
    const entry = {
      timestamp: localTimestamp(now),
      hour: now.getHours(),
      weekday: logic.WEEKDAYS[now.getDay()],
      sleepHours: context.sleepHours,
      ateLast3h: context.ateLast3h,
      caffeineLast3h: context.caffeineLast3h,
      adhdMeds: context.adhdMeds,
      hoursSinceMeds: context.hoursSinceMeds,
      workoutToday: context.workoutToday,
      selfRatedFocus: context.selfRatedFocus,
      medianReactionMs: summary.medianReactionMs,
      falseStarts: reaction.falseStarts,
      maxDigitSpan: memory.best,
      focusScore: summary.focusScore,
      reactionTimesMs: reaction.times.slice(),
    };
    try {
      saveEntries(logic.mergeEntries(loadEntries(), [entry]));
    } catch (error) {
      setStatus("This browser blocked saving. The score is still on the screen.");
    }
    document.getElementById("result-score").textContent = String(summary.focusScore);
    const detail = document.getElementById("result-detail");
    detail.replaceChildren();
    [
      "Median reaction: " + summary.medianReactionMs + " ms",
      "False starts: " + reaction.falseStarts,
      "Digit span: " + memory.best,
      "Reaction points: " + Math.round(summary.reactionPoints),
      "Memory points: " + Math.round(summary.memoryPoints),
    ].forEach(function (line) {
      const item = document.createElement("li");
      item.textContent = line;
      detail.appendChild(item);
    });
    show("result");
  }

  function addCard(parent, lines) {
    const card = document.createElement("article");
    card.className = "history-card";
    lines.forEach(function (line) {
      const paragraph = document.createElement("p");
      paragraph.textContent = line;
      card.appendChild(paragraph);
    });
    parent.appendChild(card);
  }

  function renderHistory() {
    const entries = loadEntries();
    const chart = document.getElementById("hour-chart");
    const list = document.getElementById("history-list");
    chart.replaceChildren();
    list.replaceChildren();
    document.getElementById("delete-panel").hidden = true;
    if (!entries.length) {
      addCard(chart, ["No checks yet."]);
      addCard(list, ["Finish a check and it will show up here."]);
      return;
    }
    logic.scoresByHour(entries).forEach(function (row) {
      const wrap = document.createElement("div");
      wrap.className = "hour-row";
      const label = document.createElement("p");
      label.textContent = hourLabel(row.hour) + " · average " + row.average + " · " + row.count + (row.count === 1 ? " check" : " checks");
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = Math.max(0, Math.min(100, row.average)) + "%";
      bar.appendChild(fill);
      wrap.append(label, bar);
      chart.appendChild(wrap);
    });
    entries.forEach(function (entry) {
      addCard(list, [
        entry.weekday + " " + entry.timestamp.slice(0, 10) + " · " + hourLabel(entry.hour),
        "Score " + entry.focusScore,
        "Sleep " + entry.sleepHours + "h · caffeine " + entry.caffeineLast3h + " · meds " + entry.adhdMeds,
        "Reaction " + entry.medianReactionMs + " ms · span " + entry.maxDigitSpan,
      ]);
    });
  }

  function downloadCsv() {
    const entries = loadEntries();
    if (!entries.length) {
      setStatus("There is nothing to export yet.");
      return;
    }
    const blob = new Blob([logic.entriesToCsv(entries)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "focus-checks.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  document.getElementById("start-check").addEventListener("click", function () {
    resetForm();
    show("context");
  });

  document.getElementById("open-history").addEventListener("click", function () {
    renderHistory();
    show("history");
  });

  document.getElementById("result-history").addEventListener("click", function () {
    renderHistory();
    show("history");
  });

  document.getElementById("result-done").addEventListener("click", function () {
    show("home");
  });

  backButton.addEventListener("click", function () {
    clearReactionTimer();
    clearMemoryTimers();
    reaction.phase = "idle";
    show("home");
  });

  document.querySelectorAll("[data-choice-group]").forEach(function (group) {
    group.addEventListener("click", function (event) {
      const button = event.target.closest("button");
      if (!button) {
        return;
      }
      setPressed(group.dataset.choiceGroup, button.dataset.value);
    });
  });

  document.getElementById("rating-choices").addEventListener("click", function (event) {
    const button = event.target.closest("button");
    if (!button) {
      return;
    }
    const rating = Number(button.dataset.rating);
    selfRatedFocus = selfRatedFocus === rating ? null : rating;
    document.querySelectorAll("#rating-choices button").forEach(function (choice) {
      choice.setAttribute("aria-pressed", Number(choice.dataset.rating) === selfRatedFocus ? "true" : "false");
    });
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const message = readContext();
    if (message) {
      formError.hidden = false;
      formError.textContent = message;
      return;
    }
    formError.hidden = true;
    startReaction();
  });

  document.getElementById("begin-reaction").addEventListener("click", function () {
    show("reaction");
    beginTrial();
  });

  reactionSurface.addEventListener("pointerdown", onReactionPointer);
  reactionSurface.addEventListener("contextmenu", function (event) {
    event.preventDefault();
  });

  memoryPad.addEventListener("click", function (event) {
    const button = event.target.closest("button");
    if (!button || button.id === "memory-back") {
      return;
    }
    if (button.dataset.digit !== undefined) {
      onDigit(button.dataset.digit);
    }
  });

  document.getElementById("memory-back").addEventListener("click", function () {
    if (!memory.accepting) {
      return;
    }
    memory.typed.pop();
    renderTyped();
  });

  document.getElementById("export-csv").addEventListener("click", downloadCsv);
  document.getElementById("import-csv").addEventListener("click", function () {
    document.getElementById("import-file").click();
  });

  document.getElementById("import-file").addEventListener("change", function (event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = function () {
      try {
        const incoming = logic.csvToEntries(String(reader.result));
        const merged = logic.mergeEntries(loadEntries(), incoming);
        saveEntries(merged);
        renderHistory();
        setStatus("Imported " + incoming.length + " checks. They stay on this phone.");
      } catch (error) {
        setStatus(error.message);
      }
    };
    reader.readAsText(file);
  });

  document.getElementById("delete-all").addEventListener("click", function () {
    document.getElementById("delete-panel").hidden = false;
  });

  document.getElementById("cancel-delete").addEventListener("click", function () {
    document.getElementById("delete-panel").hidden = true;
  });

  document.getElementById("confirm-delete").addEventListener("click", function () {
    localStorage.removeItem(STORAGE_KEY);
    renderHistory();
    setStatus("Deleted every saved check on this phone.");
  });

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("sw.js").catch(function () {
        // The check still works if the offline copy cannot be saved.
      });
    });
  }

  show("home");
})();
