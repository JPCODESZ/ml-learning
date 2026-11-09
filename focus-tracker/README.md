# Focus tracker

A one-minute focus check for a phone. It will later feed a neural network that tries to predict Jon's best study times. This folder is a static site: HTML, CSS, and JavaScript. There is no build step and no server.

## What it records

Each finished check saves one row on the phone:

- timestamp, hour of day, and weekday (taken from the phone clock when the check ends)
- hours of sleep last night
- ate in the last 3 hours (yes/no)
- caffeine in the last 3 hours (yes/no)
- ADHD meds today (yes/no/na) and, if yes, hours since the dose
- practice or workout today (yes/no)
- optional self-rated focus from 1 to 10
- eight reaction times in milliseconds, how many of those were early taps, the median reaction time, the longest digit span, and the 0–100 focus score

## The focus score

The score is 0 to 100. Reaction and memory each contribute up to 50 points. The same rules are written at the top of `logic.js`.

Reaction uses 8 trials. The screen turns yellow after a random wait of 1 to 3 seconds. The saved time is the milliseconds from yellow to the tap.

- A tap before the yellow screen is a false start. That trial is saved as **1000 ms**.
- If yellow stays up for 2 seconds with no tap, that trial is saved as **2000 ms**.
- The median is the middle of the 8 times (the average of the 4th and 5th after sorting).
- **250 ms or faster = 50 points. 700 ms or slower = 0 points.** Between those, the points fall in a straight line.

Memory is a digit span. It starts at 3 digits and adds one digit after each correct answer. It stops after **2 misses**, or after a correct span of 9. Max span is the longest sequence typed back correctly.

- **Span 3 = 0 points. Span 9 = 50 points.** Between those, the points fall in a straight line. A shorter span is 0.

Focus score = reaction points + memory points, rounded to the nearest whole number, and kept between 0 and 100.

## Data stays on the phone

Checks are stored in `localStorage` in this browser only. Nothing is uploaded. Do not commit a CSV or a copy of the checks into this public repo. Export CSV downloads a file to the phone. Import CSV reads a file you pick and merges it with checks already on the phone, matching rows by timestamp. Delete all asks for confirmation, then removes the saved checks from this browser.

The exported columns are the ones a later lesson can train on. That lesson will use the CSV. It is not part of this app.

## Use it

From this folder:

```bash
python3 -m http.server 8765
```

Open `http://127.0.0.1:8765/` on the phone or in a browser. The service worker, which keeps the pages for offline use, needs `http://` rather than a `file://` link.

To add it to the home screen: in the phone browser, use Add to Home Screen. The manifest and icons are in this folder. After that, it opens like a small app. Once the pages have been loaded once, the check still opens without a network.

## Published site

The GitHub Actions workflow `.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push to `main`.

The site address is:

https://jpcodesz.github.io/ml-learning/

Pages has to be turned on once by hand, because this repo cannot flip that setting by itself:

1. Open the repository on GitHub.
2. Go to **Settings > Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

After that, merging to `main` runs the workflow. If a run failed before Pages was switched to GitHub Actions, open the Actions tab and run **Publish focus tracker** again.
