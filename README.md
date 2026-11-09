# ml-learning
learning ml

## First exercise: a hand-written price rule

`predict_price.py` is practice for `def`, `return`, calling a function, `print`, and a `for` loop.

`predict_price` is a **hand-written rule**. It is **not** trained machine learning. Nothing in this file was learned from data. The numbers are invented for practice. This is **not** real pricing advice.

The rule is:

```text
predict_price(miles) = 28000 - 0.2 * miles
```

### How to run

Use Python 3. You do not need to install any packages.

From this folder, run:

```bash
python3 predict_price.py
```

### Expected output

```text
24000.0
20000.0
16000.0
12000.0
```

`24000.0` is the float form of `24000`. That formatting is fine.

Those four lines come from the miles list `[20000, 40000, 60000, 80000]`:

- `28000 - 0.2 * 20000 = 24000`
- `28000 - 0.2 * 40000 = 20000`
- `28000 - 0.2 * 60000 = 16000`
- `28000 - 0.2 * 80000 = 12000`

`80000` is in the list. The fourth output is `12000.0`.

## Second exercise: absolute error and MAE

`absolute_error.py` compares two fictional predictions with two fictional actual prices. The data is made up for practice.

Run:

```bash
python3 absolute_error.py
```

### Expected output

```text
[2000, 3000]
2500.0
```

The first line is `errors`: `[2000, 3000]`. The second line is the mean absolute error (MAE): `2500.0`.

The code starts with `errors = []`, then:

```python
for predicted, actual in zip(predictions, actual_prices):
    errors.append(abs(predicted - actual))

mae = sum(errors) / len(errors)
```

- `zip` walks two lists together, one pair at a time.
- `abs` turns a difference into a positive error.
- `append` adds that error to the end of `errors`.
- `sum` adds every number in the list.
- `len` counts how many numbers are in the list.

`abs(22000 - 20000)` is `2000`. `abs(18000 - 21000)` is `3000`. MAE is `sum(errors) / len(errors)`, which is `5000 / 2 = 2500.0`.

## Third exercise: derive the line from two points

`fit_line.py` derives the slope and intercept from two fictional training pairs: `(20000, 24000)` and `(40000, 20000)`. Each pair is `(miles, price)`. The numbers are made up for practice. This is not real pricing advice.

This two-point arithmetic illustrates fitting a line. It is not robust real-world training. Two points fix one straight line. With noisy data and more than two points, ordinary least squares picks the line that minimizes the sum of squared errors. The absolute error in this file evaluates one held-out pair. MAE is the evaluation here, not the fitting objective.

The slope and intercept match the hand-written rule in `predict_price.py`. Here they come from the two training pairs:

- slope = `(20000 - 24000) / (40000 - 20000) = -0.2`
- intercept = `24000 - (-0.2) * 20000 = 28000`

The held-out pair is 50000 miles and an actual price of 19500. Only the miles are used to predict. The actual price is used after the prediction, for the absolute error. The line is not tuned to that answer.

- predicted price = `28000 + (-0.2) * 50000 = 18000`
- absolute error = `abs(18000 - 19500) = 1500`

Test-data leakage lets a test answer influence training or tuning. That invalidates an independent evaluation. Look-ahead bias uses information that is unavailable at prediction time. Those are distinct terms for the same principle.

Run:

```bash
python3 fit_line.py
```

### Expected output

```text
-0.2
28000.0
18000.0
1500.0
```

The four lines are the slope, the intercept, the predicted price at 50000 miles, and the absolute error.
