# ml-learning
learning ml

## First exercise: a hand-written price rule

`predict_price.py` is practice for `def`, `return`, calling a function, `print`, and a `for` loop.

`predict_price` is a **hand-written rule**. It is **not** trained machine learning. Nothing here was learned from data. The numbers are invented for practice. This is **not** real pricing advice. Linear regression was only introduced in words. These files do not train a model.

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
