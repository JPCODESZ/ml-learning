# ml-learning
learning ml

## First exercise: a hand-written price rule

`predict_price.py` is practice for `def`, `return`, calling a function, `print`, and a `for` loop.

`predict_price` is a **hand-written rule**. It is **not** trained machine learning. Nothing here was learned from data. The numbers are invented for practice. This is **not** real pricing advice.

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
```

`24000.0` is the float form of `24000`. That formatting is fine.

Those three lines come from the miles list `[20000, 40000, 60000]`:

- `28000 - 0.2 * 20000 = 24000`
- `28000 - 0.2 * 40000 = 20000`
- `28000 - 0.2 * 60000 = 16000`

### Your turn

Open `predict_price.py`. Find the list in the `for` loop. Add `80000` so it looks like this:

```python
for miles in [20000, 40000, 60000, 80000]:
```

Before you run it, write down the new last number you expect. Then run:

```bash
python3 predict_price.py
```

Check the last line against what you wrote down. When you are done looking, you can put the list back to `[20000, 40000, 60000]`.
