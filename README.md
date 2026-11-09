# ml-learning
learning ml

## Focus tracker

A phone-friendly focus check lives in [`focus-tracker/`](focus-tracker/). It records a reaction-time and digit-span score plus a few context answers. Those results stay in the phone browser. They are not stored in this repo and are not sent anywhere. After GitHub Pages is enabled (Settings > Pages > Source: GitHub Actions), the app is at https://jpcodesz.github.io/ml-learning/. A later lesson will train a neural network on the exported CSV.

## Space debris

[`space-debris/close_pass.py`](space-debris/close_pass.py) estimates how close the ISS comes to objects in a real debris cloud over the next day. It is a starter for a later model, not a collision warning. See [`space-debris/README.md`](space-debris/README.md).

## Portfolio samples

Two small-business samples live in [`fiverr-samples/`](fiverr-samples/): a monthly sales forecast on public U.S. Census clothing-store sales, and a call-list score on the UCI Bank Marketing dataset. Each folder has a script, a chart, and a plain-English report.

[`jobkit/`](jobkit/) is the reusable command for the next client file: one CSV in, a draft report and chart out. Jon still checks leakage, the metric, and the wording before anything is sent.

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

## Fourth exercise: gradient descent

`gradient_descent.py` nudges one slope so the loss gets smaller. The numbers are fictional. This is a teaching illustration, not a real trained model.

The points are `x = [1, 2, 3, 4]` and `y = [2, 4, 6, 8]`. They sit on the line `y = 2 * x`, so the true slope is 2. The prediction is `slope * x` only. There is no intercept.

The loss in this file is MSE (mean squared error). For each point, square the miss with `(predicted - actual) ** 2`, then average those squares. A smaller loss means the slope fits these points better.

Gradient descent repeats the same loop: predict, measure the loss, nudge the slope, and repeat. The gradient is a formula from calculus (the derivative). You do not need to derive it. It tells which direction lowers the loss, and how steep that direction is. Subtracting it steps toward a lower loss.

`learning_rate` is the step size. At `0.01`, twenty steps from slope `0` bring the loss down toward `0` and the slope toward `2`. A step that is too big overshoots and bounces. A tiny step still moves toward `2`, but slowly.

Run:

```bash
python3 gradient_descent.py
```

### Expected output

```text
step 1 slope 0.0 mse 30.0
step 2 slope 0.3 mse 21.675
step 3 slope 0.555 mse 15.6602
step 4 slope 0.7717 mse 11.3145
step 5 slope 0.956 mse 8.1747
step 6 slope 1.1126 mse 5.9062
step 7 slope 1.2457 mse 4.2673
step 8 slope 1.3588 mse 3.0831
step 9 slope 1.455 mse 2.2275
step 10 slope 1.5368 mse 1.6094
step 11 slope 1.6063 mse 1.1628
step 12 slope 1.6653 mse 0.8401
step 13 slope 1.7155 mse 0.607
step 14 slope 1.7582 mse 0.4385
step 15 slope 1.7945 mse 0.3168
step 16 slope 1.8253 mse 0.2289
step 17 slope 1.8515 mse 0.1654
step 18 slope 1.8738 mse 0.1195
step 19 slope 1.8927 mse 0.0863
step 20 slope 1.9088 mse 0.0624
```

### Your turn

1. At the bottom of `gradient_descent.py`, uncomment the whole experiment block where `learning_rate = 0.2`. Run the file again. After the first 20 lines, a second run starts again at slope `0`. The slope jumps past `2` and the sign flips back and forth, while the MSE grows. That step size is too big. Comment the block out again when you are done.

2. Change the first `learning_rate` from `0.01` to `0.001` and run the file again. The slope still moves toward `2`, but after 20 steps it is only about `0.5` and the loss is still large. That step size is slow. Put `0.01` back when you are finished.

## Fifth exercise: slope and intercept together

`gradient_descent_two_params.py` nudges two numbers: a slope and an intercept. The points `x = [1, 2, 3]` and `y = [5, 7, 9]` are fictional. They sit on the line `y = 2 * x + 3`. This is a teaching illustration, not a real trained model.

The prediction is `slope * x + intercept`. The run starts at slope `3` and intercept `3`, with learning rate `0.01`. The intercept already matches the true line. The slope is too high.

Each step builds predictions, then errors (`predicted - actual`), then the MSE. Two signals come from those errors:

- The same kind of miss at every input is an intercept problem. The intercept gradient is `(2 / n) * sum(error)`.
- A miss that grows as `x` grows is a slope problem. The slope gradient is `(2 / n) * sum(error * x)`.

On step 1 the errors are `1`, `2`, and `3`. They grow with `x`, so the slope is pulled down. They are also all positive, so the intercept is nudged down too, even though it started at the right value `3`. As the slope settles near `2`, the intercept climbs back to about `3`.

Run:

```bash
python3 gradient_descent_two_params.py
```

### Expected output

```text
step 1 sum(error * x) 14.0 sum(error) 6.0
step 1 slope 3.0000 intercept 3.0000 mse 4.666667
step 200 slope 2.1002 intercept 2.7722 mse 0.007442
step 400 slope 2.0619 intercept 2.8593 mse 0.002842
step 600 slope 2.0383 intercept 2.9130 mse 0.001085
step 800 slope 2.0236 intercept 2.9463 mse 0.000414
step 1000 slope 2.0146 intercept 2.9668 mse 0.000158
step 1200 slope 2.0090 intercept 2.9795 mse 0.000060
step 1400 slope 2.0056 intercept 2.9873 mse 0.000023
step 1600 slope 2.0034 intercept 2.9922 mse 0.000009
step 1800 slope 2.0021 intercept 2.9952 mse 0.000003
step 2000 slope 2.0013 intercept 2.9970 mse 0.000001
prediction for x=4 11.0023
```

Step 1 prints the raw signals from the lesson: `sum(error * x)` is `14.0` and `sum(error)` is `6.0`. The intercept column shows the dip and the return: `3.0000` at step 1, `2.7722` at step 200, then back to `2.9970` at step 2000. The slope is `2.0013` and the MSE is `0.000001`. For a new input `x = 4`, the prediction is `11.0023`, near the true value `11`.

## Sixth exercise: miles and age together

`car_price_multi.py` predicts a made-up price from two features, miles and age. One feature is not enough: two cars can have the same miles and different ages, and the age still changes the price. The rows are written by hand from `28000 - 0.2 * miles - 1000 * age`, plus a small fixed noise. This is example data, not real pricing advice.

The first 9 rows train the weights. The last 3 are the test. The split is fixed. There is no random shuffle.

Miles are numbers like `30000`. Age is a number like `4`. Gradient descent uses one learning rate for every weight. Those raw sizes do not share a step size fairly, so each feature is scaled: subtract the training mean, then divide by the training standard deviation (the spread). A typical scaled value is near `0`.

The mean and the spread come from the training rows only. The test rows are the held-out check. If they helped compute those numbers, the test would leak into the setup, and the test would no longer be an independent check.

The learning rate is `0.05` for 400 steps. The scaled weights are then turned back into a change per mile and per year.

Run:

```bash
python3 car_price_multi.py
```

### Expected output

```text
train miles mean 30000.0000 spread 12909.9445
train age mean 3.6667 spread 1.8856
step 1 w_miles 0.0000 w_age 0.0000 intercept 0.0000 mse 352773333.3333
step 100 w_miles -2533.6603 w_age -1939.3272 intercept 18343.9572 mse 16107.2982
step 200 w_miles -2539.5936 w_age -1933.3940 intercept 18344.4444 mse 16078.9365
step 300 w_miles -2539.7121 w_age -1933.2755 intercept 18344.4444 mse 16078.9253
step 400 w_miles -2539.7145 w_age -1933.2731 intercept 18344.4444 mse 16078.9253
scaled w_miles -2539.7145
scaled w_age -1933.2731
scaled intercept 18344.4444
per mile -0.1967
per year -1025.2729
original intercept 28005.5415
train mse 16078.9253
test mse 2704.5988
test miles 22000 age 4 predicted 19576.4903 actual 19500
test miles 38000 age 5 predicted 15403.6104 actual 15450
test miles 48000 age 8 predicted 10360.5373 actual 10350
```

The fit settles near the made-up rule: about `-0.1967` per mile, about `-1025.2729` per year, and an original intercept of `28005.5415`. Train MSE stays near `16078.9253` because the written-in noise keeps the rows off a perfect line. On these three test rows the predictions land close to the actual prices, and the test MSE is `2704.5988`.

### Your turn

Uncomment the experiment at the bottom of `car_price_multi.py` and run the file again. It trains again with the same learning rate, `0.05`, but without scaling. The run diverges. After the normal output, the extra lines are:

```text
unscaled step 1 w_miles 0.0000 w_age 0.0000 intercept 0.0000 mse 352773333.3333
unscaled step 2 w_miles 50216666.6667 w_age 6066.6667 intercept 1834.4444 mse 2689827883106704581197824.0000
unscaled step 3 w_miles -5356444425347776.0000 w_age -627708332185.9629 intercept -150649998739.0000 mse 30604264229616661895292190064421292212224.0000
```

By step 2 the miles weight is already `50216666.6667` and the loss has exploded. Comment the block out again when you are done looking.

## Seventh exercise: will the shopper buy?

`customer_buy_logistic.py` predicts a yes-or-no label: did a made-up shopper buy? The features are `visited_pricing_page` (0 or 1) and `minutes_on_site`. The label `bought` is 0 or 1. These 20 rows are example data, not real customers. The first 15 rows train. The last 5 are the test. The split is fixed. There is no shuffle.

Minutes are scaled with the training mean and standard deviation only. The 0/1 feature is already on a small scale, so it is left as-is. The test rows do not help compute the mean or the spread.

The model builds a weighted sum, then `sigmoid` turns that sum into a probability between 0 and 1:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Log loss scores the probability given to the true answer. A probability of `0.99` on the right answer is a tiny loss (`0.0101`). A probability of `0.6` is in the middle (`0.5108`). A probability of `0.01` means the model was almost sure and wrong, so the loss is large (`4.6052`).

Both feature weights come out positive. A positive weight pushes the probability toward buy. Visiting the pricing page raises the chance. More minutes on the site raises the chance.

On the test set the script tries two cutoffs. At `0.5` it misses the buyer whose probability is `0.4599`, so recall is `0.6667` and precision is `1.0000`. At `0.3` that buyer is caught, so recall rises to `1.0000`. The non-buyer at probability `0.4364` is now called a buy too, so precision falls to `0.7500`.

Precision is: of the rows we called buy, how many did buy. Recall is: of the real buyers, how many we caught.

Run:

```bash
python3 customer_buy_logistic.py
```

### Expected output

```text
log loss at 0.99 0.0101
log loss at 0.6 0.5108
log loss at 0.01 4.6052
train minutes mean 13.4667 spread 7.6670
step 1 log loss 0.6931
step 400 log loss 0.2912
step 800 log loss 0.2908
step 1200 log loss 0.2908
step 1600 log loss 0.2908
step 2000 log loss 0.2908
weight visited_pricing_page 2.5360
weight minutes_on_site 3.4022
intercept -0.9364
A positive weight pushes the probability toward buy.
cutoff 0.5
test visited 1 minutes 20 probability 0.9890 actual 1 guess 1
test visited 0 minutes 28 probability 0.9960 actual 1 guess 1
test visited 1 minutes 9.5 probability 0.4599 actual 1 guess 0
test visited 0 minutes 15 probability 0.4364 actual 0 guess 0
test visited 0 minutes 4 probability 0.0058 actual 0 guess 0
tp 2 fp 0 fn 1 tn 2
accuracy 0.8000 precision 1.0000 recall 0.6667
cutoff 0.3
test visited 1 minutes 20 probability 0.9890 actual 1 guess 1
test visited 0 minutes 28 probability 0.9960 actual 1 guess 1
test visited 1 minutes 9.5 probability 0.4599 actual 1 guess 1
test visited 0 minutes 15 probability 0.4364 actual 0 guess 1
test visited 0 minutes 4 probability 0.0058 actual 0 guess 0
tp 3 fp 1 fn 0 tn 1
accuracy 0.8000 precision 0.7500 recall 1.0000
```
