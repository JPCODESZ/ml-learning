# Logistic regression: predict whether a shopper bought.
# These rows are made-up example data, not real customers.
# Each row is [visited_pricing_page, minutes_on_site, bought].
# visited_pricing_page and bought are 0 or 1. Nothing here is random.

import math

customers = [
    [1, 24, 1],
    [1, 18, 1],
    [1, 14, 1],
    [1, 11, 1],
    [1, 9, 0],
    [1, 6, 0],
    [1, 3, 0],
    [0, 30, 1],
    [0, 22, 1],
    [0, 18, 0],
    [0, 13, 1],
    [0, 9, 0],
    [0, 6, 0],
    [0, 3, 0],
    [1, 16, 1],
    [1, 20, 1],
    [0, 28, 1],
    [1, 9.5, 1],
    [0, 15, 0],
    [0, 4, 0],
]

# A fixed split. The first 15 rows train. The last 5 are the held-out test.
# There is no shuffle.
train = customers[:15]
test = customers[15:]


def mean(values):
    return sum(values) / len(values)


def standard_deviation(values):
    # The spread around the mean. ** 0.5 means square root.
    average = mean(values)
    squared_distances = []
    for value in values:
        squared_distances.append((value - average) ** 2)
    return (sum(squared_distances) / len(squared_distances)) ** 0.5


def sigmoid(z):
    # sigmoid(z) = 1 / (1 + e^(-z)). It turns any number into a probability from 0 to 1.
    # When z is negative, e^(-z) is huge, so use the flipped form of the same formula.
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e_to_z = math.exp(z)
    return e_to_z / (1 + e_to_z)


def log_loss(probability, bought):
    # Log loss scores the probability given to the true answer.
    # math.log cannot take 0, so stay a tiny step away from 0 and 1.
    safe = min(max(probability, 0.000000000001), 0.999999999999)
    return -(bought * math.log(safe) + (1 - bought) * math.log(1 - safe))


# A small demo. 0.99 means almost sure and right.
# 0.01 means almost sure and wrong, so the loss is large.
print("log loss at 0.99", format(log_loss(0.99, 1), ".4f"))
print("log loss at 0.6", format(log_loss(0.6, 1), ".4f"))
print("log loss at 0.01", format(log_loss(0.01, 1), ".4f"))

# Scale minutes from the training rows only.
# visited_pricing_page is already 0 or 1, so it is not scaled.
# Using the test rows here would leak the test into the setup.
train_minutes = []
for visited, minutes, bought in train:
    train_minutes.append(minutes)

minutes_mean = mean(train_minutes)
minutes_spread = standard_deviation(train_minutes)
print("train minutes mean", format(minutes_mean, ".4f"), "spread", format(minutes_spread, ".4f"))


def scaled_minutes(minutes):
    return (minutes - minutes_mean) / minutes_spread


# Start at zero. learning_rate 0.5 settles on this made-up data.
w_visited = 0.0
w_minutes = 0.0
intercept = 0.0
learning_rate = 0.5

# range(1, 2001) counts steps 1 through 2000. Print the loss every 400 steps.
for step in range(1, 2001):
    losses = []
    visited_parts = []
    minutes_parts = []
    intercept_parts = []
    for visited, minutes, bought in train:
        feature_minutes = scaled_minutes(minutes)
        z = w_visited * visited + w_minutes * feature_minutes + intercept
        probability = sigmoid(z)
        losses.append(log_loss(probability, bought))
        # The gradient of log loss is (prediction - actual) times the feature, then averaged.
        error = probability - bought
        visited_parts.append(error * visited)
        minutes_parts.append(error * feature_minutes)
        intercept_parts.append(error)

    n = len(train)
    loss = sum(losses) / n
    # Print the loss before the nudge on step 1 and every 400 steps.
    if step == 1 or step % 400 == 0:
        print("step", step, "log loss", format(loss, ".4f"))

    w_visited = w_visited - learning_rate * (sum(visited_parts) / n)
    w_minutes = w_minutes - learning_rate * (sum(minutes_parts) / n)
    intercept = intercept - learning_rate * (sum(intercept_parts) / n)

print("weight visited_pricing_page", format(w_visited, ".4f"))
print("weight minutes_on_site", format(w_minutes, ".4f"))
print("intercept", format(intercept, ".4f"))
print("A positive weight pushes the probability toward buy.")


def scores_at_cutoff(cutoff):
    # Predict buy when the probability is at least the cutoff.
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    print("cutoff", format(cutoff, ".1f"))
    for visited, minutes, bought in test:
        feature_minutes = scaled_minutes(minutes)
        z = w_visited * visited + w_minutes * feature_minutes + intercept
        probability = sigmoid(z)
        guess = 0
        if probability >= cutoff:
            guess = 1
        print(
            "test visited", visited,
            "minutes", minutes,
            "probability", format(probability, ".4f"),
            "actual", bought,
            "guess", guess,
        )
        if guess == 1 and bought == 1:
            true_positive = true_positive + 1
        elif guess == 1 and bought == 0:
            false_positive = false_positive + 1
        elif guess == 0 and bought == 1:
            false_negative = false_negative + 1
        else:
            true_negative = true_negative + 1

    total = len(test)
    # accuracy: fraction of test rows we got right.
    # precision: of the rows we called buy, how many did buy.
    # recall: of the real buyers, how many we caught.
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print(
        "tp", true_positive,
        "fp", false_positive,
        "fn", false_negative,
        "tn", true_negative,
    )
    print(
        "accuracy", format(accuracy, ".4f"),
        "precision", format(precision, ".4f"),
        "recall", format(recall, ".4f"),
    )


# 0.5 is the usual cutoff. 0.3 says "buy" more often.
scores_at_cutoff(0.5)
scores_at_cutoff(0.3)
