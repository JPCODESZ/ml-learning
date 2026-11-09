# Two features at once: miles and age.
# These cars are made-up example data, not real prices and not real pricing advice.
# Each price is 28000 - 0.2 * miles - 1000 * age, plus a small fixed noise.
# The noises, in row order, are:
# 200, -100, 150, -50, 100, -200, 50, -150, 100, -100, 50, -50.
# Nothing here is random.

# Each row is [miles, age, price].
cars = [
    [10000, 1, 25200],
    [20000, 2, 21900],
    [30000, 3, 19150],
    [15000, 4, 20950],
    [40000, 2, 18100],
    [25000, 5, 17800],
    [50000, 6, 12050],
    [35000, 3, 17850],
    [45000, 7, 12100],
    [22000, 4, 19500],
    [38000, 5, 15450],
    [48000, 8, 10350],
]

# A fixed split. cars[:9] is the first 9 rows. cars[9:] is the last 3.
# The first 9 train the weights. The last 3 are the held-out test.
# There is no shuffle.
train = cars[:9]
test = cars[9:]


def mean(values):
    return sum(values) / len(values)


def standard_deviation(values):
    # The standard deviation is the spread around the mean.
    # Square each distance from the mean, average those squares, then take the square root.
    # ** 0.5 means square root.
    average = mean(values)
    squared_distances = []
    for value in values:
        squared_distances.append((value - average) ** 2)
    return (sum(squared_distances) / len(squared_distances)) ** 0.5


# Means and spreads come from the training rows only.
# The test rows must not help build these numbers.
# If they did, the test would leak into the setup and would no longer be an independent check.
train_miles = []
train_ages = []
for miles, age, price in train:
    train_miles.append(miles)
    train_ages.append(age)

miles_mean = mean(train_miles)
miles_spread = standard_deviation(train_miles)
age_mean = mean(train_ages)
age_spread = standard_deviation(train_ages)

print("train miles mean", format(miles_mean, ".4f"), "spread", format(miles_spread, ".4f"))
print("train age mean", format(age_mean, ".4f"), "spread", format(age_spread, ".4f"))


def scale(miles, age):
    # Subtract the training mean and divide by the training spread.
    # A typical scaled value is near 0. Miles and age can then share one learning rate.
    # Raw miles are numbers like 30000. Raw ages are numbers like 4.
    scaled_miles = (miles - miles_mean) / miles_spread
    scaled_age = (age - age_mean) / age_spread
    return scaled_miles, scaled_age


def measure(rows, w_miles, w_age, intercept, use_scaling):
    # use_scaling True: use scaled miles and age.
    # use_scaling False: use the raw numbers. That is the experiment at the bottom.
    errors = []
    miles_parts = []
    age_parts = []
    for miles, age, price in rows:
        if use_scaling:
            feature_miles, feature_age = scale(miles, age)
        else:
            feature_miles = miles
            feature_age = age
        predicted = w_miles * feature_miles + w_age * feature_age + intercept
        error = predicted - price
        errors.append(error)
        miles_parts.append(error * feature_miles)
        age_parts.append(error * feature_age)

    n = len(errors)
    squared_errors = []
    for error in errors:
        squared_errors.append(error ** 2)
    mse = sum(squared_errors) / n

    # Same derivative as the two-parameter lesson. You do not need to derive it.
    # error * feature is large when that feature lines up with the miss.
    miles_gradient = (2 / n) * sum(miles_parts)
    age_gradient = (2 / n) * sum(age_parts)
    intercept_gradient = (2 / n) * sum(errors)
    return mse, miles_gradient, age_gradient, intercept_gradient


# Start at zero. learning_rate 0.05 settles on the scaled features.
w_miles = 0.0
w_age = 0.0
intercept = 0.0
learning_rate = 0.05

# range(1, 401) counts steps 1 through 400. Print every 100 steps.
for step in range(1, 401):
    mse, miles_gradient, age_gradient, intercept_gradient = measure(
        train, w_miles, w_age, intercept, True
    )
    if step == 1:
        print(
            "step", step,
            "w_miles", format(w_miles, ".4f"),
            "w_age", format(w_age, ".4f"),
            "intercept", format(intercept, ".4f"),
            "mse", format(mse, ".4f"),
        )

    w_miles = w_miles - learning_rate * miles_gradient
    w_age = w_age - learning_rate * age_gradient
    intercept = intercept - learning_rate * intercept_gradient

    if step % 100 == 0:
        mse, miles_gradient, age_gradient, intercept_gradient = measure(
            train, w_miles, w_age, intercept, True
        )
        print(
            "step", step,
            "w_miles", format(w_miles, ".4f"),
            "w_age", format(w_age, ".4f"),
            "intercept", format(intercept, ".4f"),
            "mse", format(mse, ".4f"),
        )

# w_miles and w_age are in scaled units.
# Divide by the training spread to get the change per original mile or year.
per_mile = w_miles / miles_spread
per_year = w_age / age_spread
# The scaled intercept is the price when the scaled features are 0 (an average car).
# Shift it so it is the price at 0 miles and 0 years.
original_intercept = intercept - per_mile * miles_mean - per_year * age_mean

print("scaled w_miles", format(w_miles, ".4f"))
print("scaled w_age", format(w_age, ".4f"))
print("scaled intercept", format(intercept, ".4f"))
print("per mile", format(per_mile, ".4f"))
print("per year", format(per_year, ".4f"))
print("original intercept", format(original_intercept, ".4f"))

train_mse, miles_gradient, age_gradient, intercept_gradient = measure(
    train, w_miles, w_age, intercept, True
)
test_mse, miles_gradient, age_gradient, intercept_gradient = measure(
    test, w_miles, w_age, intercept, True
)
print("train mse", format(train_mse, ".4f"))
print("test mse", format(test_mse, ".4f"))

for miles, age, price in test:
    scaled_miles, scaled_age = scale(miles, age)
    predicted = w_miles * scaled_miles + w_age * scaled_age + intercept
    print(
        "test miles", miles,
        "age", age,
        "predicted", format(predicted, ".4f"),
        "actual", price,
    )


# Experiment: same learning rate, but do not scale.
# Uncomment the lines below and run the file again.
# The miles weight and the loss explode within a few steps. That is divergence.
# w_miles = 0.0
# w_age = 0.0
# intercept = 0.0
# for step in range(1, 4):
#     mse, miles_gradient, age_gradient, intercept_gradient = measure(
#         train, w_miles, w_age, intercept, False
#     )
#     print(
#         "unscaled step", step,
#         "w_miles", format(w_miles, ".4f"),
#         "w_age", format(w_age, ".4f"),
#         "intercept", format(intercept, ".4f"),
#         "mse", format(mse, ".4f"),
#     )
#     w_miles = w_miles - learning_rate * miles_gradient
#     w_age = w_age - learning_rate * age_gradient
#     intercept = intercept - learning_rate * intercept_gradient
