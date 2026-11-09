# Gradient descent for two parameters: slope and intercept.
# Fictional points. They sit on the line y = 2 * x + 3.
# This is a teaching illustration, not a real trained model.
#
# predicted = slope * x + intercept

x = [1, 2, 3]
y = [5, 7, 9]

# The lesson starts here. Slope 3 is too high. Intercept 3 is already right.
slope = 3.0
intercept = 3.0
learning_rate = 0.01

# range(1, 2001) counts the steps 1 through 2000.
for step in range(1, 2001):
    predictions = []
    for input_x in x:
        predictions.append(slope * input_x + intercept)

    # error = predicted - actual
    errors = []
    for predicted, actual in zip(predictions, y):
        errors.append(predicted - actual)

    squared_errors = []
    for error in errors:
        squared_errors.append(error ** 2)
    mse = sum(squared_errors) / len(squared_errors)

    # error * x is the slope signal.
    # A miss that grows as x grows pulls on the slope.
    error_times_x = []
    for error, input_x in zip(errors, x):
        error_times_x.append(error * input_x)

    sum_error_times_x = sum(error_times_x)
    # sum(error) is the intercept signal.
    # A miss that is about the same at every x pulls on the intercept.
    sum_error = sum(errors)
    n = len(errors)

    # These formulas are what calculus (the derivative) gives us.
    # You do not need to derive them.
    slope_gradient = (2 / n) * sum_error_times_x
    intercept_gradient = (2 / n) * sum_error

    if step == 1:
        # Errors are 1, 2, and 3. They grow with x, so the slope comes down.
        # They are also all positive, so the intercept is nudged down too,
        # even though it started at the true value 3. It comes back later.
        print("step", step, "sum(error * x)", sum_error_times_x, "sum(error)", sum_error)
        print(
            "step", step,
            "slope", format(slope, ".4f"),
            "intercept", format(intercept, ".4f"),
            "mse", format(mse, ".6f"),
        )

    slope = slope - learning_rate * slope_gradient
    intercept = intercept - learning_rate * intercept_gradient

    # step % 200 == 0 on steps 200, 400, 600, and so on.
    # Measure the loss again so the printed MSE matches the new slope and intercept.
    if step % 200 == 0:
        squared_errors = []
        for input_x, actual in zip(x, y):
            predicted = slope * input_x + intercept
            squared_errors.append((predicted - actual) ** 2)
        mse = sum(squared_errors) / len(squared_errors)
        print(
            "step", step,
            "slope", format(slope, ".4f"),
            "intercept", format(intercept, ".4f"),
            "mse", format(mse, ".6f"),
        )

# x = 4 is a new input. On the true line, 2 * 4 + 3 = 11.
prediction = slope * 4 + intercept
print("prediction for x=4", format(prediction, ".4f"))
