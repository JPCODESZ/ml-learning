# Gradient descent for one slope. These numbers are fictional.
# This is a teaching illustration, not a real trained model.
# The points sit on y = 2 * x, so the true slope is 2.
# predicted = slope * x. There is no intercept.

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]


def mse(slope):
    # MSE is the loss: square each miss, then average the squares.
    # ** 2 means "squared".
    squared_errors = []
    for input_x, actual in zip(x, y):
        predicted = slope * input_x
        squared_errors.append((predicted - actual) ** 2)
    return sum(squared_errors) / len(squared_errors)


def gradient(slope):
    # This formula is what calculus (the derivative) gives us.
    # You do not need to derive it.
    # It tells which direction lowers the loss, and how steep that direction is.
    weighted_errors = []
    for input_x, actual in zip(x, y):
        predicted = slope * input_x
        weighted_errors.append((predicted - actual) * input_x)
    return (2 / len(weighted_errors)) * sum(weighted_errors)


# Start at slope 0. Then repeat: predict, measure the loss, nudge, repeat.
# range(1, 21) counts the steps 1 through 20.
# learning_rate is the step size.
slope = 0.0
learning_rate = 0.01

for step in range(1, 21):
    loss = mse(slope)
    # round shortens the printed number.
    print("step", step, "slope", round(slope, 4), "mse", round(loss, 4))
    # Subtract the gradient to step toward lower loss.
    slope = slope - learning_rate * gradient(slope)


# Experiment: a learning rate that is too big overshoots and bounces.
# Uncomment the lines below and run the file again.
# The slope jumps past 2, the sign flips, and the loss grows.
# slope = 0.0
# learning_rate = 0.2
# for step in range(1, 21):
#     loss = mse(slope)
#     print("step", step, "slope", round(slope, 4), "mse", round(loss, 4))
#     slope = slope - learning_rate * gradient(slope)
