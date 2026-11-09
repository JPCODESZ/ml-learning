# Derive a line from two fictional training pairs: (miles, price).
# This two-point arithmetic illustrates fitting a line.
# It is not robust real-world training, and it is not real pricing advice.
#
# With noisy data and more than two points, ordinary least squares
# picks the line that minimizes the sum of squared errors.
# The absolute error below evaluates one held-out pair.
# MAE is the evaluation here, not the fitting objective.
#
# The held-out actual price is not used to choose the slope or intercept.
# Test-data leakage lets a test answer influence training or tuning,
# which invalidates an independent evaluation.
# Look-ahead bias uses information unavailable at prediction time.
# Those are distinct terms for the same principle.

miles_a = 20000
price_a = 24000
miles_b = 40000
price_b = 20000

# Slope is the change in price divided by the change in miles.
slope = (price_b - price_a) / (miles_b - miles_a)

# price = intercept + slope * miles
intercept = price_a - slope * miles_a


def predict_price(miles):
    return intercept + slope * miles


# 50000 miles is known at prediction time.
predicted = predict_price(50000)
# 19500 is the held-out answer. It is not used to choose the line.
held_out_actual = 19500
absolute_error = abs(predicted - held_out_actual)

print(slope)
print(intercept)
print(predicted)
print(absolute_error)
