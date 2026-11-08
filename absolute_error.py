# Absolute error and mean absolute error (MAE).
# The prices below are fictional practice numbers.
# This is not a trained model and not real pricing advice.

predictions = [22000, 18000]
actual_prices = [20000, 21000]
errors = []

# zip pairs the lists: first with first, then second with second.
for predicted, actual in zip(predictions, actual_prices):
    # abs drops the minus sign. append adds that error to the list.
    errors.append(abs(predicted - actual))

# sum adds the errors. len counts them. MAE is their average.
mae = sum(errors) / len(errors)

print(errors)
print(mae)
