# Hand-written rule. This is not a trained machine learning model.
# The numbers are invented for practice. This is not real pricing advice.
#
# predict_price(miles) = 28000 - 0.2 * miles

def predict_price(miles):
    return 28000 - 0.2 * miles


for miles in [20000, 40000, 60000, 80000]:
    price = predict_price(miles)
    print(price)
