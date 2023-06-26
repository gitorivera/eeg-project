def mse(predictions, target):
    if len(predictions) != len(target):
        raise ValueError("The length of predictions and target must be the same.")
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, target)]
    mse = sum(squared_errors) / len(predictions)
    return mse