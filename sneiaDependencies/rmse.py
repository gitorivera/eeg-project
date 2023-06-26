import math

def rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("The length of predictions and targets must be the same.")
    
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
    mse = sum(squared_errors) / len(predictions)
    rmse = math.sqrt(mse)
    return rmse