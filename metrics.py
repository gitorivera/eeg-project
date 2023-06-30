import numpy as np

def mae(predictions, targets):
    targets, predictions = np.array(targets), np.array(predictions)
    mae = np.mean(np.abs(predictions - targets))
    return mae

def mse(predictions, target):
    if len(predictions) != len(target):
        raise ValueError("The length of predictions and target must be the same.")
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, target)]
    mse = sum(squared_errors) / len(predictions)
    return mse

def rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("The length of predictions and targets must be the same.")
    
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
    mse = sum(squared_errors) / len(predictions)
    rmse = np.sqrt(mse)
    return rmse