import numpy as np

def mae(predictions, targets):
    targets, predictions = np.array(targets), np.array(predictions)
    mae = np.mean(np.abs(predictions - targets))
    return mae