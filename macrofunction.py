from tensorflow.keras.models import load_model
from sneiaTools import Metrics


loadModel = load_model('model.h5')
"""
Function that allows me based on a pre-trained machine learning model, a test dataset 
and a metric of the user's choice to return that metric applied to the model that ran on 
the delivered dataset.
We will use an assignment list where each metric will be assigned to an integer.
["mae", 0, "mse", 1, "rmse", 2, "re", 3]
"""
def metricsSelector(model, dataset, metric):
    if metric == 0:
        pass
    elif metric == 1:
        pass
    elif metric == 2:
        pass
    elif metric == 3:
        pass