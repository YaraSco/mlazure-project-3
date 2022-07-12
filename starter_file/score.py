import os
import json
import joblib
import numpy as np
import pandas as pd

from azureml.core.model import Model

def init():
    global model
    try:
        model_path = Model.get_model_path('hyperdrive-lgbm')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error : ', err)


def run(input_data):
    columns = model.feature_name()
    data = np.array(json.loads(input_data)["data"])
    test_df = pd.DataFrame(data=data, columns=columns)
    forecasts = model.predict(test_df)
    return forecasts.tolist()