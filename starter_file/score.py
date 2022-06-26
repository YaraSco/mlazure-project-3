import os
import json
import joblib
import numpy as np
import pandas as pd

from azureml.core.model import Model


def init():
    
    global model

    try: 
        model_path = Model.get_model_path('./outputs/model.pkl')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error: ' + str(err))


def run(raw_data):
    
    df_data = json.loads(raw_data)
    #test_df = pd.DataFrame(data=data, columns=columns)
    
    forecasts, X_future = model.forecast(df_data)
    return forecasts, X_future