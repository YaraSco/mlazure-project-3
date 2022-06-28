import os
import json
import joblib
import pandas as pd

from azureml.core.model import Model


def init():
    
    global model

    try: 
        #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'AutoMLfad18826420')
        model_path = Model.get_model_path('AutoMLfad18826420')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error: ' + str(err))


def run(raw_data):
    print(raw_data)
    df_data = pd.read_json(raw_data)    
    print("--------------")
    print(df_data)
    print(model)
    forecasts, X_future = model.forecast(df_data)
    print(forecasts)
    print(X_future)
    return forecasts, X_future