import os
import json
import numpy as np
import pandas as pd

import joblib
from azureml.core.model import Model


def init():
    
    global model

    try: 
        model_path = Model.get_model_path('./outputs/model.pkl')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error: ' + str(err))


def run(raw_data, columns):
    
    data = np.array(json.loads(raw_data)["data"])
    test_df = pd.DataFrame(data=data, columns=columns)
    
    out = bst.predict(test_df)
    return out.tolist()