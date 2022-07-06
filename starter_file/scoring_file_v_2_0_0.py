# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"store": pd.Series([0], dtype="int64"), "brand": pd.Series([0], dtype="int64"), "week": pd.Series([0], dtype="int64"), "constant": pd.Series([0], dtype="int64"), "price1": pd.Series([0.0], dtype="float64"), "price2": pd.Series([0.0], dtype="float64"), "price3": pd.Series([0.0], dtype="float64"), "price4": pd.Series([0.0], dtype="float64"), "price5": pd.Series([0.0], dtype="float64"), "price6": pd.Series([0.0], dtype="float64"), "price7": pd.Series([0.0], dtype="float64"), "price8": pd.Series([0.0], dtype="float64"), "price9": pd.Series([0.0], dtype="float64"), "price10": pd.Series([0.0], dtype="float64"), "price11": pd.Series([0.0], dtype="float64"), "deal": pd.Series([0], dtype="int64"), "feat": pd.Series([0.0], dtype="float64"), "profit": pd.Series([0.0], dtype="float64"), "week_start": pd.Series(["2000-1-1"], dtype="datetime64[ns]")}), enforce_shape=False)
input_sample = StandardPythonParameterType({'data': data_sample})
quantiles_sample = StandardPythonParameterType([0.025, 0.975])
sample_global_params = StandardPythonParameterType({"quantiles": quantiles_sample})

result_sample = StandardPythonParameterType({
    'forecast': NumpyParameterType(np.array([0])),
    'prediction_interval': NumpyParameterType(np.array([0])),
    'index': PandasParameterType(pd.DataFrame({}), enforce_shape=False)
})
output_sample = StandardPythonParameterType({'Results': result_sample})
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_forecasting_v2')
except Exception:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, GlobalParameters={"quantiles":[0.025, 0.975]}):
    y_query = None
    data = Inputs['data']
    if 'y_query' in data.columns:
        y_query = data.pop('y_query').values
    
    quantiles = GlobalParameters.get("quantiles", [0.025, 0.975])
    quantiles = [min(quantiles), 0.5, max(quantiles)]
    PI = 'prediction_interval'
    model.quantiles = quantiles
    pred_quantiles = model.forecast_quantiles(data, y_query)
    pred_quantiles[PI] = pred_quantiles[[min(quantiles), max(quantiles)]].apply(lambda x: '[{}, {}]'.format(x[0], x[1]), axis=1)

    index_as_df = pred_quantiles.iloc[:,:-4].reset_index(drop=True)# get time column name and grain column name
    forecast_as_list = pred_quantiles[0.5].to_list()
    PI_as_list = pred_quantiles[PI].to_list()
    result = {
        "forecast": forecast_as_list,
        "prediction_interval": PI_as_list,
        "index": json.loads(index_as_df.to_json(orient='records'))
    }
    return {'Results': result}
