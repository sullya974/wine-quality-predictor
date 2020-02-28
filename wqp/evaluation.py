import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def compute_model_metrics(model: Pipeline, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
    """
    This function computes the performance metrics of a given model and returns them as a dictionary.
    :param model: The machine learning model, as a Scikit Learn pipeline. 
    :param x: The features, as a Pandas DataFrame.
    :param y: The response data, as a Pandas DataFrame.
    :return: A dictionary, containing 3 key-values:
        - rmse: the root mean square error
        - mae: the mean absolute error
        - r2: the r2 score
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return { 
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2
    }