import sys
import logging

from wqp.ml import build_wine_predictor_model
from wqp.evaluation import compute_model_metrics
from wqp.data_access import fetch_csv_data, build_train_test_sets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wqp.main")

def model_training_workflow(data_path: str):
    """
    This functions orchestrates the whole training script, as distinct steps:
    - fetching input data
    - splitting them into train and test datasets
    - definning the model
    - fitting the model on the training data
    - evaluating the model on the test data
    :param data_path: a string containing the location of the training data 
    """

    # - fetching input data
    data = fetch_csv_data(data_path, separator=",")

    # - splitting them into train and test datasets
    train_test_sets = build_train_test_sets(data, "quality", train_size=.8)
    (train_x, train_y) = train_test_sets["train"]
    (test_x, test_y) = train_test_sets["test"]

    # - definning the model
    model = build_wine_predictor_model()

    # - fitting the model on the training data
    model.fit(train_x, train_y)

    # - evaluating the model on the test data
    (rmse, mae, r2) = compute_model_metrics(model, test_x, test_y)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
