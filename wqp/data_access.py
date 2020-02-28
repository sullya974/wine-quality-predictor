from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict
import pandas as pd
import logging
import warnings

def fetch_csv_data(url: str, separator: Optional[str]) -> pd.DataFrame:
    """
    This functions fetch the CSV data from a given path (or url) and returns a Pandas DataFrame.
    :param url: a string containing the address of the data (local path, url ...)
    :param separator: an optional separator to override the default separator (comma)
    :return: a Pandas Dataframe containing the loaded data
    """
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")

    # Read the wine-quality csv file from the URL
    # csv_url = \
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        return pd.read_csv(url, sep=';')
    except Exception as e:
        return logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)


def build_train_test_sets(data: pd.DataFrame, label_col: str, train_size: float) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    A function to split the data into training and test sets.

    :param data: a pandas dataframe
    :param label_col: the label column name
    :param train_size: flaot. The fraction of the whole dataset used for training.
    :return: a Dictionary of key (string) - value (tuple of pandas dataframes) containing training and test data.
    Dictionary keys:
        - train: contains (train_x, train_y)
        - test: contains (test_x, test_y
    """
    train, test = train_test_split(data)

    train_x = train.drop([label_col], axis=1)
    test_x = test.drop([label_col], axis=1)
    train_y = train[label_col]
    test_y = test[label_col]

    return {
        "train": (train_x, train_y),
        "test": (test_x, test_y)
    }

    # # Split the data into training and test sets. (0.75, 0.25) split.
    # train, test = train_test_split(data)

    # # The predicted column is "quality" which is a scalar from [3, 9]
    # train_x = train.drop(["quality"], axis=1)
    # test_x = test.drop(["quality"], axis=1)
    # train_y = train[["quality"]]
    # test_y = test[["quality"]]

