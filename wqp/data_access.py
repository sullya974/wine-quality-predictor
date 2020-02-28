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

