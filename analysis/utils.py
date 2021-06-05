import os
from typing import Union

import numpy as np
import pandas as pd


def path_join(
    filename: str,
    path_to_folder: str = "../data/",
) -> str:
    """
    os.path.join with default path_to_data.

    :param str filename: filename.
    :param str path_to_folder: path to the folder where the filename is located.
    :return: joined path.
    :rtype: str
    """

    return os.path.join(path_to_folder, filename)


def load_train_test(
    filename: str,
    path_to_train_folder: str = "../data/",
) -> pd.DataFrame:
    """
    Load train data.

    :param str filename: train/test data filename.
    :param str path_to_train_folder: path to the folder where the data is located.
    :return: train/test dataframe.
    :rtype: pd.DataFrame
    """

    df_application_train = pd.read_csv(
        path_join(
            filename=filename,
            path_to_folder=path_to_train_folder,
        ),
        index_col="SK_ID_CURR",
    )
    return df_application_train


def is_positive_semi_definite(X: Union[np.ndarray, pd.DataFrame]) -> bool:
    """
    Check if matrix is positive semi-definite.

    :param Union[np.ndarray, pd.DataFrame] X: matrix.
    :return: boolean matrix is positive semi-definite.
    :rtype: bool
    """

    return np.all(np.linalg.eigvals(X) >= 0)


def get_covariance_matrix(
    X: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Get covariance matrix for given matrix X.

    :param Union[np.ndarray, pd.DataFrame] X: matrix.
    :return: covariance matrix.
    :rtype: Union[np.ndarray, pd.DataFrame]
    """

    assert X.ndim == 2, "X should be matrix."

    n = X.shape[0]

    centered_X = X - X.mean(axis=0)
    covariance_matrix = centered_X.T @ centered_X / (n - 1)

    assert is_positive_semi_definite(
        covariance_matrix
    ), "covariance_matrix should be positive semi-definite."

    return covariance_matrix
