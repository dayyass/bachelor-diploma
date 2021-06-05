import os

import pandas as pd


def path_join(filename: str, path_to_folder: str = "../data/") -> str:
    """
    os.path.join with default path_to_data.

    :param str filename: filename.
    :param str path_to_folder: path to the folder where the filename is located.
    :return: joined path.
    :rtype: str
    """

    return os.path.join(path_to_folder, filename)


def load_train(
    filename: str = "application_train.csv", path_to_train_folder: str = "../data/"
) -> pd.DataFrame:
    """
    Load train data.

    :param str filename: train data filename.
    :param str path_to_train_folder: path to the folder where the train data is located.
    :return: train dataframe.
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
