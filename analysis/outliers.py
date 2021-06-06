from typing import Tuple

import numpy as np
import pandas as pd
from stats_tests import mahalanobis_test
from tqdm import tqdm
from utils import get_covariance_matrix


def mahalanobis_outlier_test(
    X: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO"""

    assert X.ndim == 2, "X should be matrix."

    n, m = X.shape

    distance_vector = np.zeros(n)
    t2_statistic_vector = np.zeros(n)
    f_statistic_vector = np.zeros(n)
    p_value_vector = np.zeros(n)

    x_mean = X.mean(axis=0)
    cov_inv = np.linalg.inv(get_covariance_matrix(X))

    iterator = range(n)
    if verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        distance, t2_statistic, f_statistic, p_value = mahalanobis_test(
            u=X.iloc[i],
            v=x_mean,
            VI=cov_inv,
            n_u=1,
            n_v=n,
        )

        distance_vector[i] = distance
        t2_statistic_vector[i] = t2_statistic
        f_statistic_vector[i] = f_statistic
        p_value_vector[i] = p_value

    return distance_vector, t2_statistic_vector, f_statistic_vector, p_value_vector


# TODO: REFACTOR! below (add docstrings and type annotation)


def Smirnov_Grubbs(data, kind="min"):
    if kind == "min":
        T = abs(min(data) - data.mean())
    if kind == "max":
        T = abs(max(data) - data.mean())
    return T


def Grubbs(data, kind="min"):
    sorted_data = np.sort(data)

    if kind == "min":
        reduced_data = sorted_data[1:]
    if kind == "max":
        reduced_data = sorted_data[:-1]

    G = reduced_data.std() / sorted_data.std()

    return G


def T_M(data, k=1, kind="min"):
    sorted_data = np.sort(data)

    if kind == "min":
        reduced_data = sorted_data[k:]
    if kind == "max":
        reduced_data = sorted_data[:-k]

    L = reduced_data.std() / sorted_data.std()

    return L


def E_T_M(data, k=1):
    abs_sorted_data = np.sort(abs(data - data.mean()))

    abs_reduced_data = abs_sorted_data[:-k]

    E = abs_reduced_data.std() / abs_sorted_data.std()

    return E


def Poincare(data, e):
    table = {
        0.001: 0.004,
        0.002: 0.008,
        0.005: 0.015,
        0.01: 0.026,
        0.02: 0.043,
        0.05: 0.081,
        0.1: 0.127,
        0.15: 0.164,
        0.2: 0.194,
        0.25: 0.222,
        0.3: 0.247,
        0.4: 0.291,
        0.5: 0.332,
        0.65: 0.386,
        0.8: 0.436,
        1: 0.5,
    }

    n = len(data)
    k = int(table[e] * n)
    T = sum(data) / float(n - 2 * k)

    return T


def Windsor(data, e):
    table = {
        0.001: 0.004,
        0.002: 0.008,
        0.005: 0.015,
        0.01: 0.026,
        0.02: 0.043,
        0.05: 0.081,
        0.1: 0.127,
        0.15: 0.164,
        0.2: 0.194,
        0.25: 0.222,
        0.3: 0.247,
        0.4: 0.291,
        0.5: 0.332,
        0.65: 0.386,
        0.8: 0.436,
        1: 0.5,
    }

    n = len(data)
    k = int(table[e] * n)
    W = (sum(data[k + 2 : n - k]) + k * data[k] + k * data[n - k + 1]) / n  # noqa

    return W


def Huber(data, e):
    table = {
        0.001: 2.63,
        0.002: 2.435,
        0.005: 2.16,
        0.01: 1.945,
        0.02: 1.717,
        0.05: 1.399,
        0.1: 1.14,
        0.15: 0.98,
        0.2: 0.862,
        0.25: 0.766,
        0.3: 0.685,
        0.4: 0.55,
        0.5: 0.436,
        0.65: 0.291,
        0.8: 0.162,
        1: 0.0,
    }

    data = np.array(data)
    n = len(data)
    teta = data.mean()

    while True:
        sigma = data.std()
        D = table[e]

        n_plus = (data > teta + D * sigma).sum()
        n_minus = (data < teta - D * sigma).sum()

        teta = (
            (((data > teta - D * sigma) | (data < teta + D * sigma)) * data).sum()
            + (n_plus - n_minus) * D * sigma
        ) / float(n)

    # return W  # ???
