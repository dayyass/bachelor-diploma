from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as sts
from scipy.spatial.distance import mahalanobis
from utils import get_covariance_matrix


def matthews_significance_test(
    matthews_coef: float,
    n_samples: int,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Test significance of Matthews's correlation coefficient.
    H0: matthews_coef == 0
    H1: matthews_coef >!=< 0 (depending on alternative)

    :param float matthews_coef: Matthews's correlation coefficient.
    :param int n_samples: number of samples used to compute matthews_coef.
    :param str alternative: defines the alternative hypothesis
            the following options are available (default is ‘two-sided’):
            - ‘two-sided’
            - ‘less’: one-sided
            - ‘greater’: one-sided
    :return: chi2_statistic, p_value.
    :rtype: Tuple[float, float]
    """

    chi2_statistic = n_samples * matthews_coef ** 2

    if alternative == "two-sided":
        p_value = 2 * (1 - sts.chi2.cdf(abs(chi2_statistic), df=1))
    elif alternative == "less":
        p_value = sts.chi2.cdf(chi2_statistic, df=1)
    elif alternative == "greater":
        p_value = 1 - sts.chi2.cdf(chi2_statistic, df=1)
    else:
        raise ValueError(
            "alternative should be one of {‘two-sided’, ‘less’, ‘greater’}"
        )

    return chi2_statistic, p_value


def pearson_spearman_significance_test(
    corr_coef: float,
    n_samples: int,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Test significance of Pearson's/Spearman's correlation coefficient.
    H0: corr_coef == 0
    H1: corr_coef >!=< 0 (depending on alternative)

    :param float corr_coef: Pearson's/Spearman's correlation coefficient.
    :param int n_samples: number of samples used to compute corr_coef.
    :param str alternative: defines the alternative hypothesis
            the following options are available (default is ‘two-sided’):
            - ‘two-sided’
            - ‘less’: one-sided
            - ‘greater’: one-sided
    :return: t_statistic, p_value.
    :rtype: Tuple[float, float]
    """

    ddof = n_samples - 2
    t_statistic = corr_coef * np.sqrt(ddof) / np.sqrt(1 - corr_coef ** 2)

    if alternative == "two-sided":
        p_value = 2 * (1 - sts.t.cdf(abs(t_statistic), df=ddof))
    elif alternative == "less":
        p_value = sts.t.cdf(t_statistic, df=ddof)
    elif alternative == "greater":
        p_value = 1 - sts.t.cdf(t_statistic, df=ddof)
    else:
        raise ValueError(
            "alternative should be one of {‘two-sided’, ‘less’, ‘greater’}"
        )

    return t_statistic, p_value


def kendall_significance_test(
    kendall_coef: float,
    n_samples: int,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Test significance of Kendall's correlation coefficient.
    H0: kendall_coef == 0
    H1: kendall_coef >!=< 0 (depending on alternative)

    :param float kendall_coef: Kendall's correlation coefficient.
    :param int n_samples: number of samples used to compute kendall_coef.
    :param str alternative: defines the alternative hypothesis
            the following options are available (default is ‘two-sided’):
            - ‘two-sided’
            - ‘less’: one-sided
            - ‘greater’: one-sided
    :return: z_statistic, p_value.
    :rtype: Tuple[float, float]
    """

    z_statistic = kendall_coef * np.sqrt(
        9 * n_samples * (n_samples - 1) / (4 * n_samples + 10)
    )

    if alternative == "two-sided":
        p_value = 2 * (1 - sts.norm.cdf(abs(z_statistic)))
    elif alternative == "less":
        p_value = sts.norm.cdf(z_statistic)
    elif alternative == "greater":
        p_value = 1 - sts.norm.cdf(z_statistic)
    else:
        raise ValueError(
            "alternative should be one of {‘two-sided’, ‘less’, ‘greater’}"
        )

    return z_statistic, p_value


def hotelling_t2_1samp_test(
    X: Union[np.ndarray, pd.DataFrame],
    mu: Union[np.ndarray, pd.Series],
) -> Tuple[float, float, float]:
    """
    Test if X.mean(axis=0) == mu.
    H0: X.mean(axis=0) == mu
    H1: X.mean(axis=0) != mu

    :param Union[np.ndarray, pd.DataFrame] X: data matrix.
    :param Union[np.ndarray, pd.Series] mu: hypothesis mean.
    :return: t2_statistic, f_statistic, p_value.
    :rtype: Tuple[float, float, float]
    """

    assert X.ndim == 2, "X should be matrix."
    assert mu.ndim == 1, "mu should be vector."

    n, m = X.shape

    assert m == len(mu), "mu length should be equal to number columns of X."

    x = X.mean(axis=0) - mu
    cov_X_mean = get_covariance_matrix(X) / n
    cov_X_mean_inv = np.linalg.inv(cov_X_mean)

    t2_statistic = x @ cov_X_mean_inv @ x
    f_statistic = (n - m) / (m * (n - 1)) * t2_statistic
    p_value = 1 - sts.f.cdf(f_statistic, dfn=m, dfd=n - m)

    return t2_statistic, f_statistic, p_value


def hotelling_t2_2samp_test(
    X: Union[np.ndarray, pd.DataFrame],
    Y: Union[np.ndarray, pd.DataFrame],
) -> Tuple[float, float, float]:
    """
    Test if X.mean(axis=0) == Y.mean(axis=0).
    H0: X.mean(axis=0) == Y.mean(axis=0)
    H1: X.mean(axis=0) != Y.mean(axis=0)

    :param Union[np.ndarray, pd.DataFrame] X: data_1 matrix.
    :param Union[np.ndarray, pd.Series] Y: data_2 matrix.
    :return: t2_statistic, f_statistic, p_value.
    :rtype: Tuple[float, float, float]
    """

    assert X.ndim == 2, "X should be matrix."
    assert Y.ndim == 2, "Y should be matrix."

    n_x, m_x = X.shape
    n_y, m_y = Y.shape

    assert m_x == m_y, "X and Y should have equal number of columns."
    m = m_x

    x = X.mean(axis=0) - Y.mean(axis=0)

    cov_X = get_covariance_matrix(X)
    cov_Y = get_covariance_matrix(Y)
    cov_mean = ((n_x - 1) * cov_X + (n_y - 1) * cov_Y) / (n_x + n_y - 2)
    cov_mean_inv = np.linalg.inv(cov_mean)

    t2_statistic = n_x * n_y / (n_x + n_y) * (x @ cov_mean_inv @ x)
    f_statistic = (n_x + n_y - m - 1) / (m * (n_x + n_y - 2)) * t2_statistic
    p_value = 1 - sts.f.cdf(f_statistic, dfn=m, dfd=n_x + n_y - m - 1)

    return t2_statistic, f_statistic, p_value


def mahalanobis_test(
    u: Union[np.ndarray, pd.Series],
    v: Union[np.ndarray, pd.Series],
    VI: Union[np.ndarray, pd.DataFrame],
    n_u: int,
    n_v: int,
) -> Tuple[float, float, float, float]:
    """
    Test u == v.
    H0: u == v.
    H1: u != v.

    :param Union[np.ndarray, pd.Series] u: u vector.
    :param Union[np.ndarray, pd.Series] v: v vector.
    :param Union[np.ndarray, pd.DataFrame] VI: the inverse of the covariance matrix.
    :param int n_u: number of samples in u.
    :param int n_v: number of samples in v.
    :return: distance, t2_statistic, f_statistic, p_value.
    :rtype: Tuple[float, float, float, float]
    """

    assert u.ndim == 1, "u should be matrix."
    assert v.ndim == 1, "v should be matrix."
    assert VI.ndim == 2, "VI should be matrix."

    m = VI.shape[0]

    distance = mahalanobis(u=u, v=v, VI=VI)
    t2_statistic = n_u * n_v / (n_u + n_v) * distance
    f_statistic = (n_u + n_v - m - 1) / (n_u + n_v - 2) * t2_statistic
    p_value = 1 - sts.f.cdf(f_statistic, dfn=m, dfd=n_u + n_v - m - 1)

    return distance, t2_statistic, f_statistic, p_value
