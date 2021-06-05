from typing import Tuple

import numpy as np
import scipy.stats as sts


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
