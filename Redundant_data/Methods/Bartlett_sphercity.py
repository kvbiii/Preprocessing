import numpy as np
from scipy.stats import chi2
import typing


def bartlett_sphercity(data: typing.Union[np.ndarray]) -> tuple:
    """
    Perform Bartlett's Sphercity test.

    Args:
        data (np.ndarray): data to be tested.

    Returns:
        chi_square (float): chi square test statistic.
        p_value (float): p value.
    """
    corr_matrix = np.corrcoef(data, rowvar=False)
    N = data.shape[0]
    M = data.shape[1]
    chi_square = -(N - 1 - (2 * M - 5) / 6) * np.log(np.linalg.det(corr_matrix))
    degrees_of_freedom = (M**2 - M) / 2
    p_value = 1 - chi2.cdf(x=chi_square, df=degrees_of_freedom)
    return chi_square, p_value
