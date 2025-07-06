import numpy as np
from scipy.stats import chi2


def bartlett_sphercity(data: np.ndarray) -> tuple[float, float]:
    """
    Perform Bartlett's Sphericity test.

    Args:
        data (np.ndarray): Data to be tested.

    Returns:
        tuple[float, float]: A tuple containing:
            - chi_square (float): Chi-square test statistic.
            - p_value (float): P-value.
    """
    corr_matrix = np.corrcoef(data, rowvar=False)
    N = data.shape[0]
    M = data.shape[1]
    chi_square = -(N - 1 - (2 * M - 5) / 6) * np.log(np.linalg.det(corr_matrix))
    degrees_of_freedom = (M**2 - M) / 2
    p_value = 1 - chi2.cdf(x=chi_square, df=degrees_of_freedom)
    return chi_square, p_value


if __name__ == "__main__":
    import pandas as pd

    np.random.seed(17)
    sample_data = np.random.randn(50, 5)
    chi_square_stat, p_val = bartlett_sphercity(sample_data)

    print(f"Chi-square statistic: {chi_square_stat:.4f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        print(
            "The correlation matrix is significantly different from an identity matrix."
        )
    else:
        print(
            "The correlation matrix is not significantly different from an identity matrix."
        )
