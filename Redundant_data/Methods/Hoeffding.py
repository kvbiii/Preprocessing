import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


class Hoeffding:
    def __init__(self):
        """Initialize the Hoeffding independence test."""
        pass

    def check_X(
        self, X: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Check if X is valid and convert it to numpy array.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.

        Returns:
            np.ndarray: Converted input data.

        Raises:
            TypeError: If X is not a valid data type.
        """
        if (
            not isinstance(X, pd.DataFrame)
            and not isinstance(X, pd.Series)
            and not isinstance(X, np.ndarray)
            and not torch.is_tensor(X)
        ):
            raise TypeError(
                "Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor."
            )
        X = np.array(X)
        if X.ndim == 2:
            X = X.squeeze()
        return X

    def check_y(
        self, y: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Check if y is valid and convert it to numpy array.

        Args:
            y (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Target data.

        Returns:
            np.ndarray: Converted target data.

        Raises:
            TypeError: If y is not a valid data type.
        """
        if (
            not isinstance(y, pd.DataFrame)
            and not isinstance(y, pd.Series)
            and not isinstance(y, np.ndarray)
            and not torch.is_tensor(y)
        ):
            raise TypeError(
                "Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor."
            )
        y = np.array(y)
        if y.ndim == 2:
            y = y.squeeze()
        return y

    def fit(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        y: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        alpha: float = 0.05,
    ) -> "Hoeffding":
        """
        Perform Hoeffding independence test.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.
            y (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Target data.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            Hoeffding: Fitted instance of the class.
        """
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        ranked_X = rankdata(X)
        ranked_y = rankdata(y)
        ranked_bivariate = self.rankdata_bivariate(data1=X, data2=y)
        D1 = np.sum((ranked_bivariate - 1) * (ranked_bivariate - 2))
        D2 = np.sum((ranked_X - 1) * (ranked_X - 2) * (ranked_y - 1) * (ranked_y - 2))
        D3 = np.sum((ranked_X - 2) * (ranked_y - 2) * (ranked_bivariate - 1))
        self.hoeffding_distance_ = self.calculate_hoeffding_distance(
            D1=D1, D2=D2, D3=D3, N=X.shape[0]
        )
        self.test_statistic_ = self.calculate_test_statistic(
            D=self.hoeffding_distance_, N=X.shape[0]
        )
        self.p_value_ = self.calculate_p_value_hoeffding(N=X.shape[0], alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
        return self

    def rankdata_bivariate(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """
        Calculate bivariate ranks for Hoeffding test.

        Args:
            data1 (np.ndarray): First dataset.
            data2 (np.ndarray): Second dataset.

        Returns:
            np.ndarray: Bivariate ranks.
        """
        i = 0
        ranks = [0.75 for i in range(0, len(data1))]
        while i < len(data1):
            j = 0
            while j < len(data1):
                ranks[i] += self.check_conditions(
                    u=data1[i] - data1[j]
                ) * self.check_conditions(u=data2[i] - data2[j])
                j += 1
            i = i + 1
        return np.array(ranks)

    def check_conditions(self, u: float) -> float:
        """
        Check conditions for ranking calculation.

        Args:
            u (float): Difference value.

        Returns:
            float: Condition result (0, 0.5, or 1).
        """
        if u > 0:
            return 1
        elif u == 0:
            return 1 / 2
        else:
            return 0

    def calculate_hoeffding_distance(
        self, D1: float, D2: float, D3: float, N: int
    ) -> float:
        """
        Calculate Hoeffding distance statistic.

        Args:
            D1 (float): First component.
            D2 (float): Second component.
            D3 (float): Third component.
            N (int): Sample size.

        Returns:
            float: Hoeffding distance.
        """
        return (
            30
            * ((N - 2) * (N - 3) * D1 + D2 - 2 * (N - 2) * D3)
            / (N * (N - 1) * (N - 2) * (N - 3) * (N - 4))
        )

    def calculate_test_statistic(self, D: float, N: int) -> float:
        """
        Calculate test statistic from Hoeffding distance.

        Args:
            D (float): Hoeffding distance.
            N (int): Sample size.

        Returns:
            float: Test statistic.
        """
        return (N - 1) * np.pi**4 / 60 * D + np.pi**4 / 72

    def calculate_p_value_hoeffding(self, N: int, alpha: float) -> float:
        """
        Calculate p-value for Hoeffding test (only for N<=10).

        Args:
            N (int): Sample size.
            alpha (float): Significance level.

        Returns:
            float: Approximate p-value.

        Note:
            This approximation is only valid for N <= 10.
        """
        return np.sqrt(
            (2 * (N**2 + 5 * N - 32)) / (9 * N * (N - 1) * (N - 3) * (N - 4) * alpha)
        )

    def statistical_inference(self, p_value: float, alpha: float) -> bool:
        """
        Perform statistical inference.

        Args:
            p_value (float): P-value.
            alpha (float): Significance level.

        Returns:
            bool: True if H0 is not rejected, False otherwise.
        """
        if p_value >= alpha:
            return True
        else:
            return False


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(10)
    y = X + 0.5 * np.random.randn(10)
    hoeff = Hoeffding()
    hoeff.fit(X=X, y=y)

    print(f"Hoeffding distance: {hoeff.hoeffding_distance_:.6f}")
    print(f"Test statistic: {hoeff.test_statistic_:.6f}")
    print(f"P-value approximation: {hoeff.p_value_:.6f}")
    print(f"Keep H0 (independence): {hoeff.keep_H0}")
