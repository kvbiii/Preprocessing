import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings


class Chi_Square_Test:
    def check_X(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """
        Check if X is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray): Input data.

        Returns:
            np.ndarray: Converted input data.
        """
        if (
            not isinstance(X, pd.DataFrame)
            and not isinstance(X, pd.Series)
            and not isinstance(X, np.ndarray)
        ):
            raise TypeError(
                "Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array."
            )
        X = np.array(X)
        if X.ndim == 1:
            X = X[None, :]
        return X

    def check_y(self, y: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """
        Check if y is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            y (pd.DataFrame | pd.Series | np.ndarray): Target data.

        Returns:
            np.ndarray: Converted target data.
        """
        if (
            not isinstance(y, pd.DataFrame)
            and not isinstance(y, pd.Series)
            and not isinstance(y, np.ndarray)
        ):
            raise TypeError(
                "Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array."
            )
        y = np.array(y)
        if y.ndim != 1:
            y = y.squeeze()
        return y

    def fit(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: pd.DataFrame | pd.Series | np.ndarray,
        alpha: float = 0.05,
    ):
        """
        Perform Chi Square test.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray): Input data.
            y (pd.DataFrame | pd.Series | np.ndarray): Target data.
            alpha (float, optional): Significance level. Defaults to 0.05.
        """
        X = self.check_X(X)
        y = self.check_y(y)
        self.crosstab_ = self.crosstab_creation(X=X, y=y)
        self.check_assumptions(crosstab=self.crosstab_)
        self.test_statistic_ = self.calculate_chi_square_statistic(
            crosstab=self.crosstab_
        )
        self.df_ = self.calculate_degrees_of_freedom(crosstab=self.crosstab_)
        self.p_value_ = self.calculate_p_value(
            test_statistic=self.test_statistic_, df=self.df_
        )
        self.critical_value_ = self.calculate_critical_value(df=self.df_, alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
        self.V_Cramera_ = self.calculate_V_Cramera(
            test_statistic=self.test_statistic_, crosstab=self.crosstab_
        )

    def crosstab_creation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Create crosstab from input data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.

        Returns:
            np.ndarray: Crosstab.
        """
        return np.array(pd.crosstab(X, y, margins=True))

    def check_assumptions(self, crosstab: np.ndarray) -> None:
        """
        Check if assumptions for Chi Square test are met.

        Args:
            crosstab (np.ndarray): Crosstab.

        Warns:
            UserWarning: If assumptions are not met.
        """
        if not all(i <= 5 for i in crosstab.flatten()):
            self.assumption_ = False
            warnings.warn("Assumptions for Chi Square test are not met.")
        else:
            self.assumption_ = True

    def calculate_chi_square_statistic(self, crosstab: np.ndarray) -> float:
        """
        Calculate Chi Square statistic.

        Args:
            crosstab (np.ndarray): Crosstab.

        Returns:
            float: Chi Square statistic.
        """
        chi_square = 0
        for row in range(0, crosstab.shape[0] - 1):
            for column in range(0, crosstab.shape[1] - 1):
                observed = crosstab[row][column]
                expected = crosstab[row][-1] * crosstab[-1][column] / crosstab[-1][-1]
                chi_square += (observed - expected) ** 2 / expected
        return chi_square

    def calculate_degrees_of_freedom(self, crosstab: np.ndarray) -> int:
        """
        Calculate degrees of freedom.

        Args:
            crosstab (np.ndarray): Crosstab.

        Returns:
            int: Degrees of freedom.
        """
        return (crosstab.shape[0] - 2) * (crosstab.shape[1] - 2)

    def calculate_p_value(self, test_statistic: float, df: int) -> float:
        """
        Calculate p-value.

        Args:
            test_statistic (float): Test statistic.
            df (int): Degrees of freedom.

        Returns:
            float: P-value.
        """
        return 1 - chi2.cdf(x=test_statistic, df=df)

    def calculate_critical_value(self, df: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            df (int): Degrees of freedom.
            alpha (float): Significance level.

        Returns:
            float: Critical value.
        """
        return chi2.isf(q=alpha, df=df)

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

    def calculate_V_Cramera(self, test_statistic: float, crosstab: np.ndarray) -> float:
        """
        Calculate V Cramer test statistic.

        Args:
            test_statistic (float): Test statistic.
            crosstab (np.ndarray): Crosstab.

        Returns:
            float: V Cramer test statistic.
        """
        return (
            test_statistic
            / (
                crosstab[-1][-1]
                * np.min([crosstab.shape[0] - 2, crosstab.shape[1] - 2])
            )
        ) ** 0.5


if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame(
        {"feature1": ["A", "B", "A", "B", "A"], "feature2": ["X", "Y", "X", "Y", "X"]}
    )
    target = pd.Series(["Yes", "No", "Yes", "No", "Yes"])

    chi_square_test = Chi_Square_Test()
    chi_square_test.fit(X=data["feature1"], y=target)

    print("Chi Square Statistic:", chi_square_test.test_statistic_)
    print("Degrees of Freedom:", chi_square_test.df_)
    print("P-value:", chi_square_test.p_value_)
    print("Critical Value:", chi_square_test.critical_value_)
    print("Keep H0:", chi_square_test.keep_H0)
    print("V Cramer:", chi_square_test.V_Cramera_)
