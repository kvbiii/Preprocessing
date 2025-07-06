import pandas as pd
import numpy as np
from scipy.stats import f


class ANOVA:
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
        X = np.array(X).squeeze()
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
    ) -> "ANOVA":
        """
        Perform ANOVA test.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray): Input data.
            y (pd.DataFrame | pd.Series | np.ndarray): Target data.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            ANOVA: Fitted instance of the class.
        """
        X = self.check_X(X)
        y = self.check_y(y)
        crosstab_means, crosstab_frequency = self.crosstab_creation(X=X, y=y)
        Sa_2 = self.calculate_Sa_squared(
            y=y, crosstab_means=crosstab_means, crosstab_frequency=crosstab_frequency
        )
        Se_2 = self.calculate_Se_squared(
            y=y, crosstab_means=crosstab_means, crosstab_frequency=crosstab_frequency
        )
        self.test_statistic_ = Sa_2 / Se_2
        self.p_value_ = self.calculate_p_value_F_test(
            F_test=self.test_statistic_,
            dfn=crosstab_means.shape[0] - 1,
            dfd=X.shape[0] - crosstab_means.shape[0],
        )
        self.critical_value_ = self.calculate_critical_value(
            dfn=crosstab_means.shape[0] - 1,
            dfd=X.shape[0] - crosstab_means.shape[0],
            alpha=alpha,
        )
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
        self.summary_ = pd.DataFrame(
            {
                "Category": np.unique(X),
                "Mean of dependent variable": crosstab_means.squeeze(),
            }
        )
        return self

    def crosstab_creation(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create crosstab with means and frequencies.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - crosstab_means (np.ndarray): Crosstab with means.
                - crosstab_frequency (np.ndarray): Crosstab with frequencies.
        """
        X_series = pd.Series(X)
        y_series = pd.Series(y)
        crosstab_means = y_series.groupby(X_series).mean().values
        crosstab_frequency = y_series.groupby(X_series).count().values
        crosstab_means = crosstab_means.reshape(-1, 1)
        crosstab_frequency = crosstab_frequency.reshape(-1, 1)
        return crosstab_means, crosstab_frequency

    def calculate_Sa_squared(
        self, y: np.ndarray, crosstab_means: np.ndarray, crosstab_frequency: np.ndarray
    ) -> float:
        """
        Calculate Sa^2.

        Args:
            y (np.ndarray): Target data.
            crosstab_means (np.ndarray): Crosstab with means.
            crosstab_frequency (np.ndarray): Crosstab with frequencies.

        Returns:
            float: Square of Sa.
        """
        return (
            1
            / (crosstab_means.shape[0] - 1)
            * np.sum(crosstab_frequency * (crosstab_means - np.mean(y)) ** 2)
        )

    def calculate_Se_squared(
        self, y: np.ndarray, crosstab_means: np.ndarray, crosstab_frequency: np.ndarray
    ) -> float:
        """
        Calculate Se^2.

        Args:
            y (np.ndarray): Target data.
            crosstab_means (np.ndarray): Crosstab with means.
            crosstab_frequency (np.ndarray): Crosstab with frequencies.

        Returns:
            float: Square of Se.
        """
        return (
            1
            / (y.shape[0] - crosstab_means.shape[0])
            * (
                np.sum(
                    (
                        y
                        - np.repeat(
                            crosstab_means.squeeze(), crosstab_frequency.squeeze()
                        )
                    )
                    ** 2
                )
            )
        )

    def calculate_p_value_F_test(self, F_test: float, dfn: int, dfd: int) -> float:
        """
        Calculate p value for F test.

        Args:
            F_test (float): F test statistic.
            dfn (int): Degrees of freedom numerator.
            dfd (int): Degrees of freedom denominator.

        Returns:
            float: P-value of F test.
        """
        return 1 - f.cdf(F_test, dfn, dfd)

    def calculate_critical_value(self, dfn: int, dfd: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            dfn (int): Degrees of freedom numerator.
            dfd (int): Degrees of freedom denominator.
            alpha (float): Significance level.

        Returns:
            float: Critical value.
        """
        return f.isf(q=alpha, dfn=dfn, dfd=dfd)

    def statistical_inference(self, p_value: float, alpha: float) -> bool:
        """
        Perform statistical inference.

        Args:
            p_value (float): P-value.
            alpha (float): Significance level.

        Returns:
            bool: True if H0 is not rejected, False otherwise.
        """
        return p_value >= alpha


if __name__ == "__main__":
    np.random.seed(17)
    groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    values = np.array([2.3, 2.1, 2.5, 3.2, 3.0, 3.4, 1.8, 1.9, 2.0])
    anova = ANOVA()
    anova.fit(X=groups, y=values)

    print(f"Test statistic (F): {anova.test_statistic_:.4f}")
    print(f"P-value: {anova.p_value_:.4f}")
    print(f"Critical value: {anova.critical_value_:.4f}")
    print(f"Keep H0: {anova.keep_H0}")
    print("\nSummary:")
    print(anova.summary_)
