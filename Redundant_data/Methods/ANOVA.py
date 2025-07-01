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

    def check_for_object_columns(self, X: np.ndarray) -> np.ndarray:
        """
        Check if X contains object columns and convert it to numeric data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Converted input data.
        """
        X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError(
                "Your data contains object or string columns. Numeric data is obligated."
            )
        return np.array(X)

    def fit(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: pd.DataFrame | pd.Series | np.ndarray,
        alpha: float = 0.05,
    ):
        """
        Perform ANOVA test.

        Args:
            X (pd.DataFrame | pd.Series | np.ndarray): Input data.
            y (pd.DataFrame | pd.Series | np.ndarray): Target data.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            self: Fitted instance of the class.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
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
                "Catergory": np.unique(X),
                "Mean of dependent variable": crosstab_means.squeeze(),
            }
        )

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
        crosstab_means = np.array(
            pd.crosstab(index=X, columns="Mean", values=y, aggfunc="mean")
        )
        crosstab_frequency = np.array(pd.crosstab(index=X, columns="Sum"))
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
            * (np.sum(y**2) - np.sum(crosstab_frequency * crosstab_means**2))
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
        if p_value >= alpha:
            return True
        else:
            return False
