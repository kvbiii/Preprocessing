import pandas as pd
import numpy as np
from scipy.stats import rankdata, t
import typing


class SpearmanCorrelation:
    def check_X(
        self, X: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Check if X is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            X: (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.

        Returns:
            X: (np.ndarray): converted input data.
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
        if X.ndim != 1:
            X = X.squeeze()
        return X

    def check_y(
        self, y: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Check if y is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

        Args:
            y: (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.

        Returns:
            y: (np.ndarray): converted target data.
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
            X: (np.ndarray): input data.

        Returns:
            X: (np.ndarray): converted input data.
        """
        X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError(
                "Your data contains object or string columns. Numeric data is obligated."
            )
        return np.array(X)

    def check_sample_count(self, X1: np.ndarray, X2: np.ndarray) -> None:
        """
        Check if the number of samples in X1 and X2 is the same.

        Args:
            X1 (np.ndarray): feature 1 data.
            X2 (np.ndarray): feature 2 data.
        """
        if X1.shape[0] != X2.shape[0]:
            raise ValueError("The number of samples in X1 and X2 is not the same.")

    def fit(
        self,
        X1: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        X2: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        alpha: float = 0.05,
    ) -> None:
        """
        Perform Spearman correlation

        Args:
            X1 (Union[pd.DataFrame, pd.Series, np.ndarray]): feature 1 data.
            X2 (Union[pd.DataFrame, pd.Series, np.ndarray]): feature 2 data.
            alpha (float): significance level.
        """
        X1 = self.check_X(X1)
        X1 = self.check_for_object_columns(X1)
        X2 = self.check_X(X2)
        X2 = self.check_for_object_columns(X2)
        self.check_sample_count(X1=X1, X2=X2)
        ranked_X1 = rankdata(X1)
        ranked_X2 = rankdata(X2)
        covariance = self.calculate_covariance(X1=ranked_X1, X2=ranked_X2)
        std_X1 = self.calculate_std(data=ranked_X1)
        std_X2 = self.calculate_std(data=ranked_X2)
        self.correlation_ = self.calculate_correlation(
            covariance=covariance, std_X1=std_X1, std_X2=std_X2
        )
        self.test_statistic_ = self.calculate_test_statistic(
            correlation=self.correlation_, N=X1.shape[0]
        )
        self.p_value_ = self.calculate_p_value_t_test(
            t_test=self.test_statistic_, df=X1.shape[0] - 2
        )
        self.critical_value_ = self.calculate_critical_value(
            df=X1.shape[0] - 2, alpha=alpha
        )
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)

    def calculate_covariance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate covariance between two arrays.

        Args:
            X1 (np.ndarray): feature 1 data.
            X2 (np.ndarray): feature 2 data.

        Returns:
            np.ndarray: covariance matrix.
        """
        return np.cov(X1, X2)

    def calculate_std(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviation of the data.

        Args:
            data (np.ndarray): input data.

        Returns:
            np.ndarray: standard deviation of the data.
        """
        return np.std(data)

    def calculate_correlation(
        self, covariance: np.ndarray, std_X1: np.ndarray, std_X2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate correlation between two arrays.

        Args:
            covariance (np.ndarray): covariance matrix.
            std_X1 (np.ndarray): standard deviation of feature 1.
            std_X2 (np.ndarray): standard deviation of feature 2.

        Returns:
            np.ndarray: correlation between X and Y.
        """
        return (covariance / (std_X1 * std_X2))[1][0]

    def calculate_test_statistic(self, correlation: np.ndarray, N: int) -> np.ndarray:
        """
        Calculate test statistic.

        Args:
            correlation (np.ndarray): correlation between X and Y.
            N (int): number of samples.

        Returns:
            np.ndarray: test statistic.
        """
        return (correlation * np.sqrt(N - 2)) / np.sqrt(1 - correlation**2)

    def calculate_p_value_t_test(self, t_test: np.ndarray, df: int) -> np.ndarray:
        """
        Calculate p-value for t-test.

        Args:
            t_test (np.ndarray): test statistic.
            df (int): degrees of freedom.

        Returns:
            np.ndarray: p-value.
        """
        return 2 * (1 - t.cdf(np.abs(t_test), df))

    def calculate_critical_value(self, df: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            df (int): degrees of freedom.
            alpha (float): significance level.

        Returns:
            critical_value (float): critical value.
        """
        return t.isf(q=alpha / 2, df=df)

    def statistical_inference(self, p_value: float, alpha: float) -> bool:
        """
        Perform statistical inference.

        Args:
            p_value: (float): p value.
            alpha: (float): significance level.

        Returns:
            bool: (bool): True if H0 is not rejected, False otherwise.
        """
        if p_value >= alpha:
            return True
        else:
            return False
