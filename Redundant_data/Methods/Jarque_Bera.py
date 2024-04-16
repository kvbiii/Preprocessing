import pandas as pd
import numpy as np
from scipy.stats import chi2
import typing


class JarqueBera:
    def check_data(
        self, data: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Check if the data is in the right format.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.ndarray]): data to be checked.

        Returns:
            data (np.ndarray): one dimensional numpy array data.
        """
        if (
            not isinstance(data, pd.DataFrame)
            and not isinstance(data, pd.Series)
            and not isinstance(data, np.ndarray)
        ):
            raise TypeError(
                "Wrong type of data. It should be pandas DataFrame, pandas Series or numpy array."
            )
        data = np.array(data)
        if data.ndim == 2:
            data = data.squeeze()
        return data

    def check_for_object_columns(self, data: np.ndarray) -> np.ndarray:
        """
        Check if the data contains object columns.

        Args:
            data (np.ndarray): data to be checked.

        Returns:
            data (np.ndarray): data after checking for object columns.
        """
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError(
                "Your data contains object or string columns. Numeric data is obligated."
            )
        return np.array(data)

    def fit(
        self,
        data: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        alpha: float = 0.05,
    ):
        """
        Perform Jarque Bera test.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.ndarray]): data to be tested.
            alpha (float): significance level.
        """
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        skewness_ = self.calculate_skewness(data=data)
        kurtosis_ = self.calculate_kurtosis(data=data)
        self.test_statistic_ = self.calculate_test_statistic(
            data=data, skewness=skewness_, kurtosis=kurtosis_
        )
        self.p_value_ = self.calculate_p_value(
            test_statistic=self.test_statistic_, df=2
        )
        self.critical_value_ = self.calculate_critical_value(df=2, alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)

    def calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calculate skewness of the data.

        Args:
            data (np.ndarray): data to calculate skewness for.

        Returns:
            skewness (float): skewness of the data.
        """
        return np.sum((data - np.mean(data)) ** 3) / (
            (data.shape[0] - 1) * np.std(data) ** 3
        )

    def calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calculate kurtosis of the data.

        Args:
            data (np.ndarray): data to calculate kurtosis for.

        Returns:
            kurtosis (float): kurtosis of the data.
        """
        return np.sum((data - np.mean(data)) ** 4) / (
            (data.shape[0] - 1) * np.std(data) ** 4
        )

    def calculate_test_statistic(
        self, data: np.ndarray, skewness: float, kurtosis: float
    ) -> float:
        """
        Calculate test statistic.

        Args:
            data (np.ndarray): data to calculate test statistic for.
            skewness (float): skewness of the data.
            kurtosis (float): kurtosis of the data.

        Returns:
            test_statistic (float): value of test statistic.
        """
        return data.shape[0] / 6 * (skewness**2 + 1 / 4 * (kurtosis - 3) ** 2)

    def calculate_p_value(self, test_statistic: float, df: int) -> float:
        """
        Calculate p value.

        Args:
            test_statistic (float): value of test statistic.
            df (int): degrees of freedom.

        Returns:
            p_value (float): p value.
        """
        return 1 - chi2.cdf(x=test_statistic, df=df)

    def calculate_critical_value(self, df: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            df (int): degrees of freedom.
            alpha (float): significance level.

        Returns:
            critical_value (float): critical value.
        """
        return chi2.isf(q=alpha, df=df)

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
