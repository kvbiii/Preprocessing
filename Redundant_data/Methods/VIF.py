import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import typing


class VIF:
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
        self.features_names = data.columns.tolist()
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

    def fit(self, data: typing.Union[pd.DataFrame, pd.Series, np.ndarray]) -> None:
        """
        Perform VIF test.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.ndarray]): data to be tested.
        """
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        self.r_squared_ = []
        for feature in range(0, data.shape[1]):
            self.r_squared_.append(self.calculate_r_squared(X=data, feature=feature))
        self.vif_statistic = self.calculate_vif_statistic(r_squared=self.r_squared_)
        self.summary_ = pd.DataFrame(
            {"Feature name": self.features_names, "VIF": self.vif_statistic}
        )

    def calculate_r_squared(self, X: np.ndarray, feature: int) -> float:
        """
        Calculate R squared for a given feature.

        Args:
            X (np.ndarray): input data.
            feature (int): feature index.

        Returns:
            float: R squared value.
        """
        y = X[:, feature]
        X = np.delete(X, feature, axis=1)
        estimator = LinearRegression(fit_intercept=True)
        estimator.fit(X, y)
        y_pred = estimator.predict(X)
        RSS = np.sum((y - y_pred) ** 2)
        TSS = np.sum((y - np.mean(y)) ** 2)
        return 1 - RSS / TSS

    def calculate_vif_statistic(
        self, r_squared: typing.List[float]
    ) -> typing.List[float]:
        """
        Calculate VIF statistic.

        Args:
            r_squared (List[float]): list of R squared values.

        Returns:
            List[float]: list of VIF values.
        """
        return [1 / (1 - value) for value in r_squared]
