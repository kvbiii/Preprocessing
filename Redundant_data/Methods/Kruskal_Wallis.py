import pandas as pd
import numpy as np
from scipy.stats import rankdata, f
import typing


class KruskalWallis:
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

    def fit(
        self,
        X: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        y: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        alpha: float = 0.05,
    ):
        """
        Perform Kruskal Wallis test.

        Args:
            X: (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y: (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            alpha (float): significance level.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
        y = self.check_y(y)
        y_ranked = rankdata(y)
        y_ranked_divided_into_groups = self.divide_into_groups(X=X, y=y_ranked)
        r_c_mean, r_all_mean = self.calculate_rank_means(
            y_ranked=y_ranked_divided_into_groups
        )
        self.test_statistic_ = self.calculate_H_statistic(
            y_ranked=y_ranked_divided_into_groups,
            r_c_mean=r_c_mean,
            r_all_mean=r_all_mean,
        )
        self.p_value_ = self.calculate_p_value_F_test(
            F_test=self.test_statistic_,
            dfn=len(y_ranked_divided_into_groups) - 1,
            dfd=X.shape[0] - len(y_ranked_divided_into_groups),
        )
        self.critical_value_ = self.calculate_critical_value(
            dfn=len(y_ranked_divided_into_groups) - 1,
            dfd=X.shape[0] - len(y_ranked_divided_into_groups),
            alpha=alpha,
        )
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)

    def divide_into_groups(self, X: np.ndarray, y: np.ndarray) -> typing.List:
        """
        Divide the data into groups.

        Args:
            X: (np.ndarray): input data.
            y: (np.ndarray): target data.

        Returns:
            List: data divided into groups.
        """
        return [y[(X == value).squeeze()].flatten().tolist() for value in np.unique(X)]

    def calculate_rank_means(self, y_ranked: typing.List) -> typing.Tuple:
        """
        Calculate mean ranks for each group and all data.

        Args:
            y_ranked: (List): data divided into groups.

        Returns:
            Tuple: mean ranks for each group and all data.
        """
        r_c_mean = []
        C = len(y_ranked)
        for category in range(0, C):
            r_c_mean.append(np.mean(y_ranked[category]))
        flat_list = [item for row in y_ranked for item in row]
        r_all_mean = np.mean(flat_list)
        return r_c_mean, r_all_mean

    def calculate_H_statistic(
        self, y_ranked: typing.List, r_c_mean: typing.List, r_all_mean: float
    ) -> float:
        """
        Calculate H statistic.

        Args:
            y_ranked: (List): data divided into groups.
            r_c_mean: (List): mean ranks for each group.
            r_all_mean: (float): mean rank for all data.

        Returns:
            float: H statistic.
        """
        nominator = 0
        denominator = 0
        N = 0
        C = len(y_ranked)
        for category in range(0, C):
            N += len(y_ranked[category])
            nominator += len(y_ranked[category]) * (
                (r_c_mean[category] - r_all_mean) ** 2
            )
            denominator += np.sum((y_ranked[category] - r_all_mean) ** 2)
        return (N - 1) * nominator / denominator

    def calculate_p_value_F_test(self, F_test: float, dfn: int, dfd: int) -> float:
        """
        Calculate p value for F test.

        Args:
            F_test: (float): F test statistic.
            dfn: (int): degrees of freedom numerator.
            dfd: (int): degrees of freedom denominator.

        Returns:
            float: p value of F test.
        """
        return 1 - f.cdf(F_test, dfn, dfd)

    def calculate_critical_value(self, dfn: int, dfd: int, alpha: float) -> float:
        """
        Calculate critical value.

        Args:
            df: (int): degrees of freedom.
            alpha: (float): significance level.

        Returns:
            critical_value: (float): critical value.
        """
        return f.isf(q=alpha, dfn=dfn, dfd=dfd)

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
