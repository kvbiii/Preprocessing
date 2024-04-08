from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import typing


class Perturabtion:
    """This class is used to perform perturbation method for feature selection. It is based on the idea of changing the distribution of one column and checking how it affects the model performance."""

    def __init__(self, algorithm: typing.Any, metric: str, n_repeats: int):
        """This method initializes the class.

        Args:
            algorithm (typing.Any): algorithm to be used for feature selection.
            metric (str): metric to be used for evaluation.
            n_repeats (int): number of repeats for perturbation.
        """
        self.algorithm = algorithm
        self.n_repeats = n_repeats
        metrics = {
            "accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
            "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
            "neg_mse": [lambda y, y_pred: -mean_squared_error(y, y_pred), "preds"],
            "neg_rmse": [
                lambda y, y_pred: -mean_squared_error(y, y_pred) ** 0.5,
                "preds",
            ],
            "neg_mae": [lambda y, y_pred: -mean_absolute_error(y, y_pred), "preds"],
        }
        if metric not in metrics:
            raise ValueError("Unsupported metric: {}".format(metric))
        self.metric = metric
        self.eval_metric = metrics[metric][0]
        self.metric_type = metrics[metric][1]

    def check_X(
        self, X: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Check if X is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

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
        if X.ndim == 1:
            X = X[None, :]
        return X

    def check_y(
        self, y: typing.Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Check if y is pandas DataFrame, pandas Series or numpy array and convert it to numpy array.

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
        """Check if X contains object columns and convert it to numeric data.

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
        X_valid: typing.Union[pd.DataFrame, pd.Series, np.ndarray] = None,
        y_valid: typing.Union[pd.DataFrame, pd.Series, np.ndarray] = None,
        verbose: bool = False,
    ):
        """Fit the model and calculate feature importances.

        Args:
            X: (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y: (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            X_valid: (Union[pd.DataFrame, pd.Series, np.ndarray]): validation input data.
            y_valid: (Union[pd.DataFrame, pd.Series, np.ndarray]): validation target data.
            verbose: (bool): whether to print fitting process.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
        y = self.check_y(y)
        if X_valid is not None:
            X_train = np.copy(X)
            y_train = np.copy(y)
            X_valid = self.check_X(X_valid)
            X_valid = self.check_for_object_columns(X_valid)
            y_valid = self.check_y(y_valid)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=17
            )
        self.unscaled_feature_importances_, self.feature_importances_, self.ranking_ = (
            self.perform_perturbation(X_train, y_train, X_valid, y_valid, verbose)
        )

    def perform_perturbation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        verbose: bool = False,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform perturbation method for feature selection.

        Args:
            X_train: (np.ndarray): training input data.
            y_train: (np.ndarray): training target data.
            X_valid: (np.ndarray): validation input data.
            y_valid: (np.ndarray): validation target data.
            verbose: (bool): whether to print fitting process.

        Returns:
            unscaled_feature_importances_: (np.ndarray): unscaled feature importances.
            feature_importances_: (np.ndarray): scaled feature importances.
            ranking_: (np.ndarray): ranking of features.
        """
        self.algorithm.fit(X_train, y_train)
        if self.metric_type == "preds":
            y_pred_original = self.algorithm.predict(X_valid)
        else:
            y_pred_original = self.algorithm.predict_proba(X_valid)[:, 1]
        score = self.eval_metric(y_valid, y_pred_original)
        if verbose == True:
            print(f"Original score: {round(score, 4)}")
        i = 0
        feature_importances = []
        while i < X_train.shape[-1]:
            X_valid_copy = X_valid.copy()
            inner_scores = []
            for j in range(self.n_repeats):
                # Change distribution of our column i with random values from - median to median of this column values
                perturbation = np.random.uniform(
                    low=-np.median(X_valid_copy[:, i]),
                    high=np.median(X_valid_copy[:, i]),
                    size=X_valid_copy.shape[:-1],
                )
                X_valid_copy[:, i] = X_valid_copy[:, i] + perturbation
                if self.metric_type == "preds":
                    y_pred_inner = self.algorithm.predict(X_valid_copy)
                else:
                    y_pred_inner = self.algorithm.predict_proba(X_valid_copy)[:, 1]
                inner_scores.append(self.eval_metric(y_valid, y_pred_inner))
            feature_importances.append(score - np.mean(inner_scores))
            if verbose == True:
                print(f"Feature importance for {i}: {round(feature_importances[i], 4)}")
            i = i + 1
        unscaled_feature_importances_ = np.array(feature_importances)
        feature_importances_ = np.squeeze(
            MinMaxScaler()
            .fit_transform(np.array(feature_importances).reshape(-1, 1))
            .reshape(1, -1)
        )
        ranking_ = rankdata(1 - feature_importances_, method="dense")
        return unscaled_feature_importances_, feature_importances_, ranking_
