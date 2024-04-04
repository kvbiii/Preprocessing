from sklearn.model_selection import KFold
from sklearn.metrics import *
import numpy as np
import pandas as pd
import typing


class ForwardFeatureSelection:
    """Forward feature selection algorithm."""

    def __init__(
        self,
        algorithm: typing.Any,
        metric: str,
        cv: typing.Any = KFold(n_splits=5, shuffle=True),
    ):
        """This method initializes the class.

        Args:
            algorithm (typing.Any): algorithm that will be used for feature selection.
            metric (str): metric that will be used for evaluation.
            cv (typing.Any): cross-validation strategy that will be used for evaluation.
        """
        self.algorithm = algorithm
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
        self.cv = cv

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
        """This method fits the model.

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
            self.indices_of_best_, self.support_ = self.perform_ffs_no_cv(
                X_train, y_train, X_valid, y_valid, verbose
            )
        else:
            self.indices_of_best_, self.support_ = self.perform_ffs_cv(X, y, verbose)

    def perform_ffs_no_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        verbose: bool,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """This method performs forward feature selection without cross-validation.

        Args:
            X_train: (np.ndarray): training input data.
            y_train: (np.ndarray): training target data.
            X_valid: (np.ndarray): validation input data.
            y_valid: (np.ndarray): validation target data.
            verbose: (bool): whether to print fitting process.

        Returns:
            best_features: (np.ndarray): best features.
            support_: (np.ndarray): boolean mask of the best features.
        """
        best_init_score = -np.inf
        i = 0
        while i < X_train.shape[-1]:
            self.algorithm.fit(X_train[:, i].reshape(-1, 1), y_train)
            if self.metric_type == "preds":
                y_pred_original = self.algorithm.predict(X_valid[:, i].reshape(-1, 1))
            else:
                y_pred_original = self.algorithm.predict_proba(
                    X_valid[:, i].reshape(-1, 1)
                )[:, 1]
            score = self.eval_metric(y_valid, y_pred_original)
            if best_init_score < score:
                best_init_score = score
                best_features = [i]
            i = i + 1
        best_score = best_init_score
        rest_features = np.setdiff1d(
            [i for i in range(X_train.shape[1])], best_features
        )
        while True:
            restart = False
            for i in rest_features:
                if verbose == True:
                    print(
                        "We check how the current set: {} will manage with an additional variable: {}".format(
                            best_features, i
                        )
                    )
                X_train_copy = X_train[:, best_features + [i]]
                X_valid_copy = X_valid[:, best_features + [i]]
                self.algorithm.fit(X_train_copy, y_train)
                if self.metric_type == "preds":
                    y_pred_inner = self.algorithm.predict(X_valid_copy)
                else:
                    y_pred_inner = self.algorithm.predict_proba(X_valid_copy)[:, 1]
                score = self.eval_metric(y_valid, y_pred_inner)
                if score > best_score:
                    if verbose == True:
                        print("Improved! Because {}>{}".format(score, best_score))
                    best_score = score
                    best_features.append(i)
                    rest_features = np.setdiff1d(
                        [i for i in range(X_train.shape[-1])], best_features
                    )
                    restart = True
                    break
            if restart == False:
                break
        support_ = np.array(
            [True if i in best_features else False for i in range(0, X_train.shape[1])]
        ).astype(bool)
        return best_features, support_

    def perform_ffs_cv(
        self, X: np.ndarray, y: np.ndarray, verbose: bool
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """This method performs forward feature selection with cross-validation.

        Args:
            X: (np.ndarray): input data.
            y: (np.ndarray): target data.
            verbose: (bool): whether to print fitting process.

        Returns:
            best_features: (np.ndarray): best features.
            support_: (np.ndarray): boolean mask of the best features.
        """
        best_init_score = -np.inf
        i = 0
        while i < X.shape[-1]:
            score = self.perform_cv(X[:, i].reshape(-1, 1), y)
            if best_init_score < score:
                best_init_score = score
                best_features = [i]
            i = i + 1
        best_score = best_init_score
        rest_features = np.setdiff1d([i for i in range(X.shape[1])], best_features)
        while True:
            restart = False
            for i in rest_features:
                if verbose == True:
                    print(
                        "We check how the current set: {} will manage with an additional variable: {}".format(
                            best_features, i
                        )
                    )
                X_copy = X[:, best_features + [i]]
                score = self.perform_cv(X_copy, y)
                if score > best_score:
                    if verbose == True:
                        print("Improved! Because {}>{}".format(score, best_score))
                    best_score = score
                    best_features.append(i)
                    rest_features = np.setdiff1d(
                        [i for i in range(X.shape[-1])], best_features
                    )
                    restart = True
                    break
            if restart == False:
                break
        support_ = np.array(
            [True if i in best_features else False for i in range(0, X.shape[1])]
        ).astype(bool)
        return best_features, support_

    def perform_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """This method performs cross-validation.

        Args:
            X: (np.ndarray): input data.
            y: (np.ndarray): target data.

        Returns:
            float: cross-validation score.
        """
        scores = []
        for train_index, test_index in self.cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.algorithm.fit(X_train, y_train)
            if self.metric_type == "preds":
                y_pred = self.algorithm.predict(X_test)
            else:
                y_pred = self.algorithm.predict_proba(X_test)[:, 1]
            scores.append(self.eval_metric(y_test, y_pred))
        return np.mean(scores)
