from Mutual_Information import MutualInformation
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import pandas as pd
import typing


class RFECVMutualInformation(MutualInformation):
    """This class is a feature selection technique that uses Recursive Feature Elimination with Cross-Validation and Mutual Information."""

    def __init__(
        self,
        algorithm: typing.Any,
        metric: str,
        cv: KFold,
        discrete_features: typing.List[bool],
        target_discrete: bool,
    ):
        """This method initializes the class.

        Args:
            algorithm (typing.Any): algorithm that will be used for feature selection.
            metric (str): metric that will be used for evaluation.
            cv (KFold): cross-validation technique that will be used.
            discrete_features (typing.List[bool]): list of booleans that indicates whether the feature is discrete or not.
            target_discrete (bool): boolean that indicates whether the target is discrete or not.
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
        self.mutual_information = MutualInformation(n_neighbors=3, random_state=17)
        self.discrete_features_ = discrete_features
        self.discrete_target_ = target_discrete

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
        verbose: bool = False,
    ) -> None:
        """Fit the Recursive Feature Elimination with Cross-Validation and Mutual Information to the data.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            verbose (bool): boolean that indicates whether the information about the process will be printed out.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
        y = self.check_y(y)
        self.mi_ = list(
            self.mutual_information.fit(
                X,
                y,
                discrete_features=self.discrete_features_,
                target_discrete=self.discrete_target_,
            )
        )
        self.indices_of_best_, self.support_ = self.perform_rfecv(X, y, verbose)

    def perform_rfecv(
        self, X: np.ndarray, y: np.ndarray, verbose: bool
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Perform the Recursive Feature Elimination with Cross-Validation.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): target data.
            verbose (bool): boolean that indicates whether the information about the process will be printed out.

        Returns:
            best_features (np.ndarray): indices of the best features.
            support_ (np.ndarray): boolean mask of the best features.
        """
        original_features = [i for i in range(X.shape[-1])]
        features = original_features.copy()
        best_score = -np.inf
        worst = None
        while len(features) > 1:
            if worst is not None:
                features.remove(original_features[worst])
            X_copy = X[:, features]
            score = self.perform_cv(X_copy, y)
            if score > best_score:
                if verbose == True:
                    print(
                        "After removing the worst feature: {}, score improved, because: {}>{}".format(
                            worst, score, best_score
                        )
                    )
                best_score = score
                best_features = np.setdiff1d(features, [worst])
            worst = features[np.argmin([self.mi_[i] for i in features])]
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
