import numpy as np
import pandas as pd
import typing
from scipy.stats import binom
from itertools import accumulate
import plotly.graph_objects as go


class Boruta:
    """The Boruta algorithm is a wrapper feature selection method that finds all relevant features in a dataset."""

    def __init__(self, algorithm: typing.Any, max_iter: int) -> None:
        """Initialize the Boruta algorithm.

        Args:
            algorithm (Any): machine learning algorithm.
            max_iter (int): number of iterations.
        """
        self.algorithm = algorithm
        self.max_iter = max_iter

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
        """Fit the Boruta algorithm.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            verbose (bool): print the results of the algorithm.
        """
        X = self.check_X(X)
        X = self.check_for_object_columns(X)
        y = self.check_y(y)
        self.indices_of_best_, self.support_ = self.perform_boruta(X, y, verbose)

    def perform_boruta(
        self, X: np.ndarray, y: np.ndarray, verbose: bool
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Perform the Boruta algorithm.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): target data.
            verbose (bool): print the results of the algorithm.

        Returns:
            indices_of_best_ (np.ndarray): indices of the best features.
            support_ (np.ndarray): boolean mask of the best features.
        """
        features_dict = {i: 0 for i in range(X.shape[-1])}
        i = 0
        while i < self.max_iter:
            X_shadow = X.copy()
            for j in range(X_shadow.shape[-1]):
                np.random.shuffle(X_shadow[:, j])
            X_extended = np.c_[X, X_shadow]
            self.algorithm.fit(X_extended, y)
            if hasattr(self.algorithm, "feature_importances_"):
                feature_importances = self.algorithm.feature_importances_
            else:
                feature_importances = self.algorithm.coef_
            max_shadow = np.max(feature_importances[X.shape[-1] :])
            for j in range(X.shape[-1]):
                if feature_importances[j] > max_shadow:
                    features_dict[j] = features_dict[j] + 1
            i = i + 1
        self.feature_importances_ = np.array(
            [features_dict[i] for i in range(X.shape[-1])]
        )
        best_features = self.determine_best_features(
            self.feature_importances_, verbose=verbose
        )
        support_ = np.array(
            [True if i in best_features else False for i in range(0, X.shape[1])]
        ).astype(bool)
        return best_features, support_

    def determine_best_features(
        self, feature_importances: np.ndarray, verbose: bool
    ) -> np.ndarray:
        """Determine the best features based on binomial distribution. and plot the results.

        Args:
            feature_importances (np.ndarray): feature importances.
            verbose (bool): print the results of the algorithm.

        Returns:
            important_features (np.ndarray): indices of the best features.
        """
        pmf = [binom.pmf(i, self.max_iter, 0.5) for i in range(0, self.max_iter)]
        lower_blue, upper_blue = self.calculate_threshold(pmf)
        important_features, maybe_important, not_important = self.classify_features(
            feature_importances, lower_blue, upper_blue
        )
        if verbose == True:
            print(f"Important features: {important_features}")
            print(f"Maybe important features: {maybe_important}")
            print(f"Not important features: {not_important}")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(self.max_iter)],
                    y=pmf,
                    mode="lines+markers",
                    name="PMF",
                )
            )
            for i in range(len(important_features)):
                fig.add_vline(
                    x=feature_importances[important_features[i]],
                    line_dash="dash",
                    line_width=2,
                    line_color="green",
                    annotation_text=f"Feature {important_features[i]}",
                    annotation_position="bottom right",
                )
            for i in range(len(maybe_important)):
                fig.add_vline(
                    x=feature_importances[maybe_important[i]],
                    line_dash="dash",
                    line_width=2,
                    line_color="blue",
                    annotation_text=f"Feature {maybe_important[i]}",
                    annotation_position="bottom right",
                )
            for i in range(len(not_important)):
                fig.add_vline(
                    x=feature_importances[not_important[i]],
                    line_dash="dash",
                    line_width=2,
                    line_color="red",
                    annotation_text=f"Feature {not_important[i]}",
                    annotation_position="bottom right",
                )
            fig.add_vline(
                x=lower_blue,
                line_width=2,
                line_color="red",
                annotation_text="Unimportant treshold",
                annotation_position="left",
            )
            fig.add_vline(
                x=upper_blue,
                line_width=2,
                line_color="green",
                annotation_text="Important treshold",
                annotation_position="right",
            )
            fig.update_layout(
                template="simple_white",
                width=1200,
                height=800,
                title=f"Binomial distribution for {self.max_iter} iterations",
                title_x=0.5,
                xaxis_title="Number of iterations",
                yaxis_title="Probability",
                font=dict(family="Times New Roman", size=16, color="Black"),
            )
            fig.show("png")
        return important_features

    def calculate_threshold(self, pmf: np.ndarray) -> typing.Tuple[int, int]:
        """Calculate the threshold for the binomial distribution.

        Args:
            pmf (np.ndarray): probability mass function.

        Returns:
            lower_blue (int): lower blue threshold (5%).
            upper_blue (int): upper blue threshold (95%).
        """
        lower_blue = next(i for i, total in enumerate(accumulate(pmf)) if total > 0.05)
        upper_blue = next(i for i, total in enumerate(accumulate(pmf)) if total > 0.95)
        return lower_blue, upper_blue

    def classify_features(
        self, feature_importances: np.ndarray, lower_blue: int, upper_blue: int
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify features based on the thresholds.

        Args:
            feature_importances (np.ndarray): feature importances.
            lower_blue (int): lower blue threshold (5%).
            upper_blue (int): upper blue threshold (95%).

        Returns:
            important_features (np.ndarray): indices of important features.
            maybe_important (np.ndarray): indices of maybe important features.
            not_important (np.ndarray): indices of not important features.
        """
        important_features = []
        maybe_important = []
        not_important = []
        for i in range(len(feature_importances)):
            if feature_importances[i] > upper_blue:
                important_features.append(i)
            elif feature_importances[i] < lower_blue:
                not_important.append(i)
            else:
                maybe_important.append(i)
        return important_features, maybe_important, not_important
