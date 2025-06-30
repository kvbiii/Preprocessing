import typing
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree


class MutualInformation:
    """Class for calculating mutual information between features and target."""

    def __init__(self, n_neighbors: int = 3, random_state: int = 17) -> None:
        """Initialize the class.

        Args:
            n_neighbors (int, default=3): number of neighbors to consider for continuous features.
            random_state (int, default=17): random seed for reproducibility.
        """
        self.n_neighbors_ = n_neighbors
        self.random_state_ = random_state
        np.random.seed(17)

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

    def check_discrete_features(
        self,
        X: np.ndarray,
        discrete_features: typing.Union[bool, np.ndarray, typing.List],
    ) -> np.ndarray:
        """Check if discrete_features is bool, numpy array or list and convert it to numpy array.

        Args:
            X: (np.ndarray): input data.
            discrete_features: (Union[bool, np.ndarray, List]): discrete features.

        Returns:
            discrete_features: (np.ndarray): converted discrete features.
        """
        if (
            not isinstance(discrete_features, bool)
            and not isinstance(discrete_features, np.ndarray)
            and not isinstance(discrete_features, typing.List)
        ):
            raise TypeError(
                "Wrong type of discrete_features. It should be bool, numpy array or List."
            )
        if isinstance(discrete_features, np.ndarray) or isinstance(
            discrete_features, typing.List
        ):
            if len(discrete_features) > X.shape[1]:
                raise ValueError(
                    "Length of discrete_features list should be less or equal to the number of features in X."
                )
            if all(isinstance(i, np.bool_) for i in discrete_features):
                return np.array(discrete_features)
            return np.array(
                [True if i in discrete_features else False for i in range(X.shape[1])]
            )
        elif discrete_features == True:
            return np.array([True for i in range(X.shape[1])])
        return np.array([False for i in range(X.shape[1])])

    def check_discrete_target(self, target_discrete: bool) -> bool:
        """Check if target_discrete is bool.

        Args:
            target_discrete: (bool): target type.

        Returns:
            target_discrete: (bool): target type.
        """
        if not isinstance(target_discrete, bool):
            raise TypeError("Wrong type of target_discrete. It should be bool.")
        return target_discrete

    def fit(
        self,
        X: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        y: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
        discrete_features: typing.Union[bool, np.ndarray, typing.List],
        target_discrete: bool = True,
    ) -> np.ndarray:
        """Calculate mutual information between features and target.

        Args:
            X: (Union[pd.DataFrame, pd.Series, np.ndarray]): input data.
            y: (Union[pd.DataFrame, pd.Series, np.ndarray]): target data.
            discrete_features: (Union[bool, np.ndarray, List]): discrete features.
            target_discrete: (bool, default=True): target type.

        Returns:
            (np.ndarray): mutual information between features and target.
        """
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        self.discrete_features_ = self.check_discrete_features(
            X=X, discrete_features=discrete_features
        )
        self.discrete_target_ = self.check_discrete_target(
            target_discrete=target_discrete
        )
        return self.estimate_mi(X, y)

    def estimate_mi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate mutual information between features and target.

        Args:
            X: (np.ndarray): input data.
            y: (np.ndarray): target data.

        Returns:
            (np.ndarray): mutual information between features and target.
        """
        X, y = self.prepare_data(X, y)
        return np.array(
            [
                self.calculate_mi(X[:, i], y, self.discrete_features_[i])
                for i in range(X.shape[1])
            ]
        )

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Prepare data for mutual information calculation.

        Args:
            X: (np.ndarray): input data.
            y: (np.ndarray): target data.

        Returns:
            (np.ndarray, np.ndarray): prepared input and target data.
        """
        discrete_mask = self.discrete_features_.copy()
        continous_mask = ~discrete_mask
        X = X.astype(np.float64)
        X[:, continous_mask] = X[:, continous_mask] / np.std(
            X[:, continous_mask], axis=0
        )
        # Add small noise to continuous features
        means = np.maximum(1, np.mean(np.abs(X[:, continous_mask]), axis=0))
        X[:, continous_mask] += (
            1e-10
            * means
            * np.random.standard_normal(size=(X.shape[0], np.sum(continous_mask)))
        )
        y = y.astype(np.float64)
        if self.discrete_target_ is False:
            y = y / np.std(y)
            # Add small noise to continuous features
            y += (
                1e-10
                * np.maximum(1, np.mean(np.abs(y)))
                * np.random.standard_normal(size=(X.shape[0],))
            )
        return X, y

    def calculate_mi(
        self, x: np.ndarray, y: np.ndarray, discrete_feature: bool
    ) -> float:
        """Choose the right mutual information calculation method.

        Args:
            x: (np.ndarray): feature data.
            y: (np.ndarray): target data.
            discrete_feature: (bool): feature type.

        Returns:
            (float): mutual information between feature and target.
        """
        if discrete_feature and self.discrete_target_:
            return self.mutual_information_dd(x, y)
        elif discrete_feature and not self.discrete_target_:
            return self.mutual_information_cd(y, x)
        elif not discrete_feature and self.discrete_target_:
            return self.mutual_information_cd(x, y)
        else:
            return self.mutual_information_cc(x, y)

    def mutual_information_dd(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between discrete feature and discrete target.

        Args:
            x: (np.ndarray): feature data.
            y: (np.ndarray): target data.

        Returns:
            (float): mutual information between discrete feature and target.
        """
        crosstab = np.array(pd.crosstab(x, y, margins=True))
        crosstab = crosstab / crosstab[-1, -1]
        mi = 0
        for i in range(crosstab.shape[0] - 1):
            for j in range(crosstab.shape[1] - 1):
                if crosstab[i, j] != 0:
                    mi += crosstab[i, j] * np.log(
                        crosstab[i, j] / (crosstab[i, -1] * crosstab[-1, j])
                    )
        return mi

    def mutual_information_cd(
        self, continous: np.ndarray, discrete: np.ndarray
    ) -> float:
        """Calculate mutual information between continuous feature and dicrete target.

        Args:
            continous: (np.ndarray): continuous feature data.
            discrete: (np.ndarray): target data.

        Returns:
            (float): mutual information between continuous feature and target.
        """
        continous = continous.reshape(-1, 1)
        N = continous.shape[0]
        N_x = np.empty(N)
        radius = np.empty(N)
        k_all = np.empty(N)
        nearest_neighbors = NearestNeighbors()
        for label in np.unique(discrete):
            mask = discrete == label
            count = np.sum(mask)
            if count > 1:
                k = min(self.n_neighbors_, count - 1)
                k_all[mask] = k
                nearest_neighbors.set_params(n_neighbors=k)
                nearest_neighbors.fit(continous[mask].reshape(-1, 1))
                r = nearest_neighbors.kneighbors()[0]
                radius[mask] = np.nextafter(r[:, -1], 0)
            N_x[mask] = count
        mask = N_x > 1
        N_x = N_x[mask]
        k_all = k_all[mask]
        continous = continous[mask].reshape(-1, 1)
        radius = radius[mask]
        kd = KDTree(continous, metric="chebyshev")
        m_all = np.array(
            kd.query_radius(continous, radius, count_only=True, return_distance=False)
        )
        mi = (
            digamma(N)
            - np.mean(digamma(N_x))
            + np.mean(digamma(k_all))
            - np.mean(digamma(m_all))
        )
        return max(0, mi)

    def mutual_information_cc(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between continuous feature and continuous target.

        Args:
            x: (np.ndarray): feature data.
            y: (np.ndarray): target data.

        Returns:
            (float): mutual information between continuous feature and target.
        """
        N = x.shape[0]
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        xy = np.hstack((x, y))
        nn = NearestNeighbors(metric="chebyshev", n_neighbors=self.n_neighbors_)
        nn.fit(xy)
        radius = nn.kneighbors()[0]
        radius = np.nextafter(radius[:, -1], 0)
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=x.squeeze(), y=y.squeeze(), mode='markers', marker=dict(color='blue'), marker_size=10))
        # for i in range(x.shape[0]):
        #     fig.add_shape(type="circle", xref="x", yref="y", x0=x[i][0]-radius[i]-0.1, y0=y[i][0]-radius[i]-0.1, x1=x[i][0]+radius[i]+0.1, y1=y[i][0]+radius[i]+0.1, opacity=0.2, line=dict(color="black", width=2), fillcolor="white")
        # fig.update_layout(template="simple_white", width=600, height=600, title_text="<b>Nearest neighbors<b>", title_x=0.5, yaxis_title="y", xaxis_title="x", font=dict(family="Times New Roman",size=16,color="Black"))
        # fig.show("png")
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=x.squeeze(), y=[0 for i in range(0, x.shape[0])], mode="markers", marker=dict(color=px.colors.qualitative.Plotly), marker_size=15, showlegend=False))
        # fig.add_trace(go.Scatter(x=[x[i][0]-radius[i] for i in range(0, x.shape[0])], y=[0 for i in range(0, len(x))], mode="markers", marker_symbol="arrow-up", marker=dict(color=px.colors.qualitative.Plotly), marker_size=15, showlegend=False))
        # fig.add_trace(go.Scatter(x=[x[i][0]+radius[i] for i in range(0, x.shape[0])], y=[0 for i in range(0, len(x))], mode="markers", marker_symbol="arrow-up", marker=dict(color=px.colors.qualitative.Plotly), marker_size=15, showlegend=False))
        # fig.update_xaxes(showgrid=False)
        # fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='black', zerolinewidth=1, showticklabels=False)
        # fig.update_layout(plot_bgcolor='white', height=400, title_text="<b>X distances<b>", title_x=0.5, xaxis_title="x", font=dict(family="Times New Roman",size=16,color="Black"))
        # fig.show("png")
        kd = KDTree(x, metric="chebyshev")
        N_x = kd.query_radius(x, radius, count_only=True, return_distance=False)
        kd = KDTree(y, metric="chebyshev")
        N_y = kd.query_radius(y, radius, count_only=True, return_distance=False)
        mi = (
            digamma(N)
            - np.mean(digamma(N_x))
            + digamma(self.n_neighbors_)
            - np.mean(digamma(N_y))
        )
        return max(0, mi)
