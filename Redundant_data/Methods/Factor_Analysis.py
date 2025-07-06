import numpy as np
import pandas as pd
import torch
import random


class FactorAnalyzer:
    def __init__(
        self,
        n_components: int = 5,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 17,
    ):
        """
        Initialize FactorAnalyzer with the given parameters.

        Args:
            n_components (int, optional): Number of factors to extract. Defaults to 5.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-3.
            random_state (int, optional): Random seed for reproducibility. Defaults to 17.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.SMALL = 1e-12
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False

    def check_data(
        self, data: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Check if data is valid and convert it to numpy array.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.

        Returns:
            np.ndarray: Converted input data.

        Raises:
            TypeError: If data is not a pandas DataFrame, pandas Series, numpy array or torch tensor.
        """
        if (
            not isinstance(data, pd.DataFrame)
            and not isinstance(data, pd.Series)
            and not isinstance(data, np.ndarray)
            and not torch.is_tensor(data)
        ):
            raise TypeError(
                "Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor."
            )
        return np.array(data)

    def check_for_object_columns(self, data: np.ndarray) -> np.ndarray:
        """
        Check if data contains object columns and ensure all data is numeric.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Verified numeric data.

        Raises:
            TypeError: If data contains non-numeric columns.
        """
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError(
                "Your data contains object or string columns. Numeric data is obligated."
            )
        return np.array(data)

    def check_n_of_components(self, data: np.ndarray) -> None:
        """
        Verify that the number of components is valid for the given data.

        Args:
            data (np.ndarray): Input data.

        Raises:
            ValueError: If n_components is greater than the number of features.
        """
        if data.shape[1] < self.n_components:
            raise ValueError(
                "Specified n_components={} < number of features. It has to be greater or equal to the number of columns.".format(
                    self.n_components
                )
            )

    def fit(
        self, data: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> "FactorAnalyzer":
        """
        Fit the factor analysis model to the data.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.

        Returns:
            FactorAnalyzer: Fitted model instance.
        """
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.corr_ = self.calculate_correlation(data=self.fit_data)
        self.mean_ = self.calculate_mean(data=self.fit_data)
        self.fit_data = self.center_data(data=self.fit_data)
        Psi = self.initialize_psi(data=self.fit_data)
        self.components_, self.noise_variance_ = self.optimization(
            data=self.fit_data, Psi=Psi
        )
        self.eigenvalues_ = self.calculate_eigenvalues(components=self.components_)
        self.fit_used = True
        return self

    def fit_transform(
        self, data: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Fit the model and transform the data to its factor space.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.

        Returns:
            np.ndarray: Transformed data in factor space.
        """
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.corr_ = self.calculate_correlation(data=self.fit_data)
        self.mean_ = self.calculate_mean(data=self.fit_data)
        self.fit_data = self.center_data(data=self.fit_data)
        Psi = self.initialize_psi(data=self.fit_data)
        self.components_, self.noise_variance_ = self.optimization(
            data=self.fit_data, Psi=Psi
        )
        self.eigenvalues_ = self.calculate_eigenvalues(components=self.components_)
        self.fit_used = True
        return self.perform_transformation_to_factors(
            data=self.fit_data,
            noise_variance=self.noise_variance_,
            components=self.components_,
        )

    def transform(
        self, data: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Transform data to factor space using the fitted model.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | torch.Tensor): Input data.

        Returns:
            np.ndarray: Transformed data in factor space.

        Raises:
            AttributeError: If the model has not been fitted.
        """
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        data = self.center_data(data=data)
        return self.perform_transformation_to_factors(
            data=data, noise_variance=self.noise_variance_, components=self.components_
        )

    def calculate_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the correlation matrix of the data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Correlation matrix.
        """
        return np.corrcoef(data, rowvar=False)

    def calculate_mean(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the mean of each feature in the data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Mean values for each feature.
        """
        return np.mean(data, axis=0)

    def center_data(self, data: np.ndarray) -> np.ndarray:
        """
        Center the data by subtracting the mean.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Centered data.
        """
        return data - self.mean_

    def initialize_psi(self, data: np.ndarray) -> np.ndarray:
        """
        Initialize the noise variance (Psi) with ones.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Initialized noise variance vector.
        """
        return np.ones(data.shape[1])

    def optimization(
        self, data: np.ndarray, Psi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the optimization to find the factor loadings and noise variance.

        Args:
            data (np.ndarray): Input data.
            Psi (np.ndarray): Initial noise variance.

        Returns:
            tuple[np.ndarray, np.ndarray]: Factor loadings matrix and updated noise variance.
        """
        self.loglike_ = []
        old_loglike = -np.inf
        for i in range(self.max_iter):
            data_transformed = self.transform_data(data=data, Psi=Psi)
            Lambda, V_T, unexpected_variance = self.perform_svd(data=data_transformed)
            F = self.calculate_factor_matrix(Psi=Psi, Lambda=Lambda, V_T=V_T)
            loglike = self.calculate_loglikelihood(
                data=data_transformed,
                Lambda=Lambda,
                Psi=Psi,
                unexpected_variance=unexpected_variance,
            )
            self.loglike_.append(loglike)
            if (loglike - old_loglike) < self.tol:
                break
            old_loglike = loglike
            Psi = self.update_psi(data=data, F=F)
        return F, Psi

    def transform_data(self, data: np.ndarray, Psi: np.ndarray) -> np.ndarray:
        """
        Transform data by scaling with noise variance.

        Args:
            data (np.ndarray): Input data.
            Psi (np.ndarray): Noise variance.

        Returns:
            np.ndarray: Transformed data.
        """
        return data / np.sqrt((Psi + self.SMALL) * data.shape[0])

    def perform_svd(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Perform Singular Value Decomposition on the data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                - Modified eigenvalues
                - Right singular vectors
                - Norm of unused eigenvalues
        """
        U, Lambda, V_T = np.linalg.svd(data, full_matrices=False)
        Lambda_modified = Lambda**2
        return (
            Lambda_modified[: self.n_components],
            V_T[: self.n_components],
            np.linalg.norm(Lambda_modified[self.n_components :]),
        )

    def calculate_factor_matrix(
        self, Psi: np.ndarray, Lambda: np.ndarray, V_T: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the factor loadings matrix.

        Args:
            Psi (np.ndarray): Noise variance.
            Lambda (np.ndarray): Eigenvalues.
            V_T (np.ndarray): Right singular vectors.

        Returns:
            np.ndarray: Factor loadings matrix.
        """
        return (
            np.sqrt(Psi + self.SMALL)
            * V_T
            * np.sqrt(np.maximum(Lambda - 1.0, 0.0))[:, np.newaxis]
        )

    def calculate_loglikelihood(
        self,
        data: np.ndarray,
        Lambda: np.ndarray,
        Psi: np.ndarray,
        unexpected_variance: float,
    ) -> float:
        """
        Calculate the log-likelihood of the current model.

        Args:
            data (np.ndarray): Input data.
            Lambda (np.ndarray): Eigenvalues.
            Psi (np.ndarray): Noise variance.
            unexpected_variance (float): Variance not explained by the model.

        Returns:
            float: Log-likelihood value.
        """
        return (
            -data.shape[0]
            / 2
            * (
                np.sum(np.log(Lambda))
                + self.n_components
                + unexpected_variance
                + np.log(np.prod(2 * np.pi * Psi))
            )
        )

    def update_psi(self, data: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Update the noise variance based on the data and factor matrix.

        Args:
            data (np.ndarray): Input data.
            F (np.ndarray): Factor loadings matrix.

        Returns:
            np.ndarray: Updated noise variance.
        """
        return np.maximum(np.var(data, axis=0) - np.sum(F**2, axis=0), self.SMALL)

    def calculate_eigenvalues(self, components: np.ndarray) -> np.ndarray:
        """
        Calculate eigenvalues of the correlation matrix.

        Args:
            components (np.ndarray): Factor loadings matrix.

        Returns:
            np.ndarray: Eigenvalues in descending order.
        """
        correlation_matrix = self.corr_.copy()
        np.fill_diagonal(correlation_matrix, (components**2).sum(axis=1))
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = eigenvalues[::-1]
        return eigenvalues

    def perform_transformation_to_factors(
        self, data: np.ndarray, noise_variance: np.ndarray, components: np.ndarray
    ) -> np.ndarray:
        """
        Transform data to factor space.

        Args:
            data (np.ndarray): Input data.
            noise_variance (np.ndarray): Noise variance.
            components (np.ndarray): Factor loadings matrix.

        Returns:
            np.ndarray: Data transformed to factor space.
        """
        cov = np.linalg.inv(
            np.identity(len(components))
            + np.dot(components * noise_variance ** (-1), components.T)
        )
        rest = np.dot(data, (components * noise_variance ** (-1)).T)
        return np.dot(rest, cov)

    def set_params(self, **kwargs) -> "FactorAnalyzer":
        """
        Set the parameters of the model.

        Args:
            **kwargs: Model parameters to set.

        Returns:
            FactorAnalyzer: Model instance with updated parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def check_fit(self, fit_used: bool) -> None:
        """
        Check if the model has been fitted.

        Args:
            fit_used (bool): Flag indicating if the model has been fitted.

        Raises:
            AttributeError: If the model has not been fitted.
        """
        if fit_used == False:
            raise AttributeError("FactorAnalyzer has to be fitted first.")


if __name__ == "__main__":
    np.random.seed(17)
    sample_data = np.random.randn(100, 8)
    sample_df = pd.DataFrame(sample_data, columns=[f"feature_{i}" for i in range(8)])

    fa = FactorAnalyzer(n_components=3, random_state=17)
    fa.fit(sample_df)
    factors = fa.transform(sample_df)

    print(f"Original data shape: {sample_df.shape}")
    print(f"Transformed data shape: {factors.shape}")
    print(f"Components shape: {fa.components_.shape}")
    print(f"Eigenvalues: {fa.eigenvalues_[:5]}")  # Show first 5 eigenvalues
    print(
        f"Explained variance ratio: {fa.eigenvalues_[:fa.n_components] / np.sum(fa.eigenvalues_)}"
    )
