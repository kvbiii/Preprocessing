from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class FactorAnalyzer():
    def __init__(self, n_components=5, max_iter=1000, tol=1e-3, random_state=17):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.SMALL = 1e-12
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)

    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)
    
    def check_n_of_components(self, data):
        if(data.shape[1] < self.n_components):
            raise ValueError('Specified n_components={} < number of features. It has to be greater or equal to the number of columns.'.format(self.n_components))

    def fit(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.corr_ = self.calculate_correlation(data=self.fit_data)
        self.mean_ = self.calculate_mean(data=self.fit_data)
        self.fit_data = self.center_data(data=self.fit_data)
        Psi = self.initialize_psi(data=self.fit_data)
        self.components_, self.noise_variance_ = self.optimization(data=self.fit_data, Psi=Psi)
        self.eigenvalues_ = self.calculate_eigenvalues(components=self.components_)
        self.fit_used = True

    def fit_transform(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.corr_ = self.calculate_correlation(data=self.fit_data)
        self.mean_ = self.calculate_mean(data=self.fit_data)
        self.fit_data = self.center_data(data=self.fit_data)
        Psi = self.initialize_psi(data=self.fit_data)
        self.components_, self.noise_variance_ = self.optimization(data=self.fit_data, Psi=Psi)
        self.eigenvalues_ = self.calculate_eigenvalues(components=self.components_)
        self.fit_used = True
        return self.perform_transformation_to_factors(data=self.fit_data, noise_variance=self.noise_variance_, components=self.components_)
    
    def transform(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        data = self.center_data(data=data)
        return self.perform_transformation_to_factors(data=data, noise_variance=self.noise_variance_, components=self.components_)
    
    def calculate_correlation(self, data):
        return np.corrcoef(data, rowvar=False)

    def calculate_mean(self, data):
        return np.mean(data, axis=0)

    def center_data(self, data):
        return data - self.mean_
    
    def initialize_psi(self, data):
        return np.ones(data.shape[1])
    
    def optimization(self, data, Psi):
        self.loglike_ = []
        old_loglike = -np.inf
        for i in range(self.max_iter):
            data_transformed = self.transform_data(data=data, Psi=Psi)
            Lambda, V_T, unexpected_variance = self.perform_svd(data=data_transformed)
            F = self.calculate_factor_matrix(Psi=Psi, Lambda=Lambda, V_T=V_T)
            loglike = self.calculate_loglikelihood(data=data_transformed, Lambda=Lambda, Psi=Psi, unexpected_variance=unexpected_variance)
            self.loglike_.append(loglike)
            if (loglike - old_loglike) < self.tol:
                break
            old_loglike = loglike
            Psi = self.update_psi(data=data, F=F)
        return F, Psi
    
    def transform_data(self, data, Psi):
        return data/np.sqrt((Psi+self.SMALL)*data.shape[0])
    
    def perform_svd(self, data):
        U, Lambda, V_T = np.linalg.svd(data, full_matrices=False)
        Lambda_modified = Lambda**2
        return (Lambda_modified[:self.n_components], V_T[:self.n_components], np.linalg.norm(Lambda_modified[self.n_components:]))
    
    def calculate_factor_matrix(self, Psi, Lambda, V_T):
        return np.sqrt(Psi+self.SMALL)*V_T*np.sqrt(np.maximum(Lambda - 1.0, 0.0))[:, np.newaxis]

    def calculate_loglikelihood(self, data, Lambda, Psi, unexpected_variance):
        return -data.shape[0]/2*(np.sum(np.log(Lambda)) + self.n_components + unexpected_variance+np.log(np.prod(2*np.pi*Psi)))
    
    def update_psi(self, data, F):
        return np.maximum(np.var(data, axis=0) - np.sum(F**2, axis=0), self.SMALL)
    
    def calculate_eigenvalues(self, components):
        correlation_matrix = self.corr_.copy()
        np.fill_diagonal(correlation_matrix, (components**2).sum(axis=1))
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        eigenvalues = eigenvalues[::-1]
        return eigenvalues
    
    def perform_transformation_to_factors(self, data, noise_variance, components):
        cov = np.linalg.inv(np.identity(len(components))+np.dot(components*noise_variance**(-1), components.T))
        rest = np.dot(data, (components*noise_variance**(-1)).T)
        return np.dot(rest, cov)
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('FactorAnalyzer has to be fitted first.')