from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class SVD():
    def __init__(self, n_components=5, random_state=17):
        self.n_components = n_components
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
        self.U, self.Sigma, self.V, self.components_, self.eigenvalues_, self.explained_variance_, self.explained_variance_ratio_ = self.calculate_matrices_for_svd(data=self.fit_data)
        self.fit_used = True

    def fit_transform(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.U, self.Sigma, self.V, self.components_, self.eigenvalues_, self.explained_variance_, self.explained_variance_ratio_ = self.calculate_matrices_for_svd(data=self.fit_data)
        self.fit_used = True
        return np.dot(self.fit_data, self.V.T[:, :self.n_components])
    
    def transform(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.dot(data, self.V.T[:, :self.n_components])
    
    def calculate_matrices_for_svd(self, data):
        U, Sigma, V = np.linalg.svd(data, full_matrices=False)
        U, V = self.svd_flip(U, V)
        components_ = V[:self.n_components, :]
        eigenvalues_ = Sigma**2
        explained_variance_ = eigenvalues_
        explained_variance_ratio_ = [value/np.sum(explained_variance_) for value in explained_variance_]
        eigenvalues_ = eigenvalues_[:self.n_components]
        explained_variance_ = explained_variance_[:self.n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        return U, Sigma, V, components_, eigenvalues_, explained_variance_, explained_variance_ratio_
    
    def svd_flip(self, U, V):
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]
        return U, V
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('SVD has to be fitted first.')