from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class PCA():
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
        self.mean_ = np.mean(self.fit_data, axis=0)
        #Center data
        self.fit_data = self.fit_data-self.mean_
        self.check_n_of_components(data=self.fit_data)
        covariance = self.calculate_covariance(data=self.fit_data)
        self.eigenvalues_, self.eigenvectors_ = self.calculate_eigenvalues_and_eigenvectors(covariance=covariance)
        self.fit_used = True

    def fit_transform(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.mean_ = np.mean(self.fit_data, axis=0)
        #Center data
        self.fit_data = self.fit_data-self.mean_
        self.check_n_of_components(data=self.fit_data)
        covariance = self.calculate_covariance(data=self.fit_data)
        self.eigenvalues_, self.eigenvectors_ = self.calculate_eigenvalues_and_eigenvectors(covariance=covariance)
        self.fit_used = True
        return np.dot(data, self.eigenvectors_)
    
    def transform(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.dot(data, self.eigenvectors_)
    
    def calculate_covariance(self, data):
        return 1/(data.shape[0]-1)*(np.dot(data.T, data))
    
    def calculate_eigenvalues_and_eigenvectors(self, covariance):
        #Below returns good eigenvalues but eigenvectors does not have good signs.
        #eigenvalues, eigenvectors = np.linalg.eig(covariance)
        unitary, singular_values, eigenvectors = np.linalg.svd(self.fit_data, full_matrices=False)
        unitary, eigenvectors = self.svd_flip(unitary, eigenvectors)
        eigenvalues = singular_values**2/(self.fit_data.shape[0]-1)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[sorted_idx])
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = [value/np.sum(self.explained_variance_) for value in self.explained_variance_]
        eigenvalues = eigenvalues[:self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        eigenvectors = np.real(eigenvectors[:, sorted_idx[:self.n_components]])
        self.components_ = eigenvectors
        return eigenvalues, eigenvectors
    
    def svd_flip(self, u, v):
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('PCA has to be fitted first.')