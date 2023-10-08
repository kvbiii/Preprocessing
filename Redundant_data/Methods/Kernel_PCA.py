from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Kernel_PCA():
    def __init__(self, n_components=5, kernel="linear", gamma=1, degree=2, tol=1e-7, random_state=17):
        self.n_components = n_components
        kernels = {
            "linear": self.kernel_linear,
            "poly": self.kernel_poly,
            "rbf": self.kernel_rbf
        }
        self.calculate_kernel = kernels[kernel]
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
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
        kernel = self.calculate_kernel(X_1=self.fit_data, X_2=self.fit_data)
        kernel = self.center_kernel(kernel=kernel, train=True)
        self.eigenvalues_, self.eigenvectors_ = self.calculate_eigenvalues_and_eigenvectors(kernel=kernel)
        self.fit_used = True

    def fit_transform(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        kernel = self.calculate_kernel(X_1=self.fit_data, X_2=self.fit_data)
        kernel = self.center_kernel(kernel=kernel, train=True)
        self.eigenvalues_, self.eigenvectors_ = self.calculate_eigenvalues_and_eigenvectors(kernel=kernel)
        self.fit_used = True
        return np.dot(kernel, self.eigenvectors_)
    
    def transform(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        kernel = self.calculate_kernel(X_1=data, X_2=self.fit_data)
        kernel = self.center_kernel(kernel=kernel, train=False)
        return np.dot(kernel, self.eigenvectors_)
    
    def kernel_linear(self, X_1, X_2):
        return np.array([[X_1[row]@X_2[col] for row in range(0, X_1.shape[0])] for col in range(0, X_2.shape[0])])
    
    def kernel_poly(self, X_1, X_2):
        return np.array([[(self.gamma*X_1[row]@X_2[col]+1)**self.degree for row in range(0, X_1.shape[0])] for col in range(0, X_2.shape[0])])
    
    def kernel_rbf(self, X_1, X_2):
        return np.array([[np.exp(-self.gamma*np.linalg.norm([X_1[row]-X_2[col]])**2) for row in range(0, X_1.shape[0])] for col in range(0, X_2.shape[0])])
    
    def center_kernel(self, kernel, train):
        #Below works with linear and poly kernel. I do not know why it does not work with rbf...
        #ones = np.ones(shape=kernel.shape)/kernel.shape[0]
        #return kernel - np.dot(ones, kernel) - np.dot(kernel, ones) + np.dot(ones, np.dot(kernel, ones))
        if(train == True):
            self.kernel_centerer = KernelCenterer()
            return self.kernel_centerer.fit_transform(kernel)
        else:
            return self.kernel_centerer.transform(kernel.T)
    
    def calculate_eigenvalues_and_eigenvectors(self, kernel):
        v0 = np.random.uniform(-1, 1, kernel.shape[0])
        eigenvalues, eigenvectors = eigsh(A=kernel, k=self.fit_data.shape[1], which="LA", tol=0, maxiter=None, v0=v0)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[sorted_idx])
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = [value/np.sum(self.explained_variance_) for value in self.explained_variance_]
        eigenvalues = eigenvalues[:self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
        eigenvectors = np.real(eigenvectors[:, sorted_idx[:self.n_components]])
        normalized_eigenvectors = eigenvectors/np.sqrt(eigenvalues)
        self.components_ = normalized_eigenvectors
        return eigenvalues, normalized_eigenvectors
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Kernel_PCA has to be fitted first.')