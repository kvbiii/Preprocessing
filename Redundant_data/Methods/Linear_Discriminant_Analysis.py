from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Linear_Discriminant_Analysis:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

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
    
    def fit(self, X, y):
        X = self.check_data(X)
        X = self.check_for_object_columns(X)
        self.n_features_in_ = X.shape[1]
        y = self.check_data(y)
        self.check_n_of_components(X)
        self.classes_ = np.unique(y)
        self.means_ = self.calculate_means(X=X, y=y, unique_classes=self.classes_)
        S = self.calculate_within_class_scatter_matrix(X=X, y=y, unique_classes=self.classes_)
        B = self.calculate_between_class_scatter_matrix(X=X, y=y, unique_classes=self.classes_)
        self.eigen_values_, self.eigen_vectors_ = self.calculate_eigen_values_and_vectors(S=S, B=B)
    
    def fit_transform(self, X, y):
        self.fit(X=X, y=y)
        return self.transform(X=X)
    
    def transform(self, X):
        X = self.check_data(X)
        X = self.check_for_object_columns(X)
        self.check_n_of_components(X)
        transformed = np.dot(X, self.eigen_vectors_[:, :self.n_components])
        return transformed
    
    def calculate_means(self, X, y, unique_classes):
        return [np.mean(X[y==class_], axis=0) for class_ in unique_classes]
    
    def calculate_within_class_scatter_matrix(self, X, y, unique_classes):
        S = np.zeros((self.n_features_in_, self.n_features_in_))
        for class_, mean_vector in zip(unique_classes, self.means_):
            class_sc_mat = np.zeros((self.n_features_in_, self.n_features_in_))
            for row in X[y==class_]:
                row, mean_vector = row.reshape(self.n_features_in_, 1), mean_vector.reshape(self.n_features_in_, 1)
                class_sc_mat += np.dot((row-mean_vector), (row-mean_vector).T)
            S += class_sc_mat
        return S
    
    def calculate_between_class_scatter_matrix(self, X, y, unique_classes):
        data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
        B = np.zeros((self.n_features_in_, self.n_features_in_))
        for i, mean_vector in enumerate(self.means_):
            N_k = X[y==unique_classes[i]].shape[0]
            mean_vector = mean_vector.reshape(1, self.n_features_in_)
            B += N_k * np.dot((mean_vector - data_mean).T, (mean_vector - data_mean))
        return B
    
    def calculate_eigen_values_and_vectors(self, S, B):
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S).dot(B))
        eigen_vectors = eigen_vectors
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        return np.real(eigen_values), np.real(eigen_vectors)