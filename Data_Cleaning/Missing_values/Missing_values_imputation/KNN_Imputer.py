from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class KNN_Imputer():
    def __init__(self, n_neighbors=20, weights="uniform", distance="nan_euclidean", random_state=17):
        self.n_neighbors = n_neighbors
        weight_types = {
            "uniform": self.uniform,
            "weighted": self.weighted
        }
        self.weights = weight_types[weights]
        distances = {
            'nan_euclidean': self.euclidean_distance,
        }
        self.distance = distances[distance]
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(X)
    
    def check_for_object_columns(self, X):
        X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def fit(self, X):
        self.X_train = self.check_X(X=X)
        self.X_train  = self.check_for_object_columns(X=self.X_train )
        self.fit_used = True
    
    def fit_transform(self, X, categorical_features=[]):
        self.X_train = self.check_X(X=X)
        self.X_train = self.check_for_object_columns(X=X)
        distances = self.distance(X_1=self.X_train, X_2=self.X_train)
        k_closest_indices = self.find_k_closest_indices(distances=distances)
        X_imputed = self.impute_missing_values(X=self.X_train, k_closest_indices=k_closest_indices)
        self.X_train = X_imputed
        self.fit_used = True
        return X_imputed
    
    def transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        distances = self.distance(X_1=self.X_train, X_2=X)
        k_closest_indices = self.find_k_closest_indices(distances=distances)
        X_imputed = self.impute_missing_values(X=X, k_closest_indices=k_closest_indices)
        return X_imputed
    
    def euclidean_distance(self, X_1, X_2):
        return nan_euclidean_distances(X_1, X_2)
    
    def find_k_closest_indices(self, distances):
        #We start the selection with the second (index 1), because the first one is the same observation for which distance is equal to 0
        return np.argsort(distances)[:,1:self.n_neighbors+1]
    
    def impute_missing_values(self, X, k_closest_indices):
        rows_with_missings = np.where(np.isnan(X).any(axis=1))[0]
        for row in rows_with_missings:
            indices_to_be_imputed = np.where(np.isnan(X[row,:]))[0]
            neighbors = self.X_train[k_closest_indices[row,:], :]
            values_of_neighbors_to_be_imputed = neighbors[:, indices_to_be_imputed]
            X[row, indices_to_be_imputed] = self.weights(values=values_of_neighbors_to_be_imputed)
        return X
    
    def uniform(self, values):
        return np.nanmean(values, axis=0)

    def weighted(self, values):
        values = np.ma.MaskedArray(values, mask=np.isnan(values))
        return np.average(values, axis=0, weights=[i for i in range(self.n_neighbors, 0, -1)])

    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('KNN_Imputer has to be fitted first.')