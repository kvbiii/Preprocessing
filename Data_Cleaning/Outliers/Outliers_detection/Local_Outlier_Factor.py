from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Local_Outlier_Factor():
    def __init__(self, n_neighbors=20, distance="euclidean", random_state=17):
        self.n_neighbors = n_neighbors
        distances = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_similarity
        }
        self.distance = distances[distance]
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def fit(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        distances = self.distance(X_1=data, X_2=data)
        k_closest_indices = self.find_k_closest_indices(distances=distances)
        reachability_density = self.calculate_reachability_density(distances=distances)
        local_reachability_density = self.calculate_local_reachability_density(distances=distances)
        self.local_outlier_factor_ = self.calculate_local_outlier_factor(local_reachability_density=local_reachability_density, k_closest_indices=k_closest_indices)
        self.fit_used = True

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)
    
    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)

    def euclidean_distance(self, X_1, X_2):
        return np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2))
    
    def manhattan_distance(self, X_1, X_2):
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)
    
    def cosine_similarity(self, X_1, X_2):
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)/(np.sum(X_1[:,np.newaxis])*np.sum(X_2))
    
    def find_k_closest_indices(self, distances):
        #We start the selection with the second (index 1), because the first one is the same observation for which distance is equal to 0
        return np.argsort(distances)[:,1:self.n_neighbors+1]
    
    def calculate_reachability_density(self, distances):
        return np.sort(distances)[:, self.n_neighbors]

    def calculate_local_reachability_density(self, distances):
        return self.n_neighbors/np.sum(np.sort(distances)[:, 1:self.n_neighbors+1], axis=1)
    
    def calculate_local_outlier_factor(self, local_reachability_density, k_closest_indices):
        return np.sum(local_reachability_density[k_closest_indices], axis=1)/self.n_neighbors*1/local_reachability_density

    def remove_outliers(self, data, local_outlier_factor_to_compared):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        indices_of_outliers = self.find_indices_of_outliers(local_outlier_factor_to_compared=local_outlier_factor_to_compared)
        return np.delete(data, indices_of_outliers)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Local_Outlier_Factor has to be fitted first.')
    
    def find_indices_of_outliers(self, local_outlier_factor_to_compared):
        self.check_fit(fit_used=self.fit_used)
        return np.where(self.local_outlier_factor_ > np.quantile(local_outlier_factor_to_compared, q=0.95))[0]
