from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class DBSCAN:
    def __init__(self, epsilon, min_points, random_state=17):
        self.epsilon = epsilon
        self.min_points = min_points
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
    def check_data(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(X)
    
    def check_for_object_columns(self, X):
        X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def fit(self, X):
        X = self.check_data(X=X)
        X = self.check_for_object_columns(X=X)
        self.labels_ = self.dbscan_process(data=X, epsilon=self.epsilon, min_points=self.min_points)
    
    def dbscan_process(self, data, epsilon, min_points):
        labels = np.array([-1 for i in range(0, data.shape[0])])
        cluster_label = 0
        while True:
            start_point = self.find_start_point(data=data, labels=labels, epsilon=epsilon, min_points=min_points)
            if(start_point == None):
                break
            labels[start_point] = cluster_label
            labels = self.build_cluster(data=data, core_point=start_point, labels=labels, cluster_label=cluster_label, epsilon=epsilon, min_points=min_points)
            cluster_label += 1
        return labels

    def find_start_point(self, data, labels, epsilon, min_points):
        unlabaled_indices = np.where(labels == -1)[0]
        random_sequence_of_observations = random.sample([i for i in unlabaled_indices], unlabaled_indices.shape[0])
        start_point = None
        for i in random_sequence_of_observations:
            if(self.check_if_core_point(observation=data[i], data=data, epsilon=epsilon, min_points=min_points)):
                start_point = i
                break
        return start_point

    def euclidean_distance(self, X_1, X_2):
        return np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2))

    def check_if_core_point(self, observation, data, epsilon, min_points):
        if(np.sum(self.euclidean_distance(np.expand_dims(observation, axis=0), data) <= epsilon) >= min_points):
            return True
        else:
            return False
    
    def build_cluster(self, data, core_point, labels, cluster_label, epsilon, min_points):
        neighbors_indices = self.find_directly_density_reachable(core_point=data[core_point], data=data, epsilon=epsilon)
        neighbors_indices_unvisited = np.intersect1d(neighbors_indices, np.where(labels == -1)[0])
        labels[neighbors_indices_unvisited] = cluster_label
        for i in neighbors_indices_unvisited:
            if(self.check_if_core_point(observation=data[i], data=data, epsilon=epsilon, min_points=min_points) == True):
                self.build_cluster(data=data, core_point=i, labels=labels, cluster_label=cluster_label, epsilon=epsilon, min_points=min_points)
        return labels

    def find_directly_density_reachable(self, core_point, data, epsilon):
        return np.where(self.euclidean_distance(np.expand_dims(core_point, axis=0), data)[0] <= epsilon)[0]