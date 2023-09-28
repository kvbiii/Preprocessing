from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Agglomerative_Clustering:
    def __init__(self, metric='euclidean', linkage='ward', random_state=17):
        metrics = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_similarity
        }
        self.metric = metrics[metric]
        linkage_methods = {
            'single': self.single_method,
            'complete': self.complete_method,
            'average': self.average_method,
            'ward': self.ward_method
        }
        self.linkage = linkage_methods[linkage]
        self.random_state = random_state
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
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
        self.children_, self.distances_ = self.build_dendrogram(X=[np.expand_dims(X[i, :], axis=0) for i in range(0, X.shape[0])], N=X.shape[0], clusters_indices=[i for i in range(0, X.shape[0])], children_=[], distances_=[], iteration=0)

    def build_dendrogram(self, X, N, clusters_indices, children_, distances_, iteration):
        best_cluster_indices, best_distance = self.find_best_cluster(X=X, clusters_indices=clusters_indices)
        X, clusters_indices, children_, distances_ = self.transform_variables(X=X, clusters_indices=clusters_indices, children_=children_, distances_=distances_, N=N, iteration=iteration, best_cluster_indices=best_cluster_indices, best_distance=best_distance)
        if(len(clusters_indices) > 1):
            self.build_dendrogram(X=X, N=N, clusters_indices=clusters_indices, children_=children_, distances_=distances_, iteration=iteration+1)
        return np.array(children_), np.array(distances_)

    def find_best_cluster(self, X, clusters_indices):
        best_distance = np.inf
        for idx_i in clusters_indices:
            for idx_j in clusters_indices:
                if(idx_i != idx_j):
                    distance = self.linkage(X_1=X[idx_i], X_2=X[idx_j])
                    if(distance < best_distance):
                        best_distance = distance
                        best_cluster_indices = idx_i, idx_j
        return best_cluster_indices, best_distance

    def single_method(self, X_1, X_2):
        return np.min(self.metric(X_1=X_1, X_2=X_2))
    
    def complete_method(self, X_1, X_2):
        return np.max(self.metric(X_1=X_1, X_2=X_2))
    
    def average_method(self, X_1, X_2):
        return np.mean(self.metric(X_1=X_1, X_2=X_2))
    
    def ward_method(self, X_1, X_2):
        #Check if this is initial observations clustering
        if(len(X_1.shape) == 1 and len(X_2.shape) == 1):
            return self.metric(X_1=X_1, X_2=X_2)[0][0]
        else:
            n_A  = X_1.shape[0]*X_1.shape[1]
            n_B = X_2.shape[0]*X_2.shape[1]
            center_A = np.mean(X_1, axis=0)
            center_B = np.mean(X_2, axis=0)
            return np.sqrt(n_A*n_B/(n_A+n_B)*np.linalg.norm(center_A-center_B)**2)
    
    def euclidean_distance(self, X_1, X_2):
        return np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2))
    
    def manhattan_distance(self, X_1, X_2):
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)
    
    def cosine_similarity(self, X_1, X_2):
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)/(np.sum(X_1[:,np.newaxis])*np.sum(X_2))

    def transform_variables(self, X, clusters_indices, children_, distances_, N, iteration, best_cluster_indices, best_distance):
        X.append(np.vstack((X[best_cluster_indices[0]], X[best_cluster_indices[1]])))
        clusters_indices.remove(best_cluster_indices[0])
        clusters_indices.remove(best_cluster_indices[1])
        clusters_indices.append(N+iteration)
        children_.append(list(best_cluster_indices))
        distances_.append(best_distance)
        return X, clusters_indices, children_, distances_