from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, random_state=17):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
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
        cluster_centers_ = self.get_best_init_centers(X=X, n_clusters=self.n_clusters, n_init=self.n_init)
        self.cluster_centers_ = self.optimize_cluster_centers(X=X, cluster_centers=cluster_centers_, max_iter=self.max_iter, tol=self.tol)
        self.labels_ = self.find_labels(X=X, cluster_centers=self.cluster_centers_)
        self.fit_used = True
    
    def fit_transform(self, X):
        X = self.check_data(X=X)
        X = self.check_for_object_columns(X=X)
        cluster_centers_ = self.get_best_init_centers(X=X, n_clusters=self.n_clusters, n_init=self.n_init)
        self.cluster_centers_ = self.optimize_cluster_centers(X=X, cluster_centers=cluster_centers_, max_iter=self.max_iter, tol=self.tol)
        self.labels_ = self.find_labels(X=X, cluster_centers=self.cluster_centers_)
        self.fit_used = True
        return self.euclidean_distance(X_1=X, X_2=self.cluster_centers_)
    
    def transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_data(X=X)
        X = self.check_for_object_columns(X=X)
        return self.euclidean_distance(X_1=X, X_2=self.cluster_centers_)
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_data(X=X)
        X = self.check_for_object_columns(X=X)
        return self.find_labels(X=X, cluster_centers=self.cluster_centers_)
    
    def get_best_init_centers(self, X, n_clusters, n_init):
        best_sse = np.inf
        for i in range(0, n_init):
            current_cluster_centers = self.initialize_cluster_centers(X=X, n_clusters=n_clusters)
            labels = self.find_labels(X=X, cluster_centers=current_cluster_centers)
            current_sse = self.calculate_sse(X=X, cluster_centers=current_cluster_centers, labels=labels)
            if(current_sse < best_sse):
                best_sse = current_sse
                best_cluster_centers = current_cluster_centers
        return best_cluster_centers
    
    def initialize_cluster_centers(self, X, n_clusters):
        min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
        return np.array([np.random.uniform(low=min_, high=max_, size=(X.shape[1])) for i in range(n_clusters)])
    
    def calculate_sse(self, X, cluster_centers, labels):
        sse = 0
        for cluster in np.unique(labels):
            X_cluster = X[np.where(labels == cluster)]
            sse += np.sum(self.squared_euclidean_distance(X_1=X_cluster, X_2=cluster_centers[cluster]))
        return sse
    
    def find_labels(self, X, cluster_centers):
        distances = self.squared_euclidean_distance(X_1=X, X_2=cluster_centers)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def squared_euclidean_distance(self, X_1, X_2):
        #Squared distances, so without sqrt
        return np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2)

    def optimize_cluster_centers(self, X, cluster_centers, max_iter, tol):
        for i in range(0, max_iter):
            labels = self.find_labels(X=X, cluster_centers=cluster_centers)
            new_cluster_centers = self.calculate_new_cluster_centers(X=X, labels=labels)
            if(np.abs(np.sum(new_cluster_centers) - np.sum(cluster_centers)) < tol):
                break
            cluster_centers = new_cluster_centers
        return cluster_centers
    
    def calculate_new_cluster_centers(self, X, labels):
        new_cluster_centers = []
        for cluster in np.unique(labels):
            X_cluster = X[np.where(labels == cluster)]
            new_cluster_centers.append(np.mean(X_cluster, axis=0))
        return np.array(new_cluster_centers)

    def euclidean_distance(self, X_1, X_2):
        return np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2))
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('KMeans has to be fitted first.')
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self