from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Isolation_Forest():
    def __init__(self, threshold=0.05, n_estimators=100, max_samples=256, max_features=None, bootstrap=True, random_state=17):
        self.threshold = threshold
        self.check_threshold(threshold=self.threshold)
        self.n_estimators = n_estimators
        self.check_n_estimators(n_estimators=self.n_estimators)
        self.max_samples = max_samples
        self.check_max_samples(max_samples=self.max_samples)
        self.max_features = max_features
        self.check_max_features(max_features=self.max_features)
        self.bootstrap = bootstrap
        self.check_bootstrap(bootstrap=self.bootstrap)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_threshold(self, threshold):
        if not isinstance(threshold, float):
            raise TypeError('Wrong type of threshold. It should be float.')
    
    def check_n_estimators(self, n_estimators):
        if not isinstance(n_estimators, int):
            raise TypeError('Wrong type of n_estimators. It should be int.')
    
    def check_max_samples(self, max_samples):
        if not isinstance(max_samples, int) and max_samples!="auto":
            raise TypeError('Wrong type of max_samples. It should be int or string: `auto`')
    
    def check_max_features(self, max_features):
        if not isinstance(max_features, int) and max_features!="log2" and max_features!="sqrt" and max_features != None:
            raise TypeError("Wrong type of max_features. It should be int, or string: `log2` or `sqrt`.")
    
    def check_bootstrap(self, bootstrap):
        if not isinstance(bootstrap, bool):
            raise TypeError('Wrong type of bootstrap. It should be True or False')
    
    def fit(self, X):
        self.X_train = self.check_X(X=X, train=True)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.trees = []
        self.max_depth = math.ceil(np.log2(self.max_samples))
        for iter in range(0, self.n_estimators):
            X_bootstrap = self.get_bootstrap_data(X=self.X_train)
            self.tree = self.build_isolation_tree(X=X_bootstrap, depth=0)
            self.trees.append(self.tree)
        self.fit_used = True
    
    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if(self.X_train.shape[1] != X.shape[1]):
                raise ValueError(f"X has {X.shape[1]} features, but Isolation_Forest is expecting {self.X_train.shape[1]} features as input.")
        return X
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def get_bootstrap_data(self, X):
        if(self.bootstrap == True):
            indices = np.random.choice(X.shape[0], size=self.max_samples, replace=True)
            X_bootstrap = X[indices]
            return X_bootstrap
        else:
            return X
    
    def build_isolation_tree(self, X, depth):
        if(depth < self.max_depth and len(np.unique(X, axis=0, return_counts=True)[1]) > 1):
            best_split = self.find_best_split(X=X)
            true_branch = self.build_isolation_tree(X=X[best_split["true_rows"], :], depth=depth+1)
            false_branch = self.build_isolation_tree(X=X[best_split["false_rows"], :], depth=depth+1)
            return Decision_Node(true_branch=true_branch, false_branch=false_branch, feature=best_split["feature"], interpolation_value=best_split["interpolation_value"])
        return Leaf(n = X.shape[0])

    def find_best_split(self, X):
        ratio_best = 1
        best_split = {"feature": None, "true_rows": None, "false_rows": None, "interpolation_value": None}
        limited_columns = self.get_limited_columns(X=X)
        for feature in limited_columns:
            interpolation_values = self.find_interpolation_values(X=X[:,feature])
            for interpolation_value in interpolation_values:
                true_rows, false_rows = self.partition(X=X, feature=feature, interpolation_value=interpolation_value)
                # Skip this split if it doesn't divide the dataset. If it will happen for all the columns then it means that node is a leaf.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                # Calculate the ratio_unbalancedness of the split.
                ratio_unbalancedness = min(len(true_rows)/(len(true_rows) + len(false_rows)), len(false_rows)/(len(true_rows) + len(false_rows)))
                if ratio_unbalancedness < ratio_best:
                    ratio_best = ratio_unbalancedness
                    best_split["feature"] = feature
                    best_split["true_rows"] = true_rows
                    best_split["false_rows"] = false_rows
                    best_split["interpolation_value"] = interpolation_value
        return best_split
    
    def get_limited_columns(self, X):
        upper_bound = {None: random.sample([i for i in range(0, X.shape[1])], int(X.shape[1])),
                        "sqrt": random.sample([i for i in range(0, X.shape[1])], int(X.shape[1]**0.5)),
                        "log2": random.sample([i for i in range(0, X.shape[1])], int(np.log2(X.shape[1]))),
                        }
        try:
            return upper_bound[self.max_features]
        #Except if self.max_features is int.
        except:
            if(self.max_features <= X.shape[1]):
                return random.sample([i for i in range(0, X.shape[1])], self.max_features)
            else:
                raise ValueError("max_features cannot be bigger than number of columns in X.")
    
    def find_interpolation_values(self, X):
        return np.random.uniform(min(X), max(X), 10)
    
    def partition(self, X, feature, interpolation_value):
        return np.where(np.array(X)[:,feature] <= interpolation_value)[0], np.where(np.array(X)[:,feature] > interpolation_value)[0]

    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        scores = self.anomaly_score(X)
        prediction = self.compare_with_threshold(anomaly_scores=scores)
        return prediction

    def score_samples(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        scores = self.anomaly_score(X)
        return scores

    def find_indices_of_outliers(self, X):
        scores = self.score_samples(X=X)
        return np.where(scores >= self.threshold)[0]

    def anomaly_score(self, X):
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/self.c(self.X_train.shape[0]))for l in avg_length])
        return scores

    def path_length(self, X):
        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree
                length = 0
                while isinstance(node, Decision_Node):
                    if row[node.feature] <= node.interpolation_value:
                        node = node.true_branch
                    else:
                        node = node.false_branch
                    length += 1
                leaf_size = node.n
                pathLength = length + self.c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def c(self, n):
        if(n >= 2):
            return 2*(np.log(n-1)+0.5772156649)-2*(n-1)/n
        else:
            return 0

    def compare_with_threshold(self, anomaly_scores):
        return np.array([1 if score >= self.threshold else -1 for score in anomaly_scores])
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Isolation_Forest has to be fitted first.')

class Leaf:
    def __init__(self, n):
        self.n = n

class Decision_Node:
    def __init__(self, true_branch, false_branch, feature, interpolation_value):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.feature = feature
        self.interpolation_value = interpolation_value