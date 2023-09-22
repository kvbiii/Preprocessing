from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class VIF():
    def __init__(self, estimator):
        self.estimator = estimator

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        self.features_names = X.columns.tolist()
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X

    def fit(self, X):
        X = self.check_X(X=X)
        self.r_squared_ = []
        for feature in range(0, X.shape[1]):
            self.r_squared_.append(self.calculate_r_squared(X=X, feature=feature))
        self.vif_statistic = self.calculate_vif_statistic(r_squared=self.r_squared_)
        self.summary_ = pd.DataFrame({'Feature name': self.features_names, 'VIF': self.vif_statistic})
    
    def calculate_r_squared(self, X, feature):
        y = X[:, feature]
        X = np.delete(X, feature, axis=1)
        self.estimator.fit(X, y)
        y_pred = self.estimator.predict(X)
        RSS = np.sum((y-y_pred)**2)
        TSS = np.sum((y-np.mean(y))**2)
        return 1-RSS/TSS
    
    def calculate_vif_statistic(self, r_squared):
        return [1/(1-value) for value in r_squared]