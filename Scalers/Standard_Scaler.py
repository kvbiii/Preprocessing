from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Standard_Scaler():
    def __init__(self):
        self.fit_used = False

    def fit(self, X):
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.fit_used = True
    
    def fit_transform(self, X):
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.fit_used = True
        return (X-self.mean)/self.std
    
    def transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        return (X-self.mean)/self.std
    
    def inverse_transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        return X*self.std.T+self.mean.T
    
    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Standard_Scaler has to be fitted first.')