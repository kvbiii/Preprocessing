from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

from scipy.stats import norm
class Qunatile_Transformer():
    def __init__(self, distribution="normal"):
        self.distribution = distribution
        self.check_distribution(distribution=self.distribution)
        self.fit_used = False
    
    def check_distribution(self, distribution):
        if not distribution in ["normal", "uniform"]:
            raise ValueError('Wrong value for distribution. It should be `normal` or `uniform`.')

    def fit(self, X):
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        self.ecdf_instances = self.get_ecdf_instances(self, X=X)
        self.fit_used = True
    
    def fit_transform(self, X):
        X = self.check_X(X=X)
        X = self.check_for_object_columns(X=X)
        self.ecdf_instances = self.get_ecdf_instances(X=X)
        empirical_cdf_data = self.calculate_empirical_cdf(X=X)
        self.fit_used = True
        return empirical_cdf_data
    
    def get_ecdf_instances(self, X):
        return [ECDF(X[:, m]) for m in range(0, X.shape[1])]
    
    def calculate_empirical_cdf(self, X):
        empirical_cdf = np.array([self.ecdf_instances[m](X[:, m]) for m in range (0, X.shape[1])]).T
        if(self.distribution == "normal"):
            empirical_cdf = norm.ppf(empirical_cdf, loc=0, scale=1)
        #Avoid inf values
        empirical_cdf = np.array([[element if element!=np.inf else empirical_cdf[:,m][np.isfinite(empirical_cdf[:,m])].max() for element in empirical_cdf[:,m]] for m in range(0, X.shape[1])]).T
        return empirical_cdf
    
    def transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        empirical_cdf_data = self.calculate_empirical_cdf(X=X)
        return empirical_cdf_data
    
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
            raise AttributeError('Quantile_Transformer has to be fitted first.')