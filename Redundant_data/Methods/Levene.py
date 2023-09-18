from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Levene:
    def __init__(self, center="median"):
        centers = {
            'mean': self.calculate_mean,
            'median': self.calculate_median
        }
        self.center=centers[center]
        
    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 2):
            X = X.squeeze()
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        return y

    def fit(self, X, y, alpha=0.05):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        y_divided_into_groups = self.divide_into_groups(X=X, y=y)
        Z_c_i, Z_c_mean, Z_c_i_mean = self.calculate_Z_c_i(y=y_divided_into_groups)
        self.test_statistic_ = self.calculate_W_statistic(y=y_divided_into_groups, Z_c_i=Z_c_i, Z_c_mean=Z_c_mean, Z_c_i_mean=Z_c_i_mean)
        self.p_value_ = self.calculate_p_value_F_test(F_test=self.test_statistic_, dfn=len(y_divided_into_groups)-1, dfd=X.shape[0]-len(y_divided_into_groups))
        self.critical_value_ = self.calculate_critical_value(dfn=len(y_divided_into_groups)-1, dfd=X.shape[0]-len(y_divided_into_groups), alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
    
    def divide_into_groups(self, X, y):
        return [y[X == value].flatten().tolist() for value in np.unique(X)]
    
    def calculate_mean(self, data):
        return np.mean(data)
    
    def calculate_median(self, data):
        return np.median(data)
    
    def calculate_Z_c_i(self, y):
        Z_c_i = []
        Z_c_mean = []
        C = len(y)
        for category in range(0, C):
            Z_c_i.append(np.abs(y[category]-self.center(data=y[category]), dtype=np.float64).flatten().tolist())
            Z_c_mean.append(np.mean(Z_c_i[category]))
        flat_list = [item for row in Z_c_i for item in row]
        Z_c_i_mean = np.mean(flat_list)
        return Z_c_i, Z_c_mean, Z_c_i_mean
    
    def calculate_W_statistic(self, y, Z_c_i, Z_c_mean, Z_c_i_mean):
        nominator = 0
        denominator = 0
        N = 0
        C = len(y)
        for category in range(0, C):
            N += len(y[category])
            nominator += len(y[category])*((Z_c_mean[category]-Z_c_i_mean)**2)
            denominator += np.sum((Z_c_i[category]-Z_c_mean[category])**2)
        return (N-C)/(C-1)*nominator/denominator
    
    def calculate_p_value_F_test(self, F_test, dfn, dfd):
        return 1 - f.cdf(F_test, dfn, dfd)
    
    def calculate_critical_value(self, dfn, dfd, alpha):
        return f.isf(q=alpha, dfn=dfn, dfd=dfd)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False