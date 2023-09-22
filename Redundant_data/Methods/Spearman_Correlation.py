from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Spearman_Correlation:
    def __init__(self,):
        pass
        
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
        ranked_X = rankdata(X)
        ranked_y = rankdata(y)
        covariance = self.calculate_covariance(X=ranked_X, y=ranked_y)
        stdX = self.calculate_std(data=ranked_X)
        stdY = self.calculate_std(data=ranked_y)
        self.correlation_ = self.calculate_correlation(covariance=covariance, stdX=stdX, stdY=stdY)
        self.test_statistic_ = self.calculate_test_statistic(correlation=self.correlation_, N=X.shape[0])
        self.p_value_ = self.calculate_p_value_t_test(t_test=self.test_statistic_, df=X.shape[0]-2)
        self.critical_value_ = self.calculate_critical_value(df=X.shape[0]-2, alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
    
    def calculate_covariance(self, X, y):
        return np.cov(X, y)
    
    def calculate_std(self, data):
        return np.std(data)
    
    def calculate_correlation(self, covariance, stdX, stdY):
        return (covariance/(stdX*stdY))[1][0]
    
    def calculate_test_statistic(self, correlation, N):
        return (correlation*np.sqrt(N-2))/np.sqrt(1-correlation**2)
    
    def calculate_p_value_t_test(self, t_test, df):
        return 2*(1 - t.cdf(np.abs(t_test), df))
    
    def calculate_critical_value(self, df, alpha):
        return t.isf(q=alpha/2, df=df)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False