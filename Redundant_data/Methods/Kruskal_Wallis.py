from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Kruskal_Wallis:
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
        y_ranked = rankdata(y)
        y_ranked_divided_into_groups = self.divide_into_groups(X=X, y=y_ranked)
        r_c_mean, r_all_mean = self.calculate_rank_means(y_ranked=y_ranked_divided_into_groups)
        self.test_statistic_ = self.calculate_H_statistic(y_ranked=y_ranked_divided_into_groups, r_c_mean=r_c_mean, r_all_mean=r_all_mean)
        self.p_value_ = self.calculate_p_value_F_test(F_test=self.test_statistic_, dfn=len(y_ranked_divided_into_groups)-1, dfd=X.shape[0]-len(y_ranked_divided_into_groups))
        self.critical_value_ = self.calculate_critical_value(dfn=len(y_ranked_divided_into_groups)-1, dfd=X.shape[0]-len(y_ranked_divided_into_groups), alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
    
    def divide_into_groups(self, X, y):
        return [y[X == value].flatten().tolist() for value in np.unique(X)]
    
    def calculate_rank_means(self, y_ranked):
        r_c_mean = []
        C = len(y_ranked)
        for category in range(0, C):
            r_c_mean.append(np.mean(y_ranked[category]))
        flat_list = [item for row in y_ranked for item in row]
        r_all_mean = np.mean(flat_list)
        return r_c_mean, r_all_mean
    
    def calculate_H_statistic(self, y_ranked, r_c_mean, r_all_mean):
        nominator = 0
        denominator = 0
        N = 0
        C = len(y_ranked)
        for category in range(0, C):
            N += len(y_ranked[category])
            nominator += len(y_ranked[category])*((r_c_mean[category]-r_all_mean)**2)
            denominator += np.sum((y_ranked[category]-r_all_mean)**2)
        return (N-1)*nominator/denominator
    
    def calculate_p_value_F_test(self, F_test, dfn, dfd):
        return 1 - f.cdf(F_test, dfn, dfd)
    
    def calculate_critical_value(self, dfn, dfd, alpha):
        return f.isf(q=alpha, dfn=dfn, dfd=dfd)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False