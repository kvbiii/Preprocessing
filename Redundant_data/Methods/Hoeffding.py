from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Hoeffding:
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
        ranked_bivariate = self.rankdata_bivariate(data1=X, data2=y)
        D1 = np.sum((ranked_bivariate-1)*(ranked_bivariate-2))
        D2 = np.sum((ranked_X-1)*(ranked_X-2)*(ranked_y-1)*(ranked_y-2))
        D3 = np.sum((ranked_X-2)*(ranked_y-2)*(ranked_bivariate-1))
        self.hoeffding_distance_ = self.calculate_hoeffding_distance(D1=D1, D2=D2, D3=D3, N=X.shape[0])
        self.test_statistic_ = self.calculate_test_statistic(D=self.hoeffding_distance_, N=X.shape[0])
        self.p_value_ = self.calculate_p_value_hoeffding(N=X.shape[0], alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
    
    def rankdata_bivariate(self, data1, data2):
        i = 0
        ranks = [0.75 for i in range(0, len(data1))]
        while(i < len(data1)):
            j = 0
            while(j < len(data1)):
                ranks[i] += self.check_conditions(u=data1[i] - data1[j]) * self.check_conditions(u=data2[i] - data2[j])
                j += 1
            i = i + 1
        return np.array(ranks)
    
    def check_conditions(self, u):
        if(u > 0):
            return 1
        elif(u == 0):
            return 1/2
        else:
            return 0
    
    def calculate_hoeffding_distance(self, D1, D2, D3, N):
        return 30*((N-2)*(N-3)*D1+D2-2*(N-2)*D3)/(N*(N-1)*(N-2)*(N-3)*(N-4))

    def calculate_test_statistic(self, D, N):
        return (N-1)*np.pi**4/60*D+np.pi**4/72
    
    def calculate_p_value_hoeffding(self, N, alpha):
        #Only for N<=10 
        return np.sqrt((2*(N**2+5*N-32))/(9*N*(N-1)*(N-3)*(N-4)*alpha))
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False