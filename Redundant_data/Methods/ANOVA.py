from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class ANOVA():
    def __init__(self):
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
        crosstab_means, crosstab_frequency = self.crosstab_creation(X=X, y=y)
        Sa_2 = self.calculate_Sa_squared(y=y, crosstab_means=crosstab_means, crosstab_frequency=crosstab_frequency)
        Se_2 = self.calculate_Se_squared(y=y, crosstab_means=crosstab_means, crosstab_frequency=crosstab_frequency)
        self.test_statistic_ = Sa_2/Se_2
        self.p_value_ = self.calculate_p_value_F_test(F_test=self.test_statistic_, dfn=crosstab_means.shape[0]-1, dfd=X.shape[0]-crosstab_means.shape[0])
        self.critical_value_ = self.calculate_critical_value(dfn=crosstab_means.shape[0]-1, dfd=X.shape[0]-crosstab_means.shape[0], alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
        self.summary_ = pd.DataFrame({'Catergory': np.unique(X), 'Mean of dependent variable': crosstab_means.squeeze()})
    
    def crosstab_creation(self, X, y):
        crosstab_means = np.array(pd.crosstab(index=X, columns="Mean", values=y, aggfunc="mean"))
        crosstab_frequency = np.array(pd.crosstab(index=X, columns="Sum"))
        return crosstab_means, crosstab_frequency
    
    def calculate_Sa_squared(self, y, crosstab_means, crosstab_frequency):
        return 1/(crosstab_means.shape[0]-1)*np.sum(crosstab_frequency*(crosstab_means-np.mean(y))**2)
    
    def calculate_Se_squared(self, y, crosstab_means, crosstab_frequency):
        return 1/(y.shape[0]-crosstab_means.shape[0])*(np.sum(y**2)-np.sum(crosstab_frequency*crosstab_means**2))
    
    def calculate_p_value_F_test(self, F_test, dfn, dfd):
        return 1 - f.cdf(F_test, dfn, dfd)
    
    def calculate_critical_value(self, dfn, dfd, alpha):
        return f.isf(q=alpha, dfn=dfn, dfd=dfd)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False