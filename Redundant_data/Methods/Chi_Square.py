from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Chi_Square_Test():
    def __init__(self):
        pass

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 1):
            y = y[None, :]
        return y
    
    def check_assumptions(self, crosstab):
        if(all(i <= 5 for i in crosstab.flatten())):
            return False
        else:
            return True

    def fit(self, X, y, alpha=0.05):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        self.crosstab_ = self.crosstab_creation(X=X, y=y)
        self.assumption_ = self.check_assumptions(crosstab=self.crosstab_)
        self.test_statistic_ = self.calculate_chi_square_statistic(crosstab=self.crosstab_)
        self.df_ = self.calculate_degrees_of_freedom(crosstab=self.crosstab_)
        self.p_value_ = self.calculate_p_value(test_statistic=self.test_statistic_, df=self.df_)
        self.critical_value_ = self.calculate_critical_value(df=self.df_, alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)
        self.V_Cramera_ = self.calculate_V_Cramera(test_statistic=self.test_statistic_, crosstab=self.crosstab_)
    
    def crosstab_creation(self, X, y):
        return np.array(pd.crosstab(X, y, margins=True))
    
    def calculate_chi_square_statistic(self, crosstab):
        chi_square = 0
        for row in range(0, crosstab.shape[0]-1):
            for column in range(0, crosstab.shape[1]-1):
                observed = crosstab[row][column]
                expected = crosstab[row][-1]*crosstab[-1][column]/crosstab[-1][-1]
                chi_square += (observed-expected)**2/expected
        return chi_square
    
    def calculate_degrees_of_freedom(self, crosstab):
        return (crosstab.shape[0]-2)*(crosstab.shape[1]-2)
    
    def calculate_p_value(self, test_statistic, df):
        return 1 - chi2.cdf(x=test_statistic, df=df)
    
    def calculate_critical_value(self, df, alpha):
        return chi2.isf(q=alpha, df=df)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False
    
    def calculate_V_Cramera(self, test_statistic, crosstab):
        return (test_statistic/(crosstab[-1][-1]*np.min([crosstab.shape[0]-2, crosstab.shape[1]-2])))**0.5