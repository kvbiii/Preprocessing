from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Jarque_Bera:
    def __init__(self):
        pass

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        return data
    
    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)

    def fit(self, data, alpha=0.05):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        skewness_ = self.calculate_skewness(data=data)
        kurtosis_ = self.calculate_kurtosis(data=data)
        self.test_statistic_ = self.calculate_test_statistic(data=data, skewness=skewness_, kurtosis=kurtosis_)
        self.p_value_ = self.calculate_p_value(test_statistic=self.test_statistic_, df=2)
        self.critical_value_ = self.calculate_critical_value(df=2, alpha=alpha)
        self.keep_H0 = self.statistical_inference(p_value=self.p_value_, alpha=alpha)

    def calculate_skewness(self, data):
        return np.sum((data-np.mean(data))**3)/((data.shape[0]-1)*np.std(data)**3)
    
    def calculate_kurtosis(self, data):
        return np.sum((data-np.mean(data))**4)/((data.shape[0]-1)*np.std(data)**4)
    
    def calculate_test_statistic(self, data, skewness, kurtosis):
        return data.shape[0]/6*(skewness**2+1/4*(kurtosis-3)**2)
    
    def calculate_p_value(self, test_statistic, df):
        return 1 - chi2.cdf(x=test_statistic, df=df)
    
    def calculate_critical_value(self, df, alpha):
        return chi2.isf(q=alpha, df=df)
    
    def statistical_inference(self, p_value, alpha):
        if(p_value >= alpha):
            return True
        else:
            return False