from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Inter_Quantile_Range():
    def __init__(self):
        self.fit_used = False

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)
    
    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        return np.array(data)
    
    def fit(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        self.Q1 = np.quantile(data, q=0.25)
        self.Q3 = np.quantile(data, q=0.75)
        self.IQR = self.Q3-self.Q1
        self.fit_used = True

    def find_outliers(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        self.indices_of_outliers_ = np.where((data < self.Q1-1.5*self.IQR) | (data > self.Q3+1.5*self.IQR))[0]
        return self.indices_of_outliers_
    
    def remove_outliers(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.delete(data, self.indices_of_outliers_)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Inter_Quantile_Range has to be fitted first.')