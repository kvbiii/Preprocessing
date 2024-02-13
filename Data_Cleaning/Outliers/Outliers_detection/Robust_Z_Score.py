from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Robust_Z_Score():
    def __init__(self, const=0.6745, threshold=3):
        self.fit_used = False
        self.const_ = const
        self.threshold = threshold

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
        self.median_ = np.median(data)
        self.MAD_ = np.median(np.abs(data-np.median(data)))
        self.fit_used = True

    def find_outliers(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        self.indices_of_outliers_ = np.where(np.abs(self.const_*(data-self.median_)/self.MAD_)>self.threshold)[0]
        return self.indices_of_outliers_
    
    def remove_outliers(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.delete(data, self.indices_of_outliers_)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Robust_Z_Score has to be fitted first.')