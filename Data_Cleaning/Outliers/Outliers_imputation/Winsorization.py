from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Winsorization():
    def __init__(self, percentile=0.05, percentage=0.01):
        self.percentile = percentile
        self.percentage = percentage
        self.fit_used = False

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        return np.array(data)
    
    def fit_transform(self, data, feature_type="continous"):
        data = self.check_data(data=data)
        if(feature_type=="continous"):
            self.lower_percentile = np.quantile(data, q=self.percentile)
            self.upper_percentile = np.quantile(data, q=1-self.percentile)
            data[data < self.lower_percentile] = self.lower_percentile
            data[data > self.upper_percentile] = self.upper_percentile
        else:
            data = self.transform_categorical_feature(data=data)
        self.fit_used = True
        return data
    
    def transform_categorical_feature(self, data):
        unique_values, counts = np.unique(data, return_counts=True)
        for iter in range(0, len(unique_values)):
            if(counts[iter] < self.percentage*np.sum(counts)):
                data[data==unique_values[iter]] = -99
        return data

    def transform(self, data, feature_type="continous"):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        if(feature_type=="continous"):
            data[data < self.lower_percentile] = self.lower_percentile
            data[data > self.upper_percentile] = self.upper_percentile
        else:
            data = self.transform_categorical_feature(data=data)
        return data
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Winsorization has to be fitted first.')