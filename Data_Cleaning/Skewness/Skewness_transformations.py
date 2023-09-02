from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Skewness:
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

    def calculate_skewness(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.sum((data-np.mean(data))**3)/((data.shape[0]-1)*np.std(data)**3)

    def logarithmic_transformation(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.log(data)
    
    def square_root_transformation(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return np.sqrt(data)
    
    def box_cox_transformation(self, data, reg_lambda=0):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        if(reg_lambda==0):
            return self.logarithmic_transformation(data=data)
        else:
            return (data**reg_lambda-1)/reg_lambda