from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Mahalanobis_distance():
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.fit_used = False
    
    def fit(self, data):
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        means = self.calculate_mean(data=data)
        covariance = self.calculate_covariance(data=data)
        mahalanobis_distance = self.calculate_mahalanobis_distance(data=data, means=means, covariance=covariance)
        self.mahalanobis_distance = mahalanobis_distance
        self.threshold = self.calculate_threshold(data=data)
        print(self.threshold)
        self.indices_of_outliers = self.find_indices_of_outliers(mahalanobis_distance=self.mahalanobis_distance, threshold=self.threshold)
        self.fit_used = True

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)
    
    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)

    def calculate_mean(self, data):
        return np.mean(data, axis=0)
    
    def calculate_covariance(self, data):
        return np.cov(data.T)
    
    def calculate_mahalanobis_distance(self, data, means, covariance):
        return np.sqrt(np.sum((data-means).dot(np.linalg.inv(covariance))*(data-means), axis=1))
    
    def calculate_threshold(self, data):
        return np.sqrt(chi2.ppf(q=1-self.alpha, df=data.shape[1]))

    def remove_outliers(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        indices_of_outliers = self.find_indices_of_outliers(mahalanobis_distance=self.mahalanobis_distance, threshold=self.threshold)
        return np.delete(data, indices_of_outliers)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Mahalanobis_distance has to be fitted first.')
    
    def find_indices_of_outliers(self, mahalanobis_distance, threshold):
        return np.where(mahalanobis_distance > threshold)[0]