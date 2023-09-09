from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Simple_Imputer():
    def __init__(self, continous_strategy="mean", random_state=17):
        statistics = {
            'mean': self.calculate_mean,
            'median': self.calculate_median,
            'mode': self.calculate_mode
        }
        self.calculate_statistic = statistics[continous_strategy]
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(X)

    def fit(self, X, categorical_features=[]):
        self.categorical_features = self.find_numeric_order_of_columns(X=X, categorical_features=categorical_features)
        X = self.check_X(X=X)
        self.statistic_for_each_column = self.find_statistics(X=X, categorical_features=self.categorical_features)
        self.fit_used = True
    
    def fit_transform(self, X, categorical_features=[]):
        self.categorical_features = self.find_numeric_order_of_columns(X=X, categorical_features=categorical_features)
        X = self.check_X(X=X)
        self.statistic_for_each_column = self.find_statistics(X=X, categorical_features=self.categorical_features)
        X_imputed = self.impute_missing_values(X=X)
        self.fit_used = True
        return X_imputed
    
    def transform(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X)
        X_imputed = self.impute_missing_values(X=X)
        return X_imputed
    
    def find_numeric_order_of_columns(self, X, categorical_features):
        if(all(isinstance(element, str) for element in categorical_features) == True and isinstance(X, pd.DataFrame)):
            categorical_features = [index for index, feature in enumerate(X.columns) if feature in categorical_features]
        return categorical_features

    def find_statistics(self, X, categorical_features):
        statistic_for_each_column = []
        for feature in range(0, X.shape[1]):
            if(feature in categorical_features):
                statistic_for_each_column.append(self.calculate_mode(X=X[:, feature]))
            else:
                statistic_for_each_column.append(self.calculate_statistic(X=X[:, feature]))
        return statistic_for_each_column

    def calculate_mean(self, X):
        return np.nanmean(X)
    
    def calculate_median(self, X):
        return np.nanmedian(X)
    
    def calculate_mode(self, X):
        return mode(X, nan_policy="omit", keepdims=True)[0][0]
    
    def impute_missing_values(self, X):
        for feature in range(0, X.shape[1]):
            X[np.where(np.isnan(X[:, feature])), feature] = self.statistic_for_each_column[feature]
        return X
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Simple_Imputer has to be fitted first.')