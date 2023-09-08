from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
from requirements import *

class Iterative_Missing_values_Imputer():
    def __init__(self, estimator_regression, estimator_classification, initial_strategy="mean", imputation_order="ascending", random_state=17):
        self.estimator_regression = estimator_regression
        self.estimator_classification = estimator_classification
        statistics = {
            'mean': self.calculate_mean,
            'median': self.calculate_median,
            'mode': self.calculate_mode
        }
        self.calculate_statistic = statistics[initial_strategy]
        imputation_orders = {
            'ascending': self.ascending_order,
            'descending': self.descending_order,
            'random': self.random_order
        }
        self.imputation_order = imputation_orders[imputation_order]
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
        missing_values_mask = np.isnan(X)
        order_of_columns = self.imputation_order(missing_values_mask=missing_values_mask)
        X_imputed = self.iterative_imputation_train(X=X, missing_values_mask=missing_values_mask, order_of_columns=order_of_columns)
        self.fit_used = True
        return X_imputed
    
    def transform(self, X_train, X_test):
        self.check_fit(fit_used=self.fit_used)
        X_train = self.check_X(X=X_train)
        X_test = self.check_X(X=X_test)
        missing_values_mask = np.isnan(X_test)
        order_of_columns = self.imputation_order(missing_values_mask=missing_values_mask)
        X_imputed = self.iterative_imputation_test(X_train=X_train, X_test=X_test, missing_values_mask=missing_values_mask, order_of_columns=order_of_columns)
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
    
    def ascending_order(self, missing_values_mask):
        sorted_features_with_missing_values = np.argsort(missing_values_mask.sum(axis=0))
        features_with_missing_values = np.argwhere(np.sum(missing_values_mask, axis=0)).squeeze()
        intersection, sorted_idx, features_idx = np.intersect1d(sorted_features_with_missing_values, features_with_missing_values, return_indices=True)
        return sorted_features_with_missing_values[np.sort(sorted_idx)]
        
    def descending_order(self, missing_values_mask):
        sorted_features_with_missing_values = np.argsort(missing_values_mask.sum(axis=0))[::-1]
        features_with_missing_values = np.argwhere(np.sum(missing_values_mask, axis=0)).squeeze()
        intersection, sorted_idx, features_idx = np.intersect1d(sorted_features_with_missing_values, features_with_missing_values, return_indices=True)
        return sorted_features_with_missing_values[np.sort(sorted_idx)]
    
    def random_order(self, missing_values_mask):
        features_with_missing_values = np.argwhere(np.sum(missing_values_mask, axis=0)).squeeze()
        random.shuffle(features_with_missing_values)
        return features_with_missing_values
    
    def calculate_mean(self, X):
        return np.nanmean(X, axis=0)
    
    def calculate_median(self, X):
        return np.nanmedian(X, axis=0)
    
    def calculate_mode(self, X):
        return mode(X, nan_policy="omit", keepdims=True)[0][0]

    def statistic_imputation(self, X, missing_values_mask):
        for feature in range(0, X.shape[1]):
            X[missing_values_mask[:, feature], feature] = self.statistic_for_each_column[feature]
        return X

    def iterative_imputation_train(self, X, missing_values_mask, order_of_columns):
        X_imputed = self.statistic_imputation(X=X, missing_values_mask=missing_values_mask)
        for feature in order_of_columns:
            X_inner = np.delete(X_imputed, feature, axis=1)
            y_inner = X_imputed[:, feature]
            X_inner_train = np.delete(X_inner, missing_values_mask[:, feature], axis=0)
            y_inner_train = np.delete(y_inner, missing_values_mask[:, feature], axis=0)
            X_inner_test = X_inner[missing_values_mask[:, feature], :]
            if(feature in self.categorical_features):
                self.estimator_classification.fit(X_inner_train, y_inner_train)
                X_imputed[missing_values_mask[:, feature], feature] = self.estimator_classification.predict(X_inner_test)
            else:
                self.estimator_regression.fit(X_inner_train, y_inner_train)
                X_imputed[missing_values_mask[:, feature], feature] = self.estimator_regression.predict(X_inner_test)
        return X_imputed

    def iterative_imputation_test(self, X_train, X_test, missing_values_mask, order_of_columns):
        X_test_imputed = self.statistic_imputation(X=X_test, missing_values_mask=missing_values_mask)
        for feature in order_of_columns:
            X_inner_train = np.delete(X_train, feature, axis=1)
            y_inner_train = X_train[:, feature]
            X_inner_test = X_test_imputed[missing_values_mask[:, feature], :]
            X_inner_test = np.delete(X_inner_test, feature, axis=1)
            if(feature in self.categorical_features):
                self.estimator_classification.fit(X_inner_train, y_inner_train)
                X_test_imputed[missing_values_mask[:, feature], feature] = self.estimator_classification.predict(X_inner_test)
            else:
                self.estimator_regression.fit(X_inner_train, y_inner_train)
                X_test_imputed[missing_values_mask[:, feature], feature] = self.estimator_regression.predict(X_inner_test)
        return X_test_imputed

    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Iterative_Missing_values_Imputer has to be fitted first.')