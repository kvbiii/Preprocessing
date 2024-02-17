from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class SHAP_Linear:
    def __init__(self):
        pass
    
    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)

    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)
    
    def calculate_shap_values(self, model, X, y):
        X = self.check_data(X)
        y = self.check_data(y)
        X = self.check_for_object_columns(X)
        self.means_ = np.mean(X, axis=0)
        self.model = model
        self.model.fit(X, y)
        self.means_ = np.mean(X, axis=0)
        shap_values = self.shap_values_calculation(X)
        return shap_values
    
    def calculate_shap_single_value(self, model, X, y, observation_index):
        X = self.check_data(X)
        y = self.check_data(y)
        X = self.check_for_object_columns(X)
        self.means_ = np.mean(X, axis=0)
        self.model = model
        self.model.fit(X, y)
        shap_values = self.shap_values_calculation(X, observation_index=observation_index)
        return shap_values
    
    def shap_values_calculation(self, X, observation_index=None):
        if(observation_index is None):
            shap_values = np.zeros(X.shape)
            for feature in range(0, X.shape[1]):
                for row in range(0, X.shape[0]):
                    shap_values[row, feature] = self.compute_shap(X[row:row+1].squeeze(), feature)
        else:
            shap_values = np.zeros(X.shape[1])
            for feature in range(0, X.shape[1]):
                shap_values[feature] = self.compute_shap(X[observation_index], feature)
        return shap_values

    def compute_shap(self, row, shap_feature):
        value_function = self.make_value_function(row=row, shap_feature=shap_feature)
        return sum([value_function(coalition) for coalition in self.make_coalitions(row, shap_feature)])
    
    def make_coalitions(self, row, shap_feature):
        subset_without_shap_feature = [feature for feature in range(0, len(row)) if feature != shap_feature]
        for i in range(len(subset_without_shap_feature) + 1):
            for subset in combinations(subset_without_shap_feature, i):
                yield list(subset)
    
    def make_value_function(self, row, shap_feature):
        def value(subset):
            marginal_gain = self.pred_linear_regression(subset + [shap_feature], row) - self.pred_linear_regression(subset, row)
            coalition_probability_inversed = comb(len(row) - 1, len(subset))
            return 1/len(row)*marginal_gain/coalition_probability_inversed
        return value
    
    def pred_linear_regression(self, coalition, row):
        row_copy = row.copy()
        for i in range(0, len(row)):
            if i not in coalition:
                #Replace values from row with mean if they are not in coalition
                row_copy[i] = self.means_[i]
        return self.model.predict(row_copy.reshape(1, -1)).squeeze()