from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class SHAP_Tree:
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
        self.model = model
        self.model.fit(X, y)
        shap_values = self.shap_values_calculation(X)
        return shap_values
    
    def calculate_shap_single_value(self, model, X, y, observation_index):
        X = self.check_data(X)
        y = self.check_data(y)
        X = self.check_for_object_columns(X)
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
            marginal_gain = self.pred_tree(subset + [shap_feature], row) - self.pred_tree(subset, row)
            coalition_probability_inversed = comb(len(row) - 1, len(subset))
            return 1/len(row)*marginal_gain/coalition_probability_inversed
        return value
    
    def pred_tree(self, coalition, row, node=0):
        left_node = self.model.tree_.children_left[node]
        right_node = self.model.tree_.children_right[node]
        is_leaf = left_node == right_node
        if is_leaf:
            return self.model.tree_.value[node].squeeze()
        feature = self.model.tree_.feature[node]
        if feature in coalition:
            if row[feature] <= self.model.tree_.threshold[node]:
                # go left
                return self.pred_tree(coalition, row, node=left_node)
            # go right
            return self.pred_tree(coalition, row, node=right_node)

        # take weighted average of left and right
        wl = self.model.tree_.n_node_samples[left_node] / self.model.tree_.n_node_samples[node]
        wr = self.model.tree_.n_node_samples[right_node] / self.model.tree_.n_node_samples[node]
        value = wl * self.pred_tree(coalition, row, node=left_node)
        value += wr * self.pred_tree(coalition, row, node=right_node)
        return value