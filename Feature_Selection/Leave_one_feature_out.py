from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Leave_One_Feature_Out():
    def __init__(self, algorithm, metric):
        self.algorithm = algorithm
        metrics = {"accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
                    "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
                    "neg_mse": [lambda y, y_pred: -mean_squared_error(y, y_pred), "preds"],
                    "neg_rmse": [lambda y, y_pred: -mean_squared_error(y, y_pred)**0.5, "preds"],
                    "neg_mae": [lambda y, y_pred: -mean_absolute_error(y, y_pred), "preds"]}
        if metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.metric=metric
        self.eval_metric = metrics[metric][0]
        self.metric_type = metrics[metric][1]
    def fit(self, X, y, X_valid=None, y_valid=None):
        self.X = X
        if not isinstance(self.X, np.ndarray) and not torch.is_tensor(self.X):
            try:
                self.X = np.array(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array, torch_tensor, list or dataframe columns.')
        self.y = y
        if not isinstance(self.y, np.ndarray) and not torch.is_tensor(self.y):
            try:
                self.y = np.array(self.y)
            except:
                raise TypeError('Wrong type of y. It should be numpy array, torch_tensor, list or dataframe column.')
        if isinstance(X_valid,np.ndarray):
            self.X_train = self.X
            self.y_train = self.y
            self.X_valid = X_valid
            self.y_valid = y_valid
        else:
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, test_size=0.2, random_state=17)
        self.algorithm.fit(self.X_train, self.y_train)
        if(self.metric_type == "preds"):
            y_pred_original = self.algorithm.predict(self.X_valid)
        else:
            y_pred_original = self.algorithm.predict_proba(self.X_valid)[:, 1]
        score = self.eval_metric(self.y_valid, y_pred_original)
        #print("Original", score)
        i = 0
        feature_importances = []
        while(i < self.X.shape[-1]):
            if(torch.is_tensor(self.X) == True):
                X_train_copy = self.X_train.clone()
                X_valid_copy = self.X_valid.clone()
            else:
                X_train_copy = self.X_train.copy()
                X_valid_copy = self.X_valid.copy()
            #Drop one of the columns
            if(X_train_copy.ndim == 2):
                X_train_copy = np.delete(X_train_copy, i, axis=1)
                X_valid_copy = np.delete(X_valid_copy, i, axis=1)
            else:
                X_train_copy = np.delete(X_train_copy, i, axis=2)
                X_valid_copy = np.delete(X_valid_copy, i, axis=2)
            self.algorithm.fit(X_train_copy, self.y_train)
            if(self.metric_type == "preds"):
                y_pred_inner = self.algorithm.predict(X_valid_copy)
            else:
                y_pred_inner = self.algorithm.predict_proba(X_valid_copy)[:, 1]
            feature_importances.append(score - self.eval_metric(self.y_valid, y_pred_inner))
            #print("Feature importance for {}: {}".format(i, feature_importances[i]))
            i = i + 1
        self.unscaled_feature_importances_ = np.array(feature_importances)
        scaler = MinMaxScaler()
        self.feature_importances_ = np.squeeze(scaler.fit_transform(np.array(feature_importances).reshape(-1, 1)).reshape(1, -1))
        self.ranking_ = rankdata(1-self.feature_importances_, method='dense')