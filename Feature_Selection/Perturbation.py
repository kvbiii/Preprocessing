from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Perturabtion():
    def __init__(self, algorithm, metric, n_repeats=5):
        self.algorithm = algorithm
        self.n_repeats = n_repeats
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
    def fit(self, X, y):
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
        self.algorithm.fit(self.X, self.y)
        if(self.metric_type == "preds"):
            y_pred_original = self.algorithm.predict(self.X)
        else:
            y_pred_original = self.algorithm.predict_proba(self.X)[:, 1]
        score = self.eval_metric(self.y, y_pred_original)
        #print("Original", score)
        i = 0
        feature_importances = []
        while(i < self.X.shape[-1]):
            if(torch.is_tensor(self.X) == True):
                X_copy = self.X.clone()
            else:
                X_copy = self.X.copy()
            inner_scores = []
            for j in range(self.n_repeats):
                #Zmieniamy rozkład dla jednej z kolumn o losowe wartości z przedziału (-0.4, 0.4)
                perturbation = np.random.normal(0.0, 0.2, size=X_copy.shape[:-1])
                if(X_copy.ndim == 2):
                    X_copy[:,i] = X_copy[:,i] + perturbation
                else:
                    X_copy[:,:,i] = X_copy[:,:,i] + perturbation
                if(self.metric_type == "preds"):
                    y_pred_inner = self.algorithm.predict(X_copy)
                else:
                    y_pred_inner = self.algorithm.predict_proba(X_copy)[:, 1]
                inner_scores.append(self.eval_metric(y, y_pred_inner))
            feature_importances.append(score - np.mean(inner_scores))
            #print("Feature importance for {}: {}".format(i, feature_importances[i]))
            i = i + 1
        self.unscaled_feature_importances_ = np.array(feature_importances)
        scaler = MinMaxScaler()
        self.feature_importances_ = np.squeeze(scaler.fit_transform(np.array(feature_importances).reshape(-1, 1)).reshape(1, -1))
        self.ranking_ = (np.argsort(np.argsort(-np.array(self.feature_importances_)))+1)