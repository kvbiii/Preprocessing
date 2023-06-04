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
                X_valid_copy = self.X_valid.clone()
            else:
                X_valid_copy = self.X_valid.copy()
            inner_scores = []
            for j in range(self.n_repeats):
                #Change distribution of our column i with random values from - median to median of this column values
                if(X_valid_copy.ndim == 2):
                    perturbation = np.random.uniform(low=-np.median(X_valid_copy[:, i]), high=np.median(X_valid_copy[:, i]), size=X_valid_copy.shape[:-1])
                    X_valid_copy[:,i] = X_valid_copy[:,i] + perturbation
                else:
                    perturbation = np.random.uniform(low=-np.median(X_valid_copy[:,:, i]), high=np.median(X_valid_copy[:,:, i]), size=X_valid_copy.shape[:-1])
                    X_valid_copy[:,:,i] = X_valid_copy[:,:,i] + perturbation
                if(self.metric_type == "preds"):
                    y_pred_inner = self.algorithm.predict(X_valid_copy)
                else:
                    y_pred_inner = self.algorithm.predict_proba(X_valid_copy)[:, 1]
                inner_scores.append(self.eval_metric(self.y_valid, y_pred_inner))
            feature_importances.append(score - np.mean(inner_scores))
            #print("Feature importance for {}: {}".format(i, feature_importances[i]))
            i = i + 1
        self.unscaled_feature_importances_ = np.array(feature_importances)
        scaler = MinMaxScaler()
        self.feature_importances_ = np.squeeze(scaler.fit_transform(np.array(feature_importances).reshape(-1, 1)).reshape(1, -1))
        self.ranking_ = rankdata(1-self.feature_importances_, method='dense')