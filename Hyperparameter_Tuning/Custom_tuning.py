from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Custom_Tuning_CV():
    def __init__(self, metric, algorithm_instance, cross_validation_instance, num_trials=100, random_state=17):
        metrics = {"accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
                    "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
                    "mse": [lambda y, y_pred: mean_squared_error(y, y_pred), "preds"],
                    "rmse": [lambda y, y_pred: mean_squared_error(y, y_pred)**0.5, "preds"],
                    "mae": [lambda y, y_pred: mean_absolute_error(y, y_pred), "preds"]}
        if metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.metric = metric
        self.eval_metric = metrics[self.metric][0]
        self.metric_type = metrics[self.metric][1]
        self.algorithm = algorithm_instance
        self.cv = cross_validation_instance
        self.num_trials = num_trials
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
    def fit(self, X, y, params_dict, verbose=False):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        params_dict = self.check_params_dict(params_dict=params_dict)
        self.summary_frame_, self.best_params_ = self.perform_tuning(X=X, y=y, params_dict=params_dict, verbose=verbose)

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be dataframe, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        return y

    def check_params_dict(self, params_dict):
        if not isinstance(params_dict, dict):
            raise TypeError('Wrong type of params_dict. It should be dict.')
        return params_dict
    
    def perform_tuning(self, X, y, params_dict, verbose):
        all_selected_params, mean_of_train_scores, mean_of_valid_scores = [], [], []
        for trial in range(0, self.num_trials):
            current_params = {key: self.random_select(value[0], value[1]) for key, value in params_dict.items()}
            all_selected_params.append(current_params)
            train_scores, valid_scores = [], []
            for iter, (train_idx, valid_idx) in enumerate(self.cv.split(X, y)):
                X_train, X_valid = X[train_idx, :], X[valid_idx, :]
                y_train, y_valid = y[train_idx], y[valid_idx]
                model = self.algorithm.set_params(**current_params)
                model.fit(X_train, y_train)
                if(self.metric_type == "preds"):
                    y_train_pred = self.algorithm.predict(X_train)
                    y_valid_pred = self.algorithm.predict(X_valid)
                else:
                    y_train_pred = self.algorithm.predict_proba(X_train)[:, 1]
                    y_valid_pred = self.algorithm.predict_proba(X_valid)[:, 1]
                train_scores.append(self.eval_metric(y_train, y_train_pred))
                valid_scores.append(self.eval_metric(y_valid, y_valid_pred))
                if(verbose == True):
                    print("Iter {}: train scores: {}; valid scores: {}".format(iter, np.round(self.eval_metric(y_train, y_train_pred), 5), np.round(self.eval_metric(y_valid, y_valid_pred), 5)))
            mean_of_train_scores.append(np.mean(train_scores))
            mean_of_valid_scores.append(np.mean(valid_scores))
        summary_frame_ = pd.DataFrame.from_dict(all_selected_params) # type: ignore
        summary_frame_[f"Mean of Train {self.metric.upper()} Scores"] = mean_of_train_scores
        summary_frame_[f"Mean of Valid {self.metric.upper()} Scores"] = mean_of_valid_scores
        summary_frame_ = summary_frame_.sort_values(by=f"Mean of Valid {self.metric.upper()} Scores", ascending=False)
        best_params_ = {key: summary_frame_.loc[summary_frame_.index[0], key] for key in params_dict.keys()}
        return summary_frame_, best_params_
    
    def random_select(self, type, range):
        if(type == "categorical"):
            return random.choice(range)
        elif(type == "float"):
            return random.uniform(range[0], range[1])
        elif(type == "int"):
            return random.randint(range[0], range[1])
        else:
            raise ValueError(f"Unknown type: {type}. Permitted types are: `categorical`, `float`, `int`.")

"""Initial ranges for selected algorithms:
- XGBoost:
params_dict = {"n_estimators": ("int", [10, 500]),
                "learning_rate": ("float", [0.01, 0.8]),
                "reg_lambda": ("float", [0.1, 10]),
                "gamma": ("float", [0.1, 10]),
                "max_depth": ("int", [2, 50]),
                "colsample_bytree": ("float", [0.5, 1]),
                "subsample": ("float", [0.5, 1]),
                "min_child_weight": ("float", [0.5, 10])}
-Light GBM:
"""