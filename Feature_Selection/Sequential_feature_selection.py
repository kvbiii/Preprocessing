from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Sequential_feature_selection():
    
    def __init__(self, algorithm, metric, cv=None):
        metrics = {"accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
                     "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
                     "neg_mse": [lambda y, y_pred: -mean_squared_error(y, y_pred), "preds"],
                     "neg_rmse": [lambda y, y_pred: -mean_squared_error(y, y_pred)**0.5, "preds"],
                     "neg_mae": [lambda y, y_pred: -mean_absolute_error(y, y_pred), "preds"]}
        if metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.eval_metric = metrics[metric][0]
        self.metric_type = metrics[metric][1]
        self.algorithm = algorithm
        self.cv=cv

    def cross_validation(self, X, y):
        cross_validation_scores = []
        for train_idx, test_idx in self.cv.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            self.algorithm.fit(X_train, y_train)
            if(self.metric_type== "preds"):
                y_pred = self.algorithm.predict(X_test)
            else:
                y_pred = self.algorithm.predict_proba(X_test)[:, 1]
            cross_validation_scores.append(self.eval_metric(y_test, y_pred))
        return np.mean(cross_validation_scores)
    def no_cross_validation(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.algorithm.fit(X_train, y_train)
        if(self.metric_type == "preds"):
            y_pred = self.algorithm.predict(X_valid)
        else:
            y_pred = self.algorithm.predict_proba(X_valid)[:, 1]
        return self.eval_metric(y_valid, y_pred)
    
    def forward_fit(self, X, y):
        self.X = X
        if not isinstance(self.X, np.ndarray) or not torch.is_tensor(self.X):
            try:
                self.X = np.array(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array, torch_tensor, list or dataframe columns.')
        self.y = y
        if not isinstance(self.y, np.ndarray) or not torch.is_tensor(self.y):
            try:
                self.y = np.array(self.y)
            except:
                raise TypeError('Wrong type of y. It should be numpy array, torch_tensor, list or dataframe column.')
        i = 0
        best_first_score = -np.inf
        while(i < self.X.shape[-1]):
            if(self.cv == None):
                score = self.no_cross_validation(X=self.X[:, i].reshape(-1, 1), y=self.y)
            else:
                score = self.cross_validation(X=self.X[:, i].reshape(-1, 1), y=self.y)
            if(best_first_score < score):
                best_first_score = score
                self.list_of_best_variables_indexes = [i]
            i = i + 1
        #We create a set with the best feature, and then see if we get better results by taking a new variable. The one after which we input the best score is taken into the overall set of features.
        #print("Best first score: {}".format(best_first_score))
        #print("Best first variable: {}".format(self.list_of_best_variables_indexes))
        best_score = best_first_score
        i = 0
        #Let's keep the indexes of those variables that were not used in the rest_of_the_features list.
        rest_of_the_features = np.setdiff1d([i for i in range(self.X.shape[1])], self.list_of_best_variables_indexes)
        while True:
            #print("Starting or restarting loop")
            restart = False
            for i in rest_of_the_features:
                #print("We check how the current set: {} will manage with an additional variable: {}".format(self.list_of_best_variables_indexes, i))
                X_copy = self.X[:, self.list_of_best_variables_indexes+[i]]
                if(self.cv == None):
                    score = self.no_cross_validation(X=X_copy, y=self.y)
                else:
                    score = self.cross_validation(X=X_copy, y=self.y)
                if(best_score <= score):
                    #print("Improved! Because {}>{}".format(score, best_score))
                    best_score = score
                    self.list_of_best_variables_indexes.append(i)
                    rest_of_the_features = np.setdiff1d([i for i in range(self.X.shape[-1])], self.list_of_best_variables_indexes)
                    restart = True
                    break
            if(restart == False):
                break
        self.support_ = np.array([True if i in self.list_of_best_variables_indexes else False for i in range(0, self.X.shape[1])]).astype(bool)

    def forward_fit_lstm(self, X, y):
        self.X = X
        if not isinstance(self.X, np.ndarray) and not torch.is_tensor(self.X):
            try:
                self.X = torch.tensor(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array or torch_tensor.')
        self.y = y
        if not isinstance(self.y, np.ndarray) and not torch.is_tensor(self.y):
            try:
                self.y = torch.tensor(self.y)
            except:
                raise TypeError('Wrong type of y. It should be numpy array or torch_tensor')
        i = 0
        best_first_score = -np.inf
        while(i < self.X.shape[-1]):
            if(self.cv == None):
                score = self.no_cross_validation(X=self.X[:,:,i].unsqueeze(2), y=self.y)
            else:
                score = self.cross_validation(X=self.X[:,:,i].unsqueeze(2), y=self.y)
            if(best_first_score < score):
                best_first_score = score
                self.list_of_best_variables_indexes = [i]
            i = i + 1
        #We create a set with the best feature, and then see if we get better results by taking a new variable. The one after which we input the best score is taken into the overall set of features.
        #print("Best first score: {}".format(best_first_score))
        #print("Best first variable: {}".format(self.list_of_best_variables_indexes))
        best_score = best_first_score
        i = 0
        #Let's keep the indexes of those variables that were not used in the rest_of_the_features list.
        rest_of_the_features = np.setdiff1d([i for i in range(self.X.shape[-1])], self.list_of_best_variables_indexes)
        while True:
            #print("Starting or restarting loop")
            restart = False
            for i in rest_of_the_features:
                #print("We check how the current set: {} will manage with an additional variable: {}".format(self.list_of_best_variables_indexes, i))
                X_copy = self.X[:,:, self.list_of_best_variables_indexes+[i]]
                if(self.cv == None):
                    score = self.no_cross_validation(X=X_copy, y=self.y)
                else:
                    score = self.cross_validation(X=X_copy, y=self.y)
                if(best_score <= score):
                    #print("Improved! Because {}>{}".format(score, best_score))
                    best_score = score
                    self.list_of_best_variables_indexes.append(i)
                    rest_of_the_features = np.setdiff1d([i for i in range(self.X.shape[-1])], self.list_of_best_variables_indexes)
                    restart = True
                    break
            if(restart == False):
                break
        self.support_ = np.array([True if i in self.list_of_best_variables_indexes else False for i in range(0, self.X.shape[-1])]).astype(bool)

    def transform_lstm(self, X):
        variables_to_keep = self.list_of_best_variables_indexes
        return X[:,:,variables_to_keep]
    
    def transform(self, X):
        variables_to_keep = self.list_of_best_variables_indexes
        return X[:,variables_to_keep]
    
    def backward_fit(self, X, y):
        self.X = X
        if not isinstance(self.X, np.ndarray):
            try:
                self.X = np.array(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array, list or dataframe columns.')
        self.y = y
        if not isinstance(self.y, np.ndarray):
            try:
                self.y = np.array(self.y)
            except:
                raise TypeError('Wrong type of y. It should be numpy array, list or dataframe column.')
        best_score_all = -np.inf
        X_copy = self.X.copy()
        original_indices = [i for i in range(X_copy.shape[-1])]
        final_variables_to_drop = []
        i = 1
        #We open loop with a copy of our X
        while(i < X_copy.shape[-1]):
            #print("Current set of variables: {}".format(original_indices))
            try:
                X_copy = np.delete(X_copy, worst_variable_index, axis=1)
            except:
                pass
            best_inner_score = -np.inf
            inner_iterator = 0
            while(inner_iterator < X_copy.shape[-1] and X_copy.shape[-1]>1):
                X_inner_copy = X_copy.copy()
                X_inner_copy = np.delete(X_inner_copy, inner_iterator, axis=1)
                if(self.cv == None):
                    score = self.no_cross_validation(X=X_inner_copy, y=self.y)
                else:
                    score = self.cross_validation(X=X_inner_copy, y=self.y)
                if(score > best_inner_score):
                    best_inner_score = score
                    worst_variable_index = inner_iterator
                inner_iterator = inner_iterator + 1
            if(best_inner_score >= best_score_all):
                #print("Improved relative to best score because: {} > {}".format(best_inner_score, best_score_all))
                best_score_all = best_inner_score
                final_variables_to_drop.append(original_indices[worst_variable_index])
                #print("Updated the best result. New set with variables to drop: {}".format(final_variables_to_drop))
            #Here a condition in case (when X_copy.shape[1]==1), so that it does not once again remove a variable from some index
            if(len(original_indices) > 1):
                del original_indices[worst_variable_index]
        self.list_of_best_variables_indexes = np.setdiff1d([i for i in range(self.X.shape[-1])], final_variables_to_drop)
        self.support_ = [True if i in self.list_of_best_variables_indexes else False for i in range(0, self.X.shape[-1])]

    def backward_fit_lstm(self, X, y):
        self.X = X
        if not isinstance(self.X, np.ndarray) and not torch.is_tensor(self.X):
            try:
                self.X = torch.tensor(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array or torch_tensor.')
        self.y = y
        if not isinstance(self.y, np.ndarray) and not torch.is_tensor(self.y):
            try:
                self.y = torch.tensor(self.y)
            except:
                raise TypeError('Wrong type of y. It should be numpy array or torch_tensor')
        best_score_all = -np.inf
        X_copy = self.X.clone()
        original_indices = [i for i in range(X_copy.shape[-1])]
        final_variables_to_drop = []
        i = 1
        #We open loop with a copy of our X
        while(i < X_copy.shape[-1]):
            #print("Current set of variables: {}".format(original_indices))
            try:
                X_copy = np.delete(X_copy, worst_variable_index, axis=2)
            except:
                pass
            best_inner_score = -np.inf
            inner_iterator = 0
            while(inner_iterator < X_copy.shape[-1] and X_copy.shape[-1]>1):
                X_inner_copy = X_copy.clone()
                X_inner_copy = np.delete(X_inner_copy, inner_iterator, axis=2)
                if(self.cv == None):
                    score = self.no_cross_validation(X=X_inner_copy, y=self.y)
                else:
                    score = self.cross_validation(X=X_inner_copy, y=self.y)
                if(score > best_inner_score):
                    best_inner_score = score
                    worst_variable_index = inner_iterator
                inner_iterator = inner_iterator + 1
            if(best_inner_score >= best_score_all):
                #print("Improved relative to best score because: {} > {}".format(best_inner_score, best_score_all))
                best_score_all = best_inner_score
                final_variables_to_drop.append(original_indices[worst_variable_index])
                #print("Updated the best result. New set with variables to drop: {}".format(final_variables_to_drop))
            #Here a condition in case (when X_copy.shape[1]==1), so that it does not once again remove a variable from some index
            if(len(original_indices) > 1):
                del original_indices[worst_variable_index]
        self.list_of_best_variables_indexes = np.setdiff1d([i for i in range(self.X.shape[-1])], final_variables_to_drop)
        self.support_ = [True if i in self.list_of_best_variables_indexes else False for i in range(0, self.X.shape[-1])]