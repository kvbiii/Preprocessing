from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class NNMF:
    def __init__(self, n_components, max_iter=1000, learning_rate=0.001, alpha_W=0, alpha_H=0, l1_ratio=0, random_state=17):
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)

    def check_for_object_columns(self, data):
        data = pd.DataFrame(data)
        if data.select_dtypes(include=np.number).shape[1] != data.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(data)
    
    def check_n_of_components(self, data):
        if(data.shape[1] < self.n_components):
            raise ValueError('Specified n_components={} < number of features. It has to be greater or equal to the number of columns.'.format(self.n_components))

    def fit(self, data, verbose=False):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.N = self.fit_data.shape[0]
        self.M = self.fit_data.shape[1]
        self.W, self.H = self.perform_nnmf(data=self.fit_data, verbose=verbose)
        self.fit_used = True
    
    def fit_transform(self, data, verbose=False):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        self.N = self.fit_data.shape[0]
        self.M = self.fit_data.shape[1]
        self.W, self.H = self.perform_nnmf(data=self.fit_data, verbose=verbose)
        self.components_ = self.H
        self.fit_used = True
        return self.W
    
    def transform(self, data):
        self.check_fit(fit_used=self.fit_used)
        data = self.check_data(data=data)
        data = self.check_for_object_columns(data=data)
        return self.W
    
    def perform_nnmf(self, data, verbose):
        W, H = self.initialize_matrices()
        for epoch in range(1, self.max_iter+1):
            loss = self.calculate_loss(data=data, W=W, H=H)
            gradient_W, gradient_H = self.calculate_gradient(data=data, W=W, H=H)
            W = W - self.learning_rate*gradient_W
            H = H - self.learning_rate*gradient_H
            W = self.update_to_nonegative(matrix=W)
            H = self.update_to_nonegative(matrix=H)
            if((epoch%50 == 0 and verbose==True) or (epoch == 1 and verbose == True)):
                print("Epoch: {}, Loss: {}".format(epoch, loss))
        return W, H
    
    def initialize_matrices(self):
        W = np.abs(np.random.normal(loc=0, scale=1e-2, size=(self.N, self.n_components)))
        H = np.abs(np.random.normal(loc=0, scale=1e-2, size=(self.n_components, self.M)))
        return W, H
    
    def calculate_loss(self, data, W, H):
        return 0.5*np.linalg.norm(data - np.dot(W, H))**2+self.alpha_W*self.l1_ratio*self.M*np.sum(np.abs(W))+self.alpha_H*self.l1_ratio*self.N*np.sum(np.abs(H))+0.5*self.alpha_W*(1-self.l1_ratio)*np.sum(W**2)+0.5*self.alpha_H*(1-self.l1_ratio)*np.sum(H**2)
    
    def calculate_gradient(self, data, W, H):
        gradient_W = 0.5*(-2*np.dot(data, H.T) + 2*np.dot(np.dot(W, H), H.T)) + self.alpha_W*self.l1_ratio*self.M + 2*self.alpha_W*(1-self.l1_ratio)*self.M*W
        gradient_H = 0.5*(-2*np.dot(W.T, data) + 2*np.dot(np.dot(W.T, W), H)) + self.alpha_H*self.l1_ratio*self.N + 2*self.alpha_H*(1-self.l1_ratio)*self.N*H
        return gradient_W, gradient_H
    
    def update_to_nonegative(self, matrix):
        return np.where(matrix >= 0, matrix, 0)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('NNMF has to be fitted first.')