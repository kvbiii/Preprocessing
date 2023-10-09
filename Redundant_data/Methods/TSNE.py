from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=500, n_iter=1000, tolerance=1e-5, early_exaggeration=4, random_state=17):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
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

    def fit(self, data):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
    
    def fit_transform(self, data, verbose=False):
        self.fit_data = self.check_data(data=data)
        self.fit_data = self.check_for_object_columns(data=self.fit_data)
        self.check_n_of_components(data=self.fit_data)
        y = self.perform_tsne(data=self.fit_data, verbose=verbose)
        return y
    
    def perform_tsne(self, data, verbose):
        p_joint = self.calculate_p_joint(data=data)
        Y = np.zeros(shape=(self.n_iter, data.shape[0], self.n_components))
        Y[1, :, :] = self.initialize_y(data=data)
        for epoch in range(1, self.n_iter-1):
            alpha, early_exaggeration = self.calculate_momentum(epoch=epoch)
            q_joint = self.calculate_q_joint(y=Y[epoch, :, :])
            loss = self.calculate_loss(p_joint=p_joint, q_joint=q_joint)
            gradient = self.calculate_gradient(p_joint=p_joint*early_exaggeration, q_joint=q_joint, y=Y[epoch, :, :])
            Y[epoch+1, :, :] = Y[epoch, :, :] - self.learning_rate*gradient + alpha*(Y[epoch, :, :] - Y[epoch-1, :, :])
            if((epoch%50 == 0 and verbose==True) or (epoch == 1 and verbose == True)):
                print("Epoch: {}, Loss: {}".format(epoch, loss))
        return Y[-1]
    
    def calculate_p_joint(self, data):
        pairwise_distances = self.pairwise_distances(X=data)
        sigmas = self.find_sigmas(data=data, pairwise_distances=pairwise_distances)
        p_conditional_matrix = self.calculate_p_conditional(pairwise_distances=pairwise_distances, sigma=sigmas)
        p_joint_matrix = (p_conditional_matrix + p_conditional_matrix.T)/(2*data.shape[0])
        p_joint_matrix = np.maximum(p_joint_matrix, np.nextafter(0, 1))
        return p_joint_matrix

    def pairwise_distances(self, X):
        return np.sum((X[np.newaxis, :] - X[:, np.newaxis])**2, axis=2)
    
    def calculate_p_conditional(self, pairwise_distances, sigma):
        nominator = np.exp(-1/2*pairwise_distances/sigma[:, np.newaxis]**2)
        np.fill_diagonal(nominator, 0)
        denominator = np.sum(nominator, axis=1)
        p_conditional_matrix = nominator/denominator[:, np.newaxis]
        p_conditional_matrix = np.maximum(p_conditional_matrix, np.nextafter(0, 1))
        return p_conditional_matrix
    
    def calculate_perplexity(self, p_conditional_matrix):
        return 2**(-np.sum(p_conditional_matrix*np.log2(p_conditional_matrix), axis=1))
    
    def find_sigmas(self, data, pairwise_distances):
        sigmas = np.zeros(shape=(data.shape[0], ))
        for i in range(0, data.shape[0]):
            upper = np.std(pairwise_distances[i, :]**0.5)*5
            lower = np.std(pairwise_distances[i, :]**0.5)*0.01
            sigma = self.binary_search(perplexity_function=lambda sigma: self.calculate_perplexity(p_conditional_matrix=self.calculate_p_conditional(pairwise_distances=pairwise_distances[i:i+1, :], sigma=np.array([sigma]))), upper=upper, lower=lower)
            sigmas[i] = sigma
        return sigmas
    
    def binary_search(self, perplexity_function, upper, lower):
        for i in range(0, self.n_iter+1):
            sigma_guess = (upper + lower)/2
            perplexity = perplexity_function(sigma_guess)
            if(np.abs(perplexity - self.perplexity) < self.tolerance):
                return sigma_guess
            elif(perplexity > self.perplexity):
                upper = sigma_guess
            else:
                lower = sigma_guess
        return sigma_guess
    
    def initialize_y(self, data):
        return np.random.normal(loc=0, scale=1e-4, size=(data.shape[0], self.n_components))
    
    def calculate_momentum(self, epoch):
        if(epoch < 250):
            return 0.5, self.early_exaggeration
        else:
            return 0.8, 1
    
    def calculate_q_joint(self, y):
        pairwise_distances = self.pairwise_distances(X=y)
        nominator = 1/(1 + pairwise_distances)
        np.fill_diagonal(nominator, 0)
        denominator = np.sum(nominator)
        q_joint = nominator/denominator
        q_joint = np.maximum(q_joint, np.nextafter(0, 1))
        return q_joint
    
    def calculate_loss(self, p_joint, q_joint):
        return np.sum(p_joint*np.log(p_joint/q_joint))
    
    def calculate_gradient(self, p_joint, q_joint, y):
        pairwise_distances = self.pairwise_distances(X=y)
        reciprocal = 1/(1 + pairwise_distances)
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        reciprocal = (1+np.linalg.norm(y_diff, axis=2))**(-1)
        pq_diff = p_joint - q_joint
        return 4*np.sum(np.expand_dims(pq_diff, axis=2)*y_diff*np.expand_dims(reciprocal, axis=2), axis=1)