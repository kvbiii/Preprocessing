from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class LeaveOneOut():
    def __init__(self, random_state=17):
        self.random_state = random_state
        np.random.seed(self.random_state)
        
    def split(self, X, y):
        n_samples = X.shape[0]
        all_indices = set(np.arange(n_samples))
        possible_test_indices = set(np.arange(n_samples))
        for indice in range(n_samples-1):
            test_index = np.random.choice(np.array(list(possible_test_indices)))
            train_indices = all_indices.difference(set({test_index}))
            possible_test_indices.remove(test_index)
            yield list(train_indices), test_index