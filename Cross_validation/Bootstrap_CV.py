from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class BootstrapCV():
    def __init__(self, n_splits, random_state=17):
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(self.random_state)

    def split(self, X, y):
        all_indices = [i for i in range(X.shape[0])]
        bootstrap_instance = MovingBlockBootstrap(7, X, y=y, seed=self.random_state)
        for iter in bootstrap_instance.bootstrap(self.n_splits):
            train_indices = bootstrap_instance.index
            test_indices = np.setdiff1d(all_indices, train_indices)
            yield train_indices, test_indices