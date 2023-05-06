from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
      self.n_splits = n_splits
        
    def get_n_splits(self, X, y, groups):
      return self.n_splits
        
    def split(self, X, y=None, groups=None, train_size=0.8):
      n_samples = len(X)
      k_fold_size = math.ceil(n_samples/self.n_splits)
      indices = np.arange(n_samples)
      margin = 0
      for i in range(self.n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = math.ceil(train_size * (stop - start)) + start
        yield indices[start: mid], indices[mid + margin: stop]
        
class BlockingTimeSeriesSplit_with_valid():
    def __init__(self, n_splits, train_size=13/20, validation_size=3/20):
      self.n_splits = n_splits
      self.train_size = train_size
      self.validation_size = validation_size
        
    def get_n_splits(self, X, y, groups):
      return self.n_splits
        
    def split(self, X, y=None, groups=None, train_size=0.8):
      n_samples = len(X)
      k_fold_size = math.ceil(n_samples/self.n_splits)
      indices = np.arange(n_samples)
      for i in range(0, self.n_splits):
        start_train = i * k_fold_size
        stop_train = start_train + math.ceil(self.train_size*k_fold_size)
        stop_valid = stop_train + math.ceil(self.validation_size*k_fold_size)
        stop_test = (i+1)*k_fold_size
        yield indices[start_train: stop_train], indices[stop_train: stop_valid], indices[stop_valid: stop_test]