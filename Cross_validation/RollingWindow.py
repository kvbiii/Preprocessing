from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Rolling_window():
  def __init__(self, n_splits, test_size=0.2):
    self.n_splits = n_splits
    self.test_size = test_size
          
  def get_n_splits(self, df, y, groups):
    return self.n_splits
          
  def split(self, df, y=None, groups=None):
    n_samples = len(df)
    test_len = int(len(df)*(self.test_size))
    train_len = int(len(df)*(1-self.test_size))
    indices = np.arange(n_samples)
    for i in range(1, self.n_splits+1):
      start = int((i-1)*int(test_len)/self.n_splits)
      stop = int(train_len+i*int(test_len)/self.n_splits)
      mid = int(train_len+(i-1)*int(test_len)/self.n_splits)
      yield indices[start: mid], indices[mid: stop]