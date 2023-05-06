from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Time_Series():
  def __init__(self, n_splits):
    self.n_splits = n_splits
  
  def get_n_splits(self, df, y, groups):
    return self.n_splits
  
  def split(self, df, y=None, groups=None):
    n_samples = len(df)
    test_len = int(len(df)/(self.n_splits+1))
    indices = np.arange(n_samples)
    for i in range(1, self.n_splits+1):
      mid = int(i*test_len)
      stop = int((i+1)*test_len)
      yield indices[: mid], indices[mid: stop]