from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Covert_to_Time_Series():
    def __init__(self):
        pass
    def create_lstm_data_torch(self, X, y, seq_length):
        X_df = []
        y_df = []
        for i in range(X.shape[0]-seq_length+1):
            X_df.append(X[i:i+seq_length])
            y_df.append(y[i+seq_length-1])
        return torch.from_numpy(np.array(X_df)).float(), torch.from_numpy(np.array(y_df)).float()
    def create_lstm_data(self, X, y, seq_length):
        X_df = []
        y_df = []
        for i in range(X.shape[0]-seq_length+1):
            X_df.append(X[i:i+seq_length])
            y_df.append(y[i+seq_length-1])
        return np.array(X_df), np.array(y_df)