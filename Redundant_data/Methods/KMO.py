from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

def Kaiser_Mayer_Olkin(data):
    dataframe = pd.DataFrame(data)
    corr = dataframe.corr(method="spearman")
    corr = np.asarray(corr)
    corr_inv = np.linalg.inv(corr)
    kmo_numerator = np.sum(np.square(corr)) - np.sum(np.square(np.diag(corr)))
    partial_corr = np.zeros((corr.shape[0], corr.shape[0]))
    for i in range(corr.shape[0]):
        for j in range(corr.shape[0]):
            partial_corr[i, j] = -corr_inv[i, j]/np.sqrt(corr_inv[i, i]*corr_inv[j, j])
            partial_corr[j, i] = partial_corr[i, j]
    kmo_denominator = kmo_numerator + np.sum(np.square(partial_corr)) - np.sum(np.square(np.diag(partial_corr)))
    return kmo_numerator/kmo_denominator