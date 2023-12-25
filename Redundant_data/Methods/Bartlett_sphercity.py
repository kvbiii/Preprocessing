from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

def bartlett_sphercity(data):
    corr_matrix = np.corrcoef(data, rowvar=False)
    N = data.shape[0]
    M = data.shape[1]
    chi_square = -(N-1-(2*M-5)/6)*np.log(np.linalg.det(corr_matrix))
    degrees_of_freedom = (M**2-M)/2
    p_value = 1 - chi2.cdf(x=chi_square, df=degrees_of_freedom)
    return chi_square, p_value