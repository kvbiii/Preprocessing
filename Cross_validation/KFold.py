from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class KFold():
    def __init__(self, n_splits, shuffle=False, stratify=False, random_state=17):
        self.random_state = random_state
        random.seed(self.random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.stratify = stratify         

    def split(self, X, y=None):
        n_samples = len(X)
        k_fold_size = int(n_samples//self.n_splits)
        reszta = int(n_samples%self.n_splits)
        if(self.shuffle==True and self.stratify==True):
            self.liczba_klas = len(np.unique(y))
            #Classification or multiclassification
            if(self.liczba_klas <= 30):
                for i in range(self.liczba_klas):
                    globals()[f"indices_{i}"] = [m for m in range(X.shape[0]) if y[m]==i]
                    globals()[f"indices_test_{i}_left"] = globals()[f"indices_{i}"].copy()
                    globals()[f"k_{i}_size"] = int(len(globals()[f"indices_{i}"])//self.n_splits)
                    globals()[f"reszta_{i}"] = int(len(globals()[f"indices_{i}"])%self.n_splits)
            #Regression
            else:
                self.liczba_klas = 30
                #The empirical normal distribution of our set of y
                ecdf = ECDF(y.flatten())
                y_scaled = ecdf(y.flatten())
                bins = np.linspace(0, 1, self.liczba_klas)
                #index stores the values of the baskets to which a given y has been assigned.
                index = np.digitize(y_scaled, bins)
                for i in range(0, self.liczba_klas):
                    globals()[f"indices_{i}"] = [m for m in range(X.shape[0]) if index[m]==i+1]
                    globals()[f"indices_test_{i}_left"] = globals()[f"indices_{i}"].copy()
                    globals()[f"k_{i}_size"] = int(len(globals()[f"indices_{i}"])//self.n_splits)
                    globals()[f"reszta_{i}"] = int(len(globals()[f"indices_{i}"])%self.n_splits)
            for i in range(self.n_splits):
                indices_train, indices_test = np.array([]), np.array([])
                for j in range(0, self.liczba_klas):
                    if(j != 0 and w_kolejnym_bez==True):
                        globals()[f"indices_test_{j}"] = random.sample(globals()[f"indices_test_{j}_left"], k=globals()[f"k_{j}_size"])
                    else:
                        globals()[f"indices_test_{j}"] = random.sample(globals()[f"indices_test_{j}_left"], k=globals()[f"k_{j}_size"]+1 if globals()[f"reszta_{j}"]>0 else globals()[f"k_{j}_size"])
                    globals()[f"indices_test_{j}_left"] = [m for m in globals()[f"indices_test_{j}_left"] if m not in globals()[f"indices_test_{j}"]]
                    globals()[f"indices_train_{j}"] = np.setdiff1d(globals()[f"indices_{j}"], globals()[f"indices_test_{j}"])
                    indices_train = np.concatenate([indices_train, globals()[f"indices_train_{j}"]])
                    indices_test = np.concatenate([indices_test, globals()[f"indices_test_{j}"]])
                    #Here in case, so as not to duplicate too many test residuals at once
                    if(globals()[f"reszta_{j}"] > 0):
                        w_kolejnym_bez = True
                    else:
                        w_kolejnym_bez = False
                    globals()[f"reszta_{j}"] = globals()[f"reszta_{j}"] - 1
                yield indices_train.astype(int), indices_test.astype(int)
        elif(self.shuffle == True):
            all_indices = [i for i in range(X.shape[0])]
            indices_test_left = [i for i in range(X.shape[0])]
            for i in range(self.n_splits):
                indices_test = random.sample(indices_test_left, k=k_fold_size+1 if reszta>0 else k_fold_size)
                indices_test_left = [j for j in indices_test_left if j not in indices_test]
                #Difference between two sets
                indices_train = np.setdiff1d(all_indices, indices_test)
                reszta = reszta-1
                yield indices_train, indices_test
        else:
            indices = np.arange(n_samples)
            for i in range(self.n_splits):
                try:
                    test_start = test_stop
                except:
                    test_start = 0
                if(reszta != 0):
                    test_stop = test_start + k_fold_size + 1
                    reszta = reszta - 1
                else:
                    test_stop = test_start + k_fold_size
                yield [i for i in indices if i not in indices[test_start : test_stop]], indices[test_start : test_stop]