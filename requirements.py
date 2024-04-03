import random
import numpy as np
import math
from sklearn.metrics import *
from sklearn.preprocessing import OrdinalEncoder, KernelCenterer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import *
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
import plotly
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import express as px
from matplotlib import pyplot as plt
import pandas as pd
from torch.autograd import Variable
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))
from scipy.stats import rankdata, norm, mode, chi2, f, t
from scipy.sparse.linalg import eigsh
from scipy.cluster.hierarchy import dendrogram
from scipy.special import comb
from statsmodels.distributions.empirical_distribution import ECDF
from arch.bootstrap import MovingBlockBootstrap, StationaryBootstrap
from itertools import combinations
from collections import Counter
import shap
import optuna
import typing