import random
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import math
from sklearn.metrics import *
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import nan_euclidean_distances
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import plotly
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import express as px
import pandas as pd
from sklearn.tree import *
from torch.autograd import Variable
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))
from scipy.stats import rankdata, norm, mode, chi2, f
from statsmodels.distributions.empirical_distribution import ECDF
from arch.bootstrap import MovingBlockBootstrap