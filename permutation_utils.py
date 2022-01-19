import scipy.special
import numpy as np
import itertools
import torch
from scipy.special import comb, perm
import shap

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class PermutationGame:

    def __init__(self, f, reference, batch_size=16, antithetical=False):

        self.M = reference.shape[-1]
        self.batch_size = batch_size
        self.antithetical = antithetical
        self.f = f
        self.reference = reference
        self.baseline_value = self.f(self.reference.unsqueeze(dim=0)).repeat((self.batch_size, 1)).type(torch.double)

    def queue_resample(self):

        self.queue = torch.arange(self.M).unsqueeze(dim=0).repeat((self.batch_size, 1))
        for idx in range(self.batch_size):
            if self.antithetical and idx > self.batch_size // 2:
                self.queue[idx] = torch.flip(self.queue[self.batch_size - 1 - idx], dims=(0,))
            else:
                self.queue[idx] = self.queue[idx, torch.randperm(self.M)]

    def shapley_value_estimation(self, x):

        self.queue_resample()
        arange = torch.arange(self.batch_size)

        S = torch.zeros_like(self.queue).type(torch.long)
        deltas = torch.zeros_like(self.queue).type(torch.double)
        baseline_value = self.baseline_value.clone()
        for index in range(self.M):
            S[arange, self.queue[:, index]] = 1
            # print(S.shape, x.shape, reference.shape)
            x_mask = S * x + (1 - S) * self.reference.unsqueeze(dim=0)
            cur_value = self.f(x_mask).type(torch.double)
            deltas[arange, self.queue[:, index]] = (cur_value - baseline_value).squeeze(dim=1)
            baseline_value = cur_value

        return deltas.mean(dim=0).unsqueeze(dim=0).type(torch.float).detach()



@torch.no_grad()
def permutation_sample(f, x, reference, batch_size=16, antithetical=False):

    M = x.shape[-1]
    queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))

    for idx in range(batch_size):
        if antithetical and idx > batch_size//2:
            queue[idx] = torch.flip(queue[batch_size-1-idx], dims=(0,))
        else:
            queue[idx] = queue[idx, torch.randperm(M)]

    arange = torch.arange(batch_size)
    S = torch.zeros_like(queue).type(torch.long)
    deltas = torch.zeros_like(queue).type(torch.float)
    baseline_value = f(reference.unsqueeze(dim=0)).repeat((batch_size, 1))
    for index in range(M):
        S[arange, queue[:, index]] = 1
        # print(S.shape, x.shape, reference.shape)
        x_mask = S * x + (1-S) * reference.unsqueeze(dim=0)
        cur_value = f(x_mask)
        deltas[arange, queue[:, index]] = (cur_value - baseline_value).squeeze(dim=1)
        baseline_value = cur_value

    return deltas.mean(dim=0).unsqueeze(dim=0)




class Model_for_shap:

    def __init__(self, model):
        self.model = model

    def predict_prep(self, columns, **kwargs):
        self.columns = columns
        self.predict_args = kwargs

        # print(self.data_column_keys)

    def data_transform(self, x):
        x_dataframe = pd.DataFrame(x, columns=self.columns)
        x_dict = {name: x_dataframe[name] for name in self.columns}
        return x_dict

    def predict(self, x):
        x_dict = self.data_transform(x)
        return self.model.predict(x_dict, **self.predict_args).squeeze(axis=1)

    def predict_tensor(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x_numpy = x.numpy()
        y = self.predict(x_numpy)
        y = y.reshape(x_shape[:-1])
        return torch.from_numpy(y).type(torch.float)



class Model_for_captum:

    def __init__(self, model):
        self.model = model

    def predict_prep(self, columns):
        self.columns = columns

    def data_transform(self, x):
        x_dataframe = pd.DataFrame(x, columns=self.columns)
        x_dict = {name: x_dataframe[name] for name in self.columns}
        return x_dict

    def predict(self, x, batch_size):
        x_dict = self.data_transform(x)
        return self.model.predict(x_dict, batch_size)

    def predict_tensor(self, x, batch_size):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x_numpy = x.numpy()
        y = self.predict(x_numpy, batch_size)
        return torch.from_numpy(y).type(torch.float)

    def sparse_feature_harsh(self):

        pass

    def sparse_feature_deharsh(self):

        pass