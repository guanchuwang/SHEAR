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



def f(X):
    np.random.seed(0)
    N = X.shape[-1]
    beta = np.random.randn(N - 1) * 100
    y = 0 # (X[:, 0:N] * X[:, 1:]).sum(axis=-1)
    for index in range(X.shape[1]-1):
        y += X[:, index] * X[:, index+1] * beta[index]

    return y


def f_torch(X):
    x_ndim = X.ndim
    if x_ndim == 1:
        X = X.unsqueeze(dim=0)
    # print(X.shape)
    X_shape = X.shape[:-1]
    # print(X_shape)
    X = X.view(-1, X.shape[-1])
    y = f(X)
    # y = X[:, 0] * X[:, 1] * beta[0] + X[:, 1] * X[:, 2] * beta[1] + X[:, 2] * X[:, 3] * beta[2]
    y = y.view(X_shape)
    if x_ndim == 1:
        y = y.squeeze(dim=0)
    return y


def f_numpy(X):
    x_ndim = X.ndim
    if x_ndim == 1:
        X = np.expand_dims(X, axis=0)
    X_shape = X.shape[:-1]
    X = X.reshape(-1, X.shape[-1])
    y = f(X)
    # y = X[:, 0] * X[:, 1] * beta[0] + X[:, 1] * X[:, 2] * beta[1] + X[:, 2] * X[:, 3] * beta[2]
    if len(X_shape) > 1:
        y = y.view(X_shape)
    if x_ndim == 1:
        y = y.squeeze(axis=0)
    return y


def binary(x, bits):
    mask = 2 ** torch.arange(bits) # .to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


@torch.no_grad()
def sub_brute_force_shapley(f, x, reference, mask, feature_index, M):

    set0 = torch.cat((mask, torch.zeros((mask.shape[0], 1)).byte()), dim=1)
    set1 = torch.cat((mask, torch.ones((mask.shape[0], 1)).byte()), dim=1)
    # set01 = torch.cat((set0, set1), dim=0)
    set0[:, [feature_index, -1]] = set0[:, [-1, feature_index]]
    set1[:, [feature_index, -1]] = set1[:, [-1, feature_index]]
    # set01[:, [feature_index, -1]] = set01[:, [-1, feature_index]]
    S = set0.sum(dim=1)
    # weights = 1. / torch.from_numpy(comb(M, S)).type(torch.float) / (M - S)

    weights = 1. / torch.from_numpy(comb(M-1, S)).type(torch.float)
    f_set0 = f(set0 * x + (1-set0) * reference.unsqueeze(dim=0)) # .unsqueeze(dim=0)
    f_set1 = f(set1 * x + (1-set1) * reference.unsqueeze(dim=0)) # .unsqueeze(dim=0)
    # print(set01)
    # f_set01 = f(set01 * x + (1-set01) * reference.unsqueeze(dim=0))
    # N = set0.shape[0]
    # f_set0, f_set1 = f_set01[:N], f_set01[N:]
    # print(set0)
    # print(set1)
    # shapley_value = weights.unsqueeze(dim=0).mm(f_set1 - f_set0)
    shapley_value = 1./M * weights.unsqueeze(dim=0).mm(f_set1 - f_set0)
    # shapley_value = (f_set1 - f_set0).mean()

    # print(shapley_value)
    return shapley_value
    # return (weights.unsqueeze(dim=0) * (f_set1 - f_set0)).sum()

@torch.no_grad()
def brute_force_shapley(f, x, reference, shap_index=None, batch_size=None):

    M = x.shape[1]
    shap_index = torch.arange(M) if shap_index is None else shap_index
    mask_dec = torch.arange(0, 2 ** (M-1))
    mask = binary(mask_dec, M - 1)

    shapley_value = torch.zeros((x.shape[0], len(shap_index)))
    for idx, feature_index in enumerate(shap_index):
        shapley_value[:, idx] = sub_brute_force_shapley(f, x, reference, mask, feature_index, M).squeeze(dim=0)

    return shapley_value

# @torch.no_grad()
# def brute_force_shapley(f, x, reference, shap_index=None, batch_size=None):
#
#     M = x.shape[1]
#     shap_index = torch.arange(M) if shap_index is None else shap_index
#     batch_size = 2**(M-1) if batch_size is None else batch_size
#     n_batch = int(2**(M - 1 - int(np.log2(batch_size))))
#
#     shapley_value = torch.zeros((x.shape[0], len(shap_index)))
#
#     for idx, feature_index in enumerate(shap_index):
#         # print("Feature index: {}".format(feature_index))
#         for batch_idx in range(n_batch):
#             # print(idx, batch_idx)
#             # import time
#             # t0 = time.time()
#             mask_dec = torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
#             mask = binary(mask_dec, M-1)
#             # t1 = time.time()
#             shapley_value[:, idx] += sub_brute_force_shapley(f, x, reference, mask, feature_index, M).squeeze(dim=0)
#             # t2 = time.time()
#             # print(t1-t0, t2-t1)
#     # stop
#     return shapley_value

@torch.no_grad()
def sub_eff_shap(f, x, reference, feature_idx, inter_index, other_index, antithetical=True):

    K = inter_index.sum()
    # other_index_mask = np.random.randint(low=0, high=2, size=(1, other_index.sum()))
    if antithetical:
        other_index_mask_half = torch.randint(low=0, high=2, size=(2 ** (K - 2), other_index.sum()))
        other_index_mask = torch.cat((other_index_mask_half, 1 - torch.flip(other_index_mask_half, dims=(0,)))).type(torch.int)
        other_index_mask_double = torch.cat((other_index_mask, other_index_mask), dim=0)

    else:
        other_index_mask = torch.randint(low=0, high=2, size=(2 ** (K - 1), other_index.sum()))
        other_index_mask_double = torch.cat((other_index_mask, other_index_mask), dim=0)


    # print(other_index_mask)
    # other_index_mask = np.random.randint(low=0, high=2, size=(1, other_index.sum()))
    # other_index_mask = torch.zeros((1, other_index.sum()))
    # other_index_mask = torch.ones((1, other_index.sum()))

    @torch.no_grad()
    def f_mask(x_inter):
        x_mask = torch.zeros((x_inter.shape[0], x.shape[1])).type(x.dtype)
        # print(reference.dtype, x_mask.dtype, other_index_mask.dtype, x.dtype)
        x_mask[:, other_index] = x[:, other_index] * other_index_mask + reference.unsqueeze(dim=0)[:, other_index] * (1 - other_index_mask)
        # x_mask[:, other_index] = x[:, other_index] * other_index_mask_double + reference.unsqueeze(dim=0)[:, other_index] * (1 - other_index_mask_double)
        x_mask[:, inter_index] = x_inter
        # print(x_mask)
        # print(f(x_mask))
        return f(x_mask)

    reference_mask = reference[inter_index]
    x_inter_ = x[:, inter_index]
    shap_values = brute_force_shapley(f_mask, x_inter_, reference_mask, shap_index=feature_idx, batch_size=None)

    # explainer = shap.KernelExplainer(f_mask, np.reshape(reference_mask, (1, len(reference_mask))))
    # shap_values = explainer.shap_values(x[:, inter_index], n_sample=np.power(2, inter_index.sum()-1))

    return shap_values

@torch.no_grad()
def eff_shap(f, x, reference, error_matrix, topK):

    M = x.shape[-1]
    shapley_value = torch.zeros((x.shape[0], M))

    for index in range(M):

        error_vector = error_matrix[index]

        topK_node = (error_vector.argsort()[::-1])[0:topK]
        interactions = [(index, node) for node in topK_node]

        allinter = set([index])
        for inter in interactions:
            if index in inter:
                allinter = allinter | set(inter)

        allinter = np.sort(np.array(list(allinter)).astype(np.int))

        # print(error_vector)
        # print(allinter)

        local_idx = np.where(allinter == index)[0] # .squeeze(axis=0)
        inter_index = np.zeros(M).astype(np.bool)
        inter_index[allinter] = True
        other_index = np.ones(M).astype(np.bool)
        other_index[allinter] = False

        # print(local_idx)
        # print(torch.where(inter_index)[0])
        # print(torch.where(other_index)[0])

        # inter = np.where(inter_index)[0]
        # print("==============")
        # print(index, inter)

        local_shapley_value = sub_eff_shap(f, x, reference, local_idx, inter_index, other_index)
        shapley_value_valid = local_shapley_value # [:, local_idx]
        # print(shapley_value_valid)

        shapley_value[:, index] = shapley_value_valid
        # break

    return shapley_value


@torch.no_grad()
def efs_ablation(f, x, reference, topK):

    M = x.shape[-1]
    shapley_value = torch.zeros((x.shape[0], M))

    for index in range(M):

        allinter = set([index])
        index_buf = list(set(range(0, M)) - set([index]))
        # print(index_buf, topK)
        cooperators = set(np.random.choice(index_buf, int(topK), replace=False))
        allinter = np.array(list(allinter | cooperators)).astype(np.int)

        # print(error_vector)
        # print(index, allinter)

        local_idx = np.where(allinter == index)[0] # .squeeze(axis=0)
        inter_index = np.zeros(M).astype(np.bool)
        inter_index[allinter] = True
        other_index = np.ones(M).astype(np.bool)
        other_index[allinter] = False

        # print(local_idx)
        # print(torch.where(inter_index)[0])
        # print(torch.where(other_index)[0])

        # inter = np.where(inter_index)[0]
        # print("==============")
        # print(index, inter)

        local_shapley_value = sub_eff_shap(f, x, reference, local_idx, inter_index, other_index)
        shapley_value_valid = local_shapley_value # [:, local_idx]
        # print(shapley_value_valid)

        shapley_value[:, index] = shapley_value_valid
        # break
    # stop
    return shapley_value


# class PermutationGame:
#
#     def __init__(self, f, reference, M, batch_size=16, antithetical=False):
#
#         self.M = M
#         self.batch_size = batch_size
#         self.antithetical = antithetical
#         self.queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))
#         self.f = f
#         self.reference = reference
#         self.baseline_value = self.f(self.reference.unsqueeze(dim=0)).repeat((self.batch_size, 1)).type(torch.double)
#
#         for idx in range(batch_size):
#             if antithetical and idx > batch_size // 2:
#                 self.queue[idx] = torch.flip(self.queue[batch_size - 1 - idx], dims=(0,))
#             else:
#                 self.queue[idx] = self.queue[idx, torch.randperm(M)]
#
#     def shapley_value_estimation(self, x):
#
#         arange = torch.arange(self.batch_size)
#
#         S = torch.zeros_like(self.queue).type(torch.long)
#         deltas = torch.zeros_like(self.queue).type(torch.double)
#         baseline_value = self.baseline_value.clone()
#         for index in range(self.M):
#             S[arange, self.queue[:, index]] = 1
#             # print(S.shape, x.shape, reference.shape)
#             x_mask = S * x + (1 - S) * self.reference.unsqueeze(dim=0)
#             cur_value = self.f(x_mask).type(torch.double)
#             deltas[arange, self.queue[:, index]] = (cur_value - baseline_value).squeeze(dim=1)
#             baseline_value = cur_value
#
#         return deltas.mean(dim=0).unsqueeze(dim=0).type(torch.float).detach()
#
# @torch.no_grad()
# def permutation_sample(f, x, reference, batch_size=16, antithetical=False):
#
#     M = x.shape[-1]
#     queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))
#
#     for idx in range(batch_size):
#         if antithetical and idx > batch_size//2:
#             queue[idx] = torch.flip(queue[batch_size-1-idx], dims=(0,))
#         else:
#             queue[idx] = queue[idx, torch.randperm(M)]
#
#     return sub_permutation_sample(f, x, reference, queue, batch_size)
#
#
# @torch.no_grad()
# def sub_permutation_sample(f, x, reference, queue, batch_size):
#
#     M = x.shape[-1]
#     arange = torch.arange(batch_size)
#     S = torch.zeros_like(queue).type(torch.long)
#     S_old = torch.zeros_like(queue).type(torch.long)
#     # val_buf = torch.zeros_like(queue.T).type(torch.float)
#     deltas = torch.zeros_like(queue).type(torch.float)
#     # baseline_value = f(reference.unsqueeze(dim=0)).repeat((batch_size, 1))
#     for index in range(M):
#         S[arange, queue[:, index]] = 1
#         if index == 0:
#             deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S) - f(reference.unsqueeze(dim=0)).repeat((batch_size, 1))).squeeze(dim=1)
#         else:
#             deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S) - f_mask(f, x, reference, S_old)).squeeze(dim=1)
#         S_old = S.clone()
#
#     return deltas.mean(dim=0).unsqueeze(dim=0).detach()


def f_mask(f, x, reference, S):
    x_mask = S * x + (1 - S) * reference.unsqueeze(dim=0)
    return f(x_mask)


@torch.no_grad()
def permutation_sample_parallel(f, x, reference, batch_size=16, antithetical=False):

    M = x.shape[-1]
    queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))

    for idx in range(batch_size):
        if antithetical and idx > batch_size//2:
            queue[idx] = torch.flip(queue[batch_size-1-idx], dims=(0,))
        else:
            queue[idx] = queue[idx, torch.randperm(M)]

    arange = torch.arange(batch_size)
    deltas = torch.zeros_like(queue).type(torch.float)
    # deltas = []

    S = torch.zeros_like(queue).type(torch.long)
    # S_ = torch.zeros_like(queue).type(torch.long)

    S_buf = []
    for index in range(M):
        S[arange, queue[:, index]] = 1
        S_buf.append(S.clone())
    # S_buf = torch.cat(S_buf, dim=0)

    for index in range(M):
        # S[arange, queue[:, index]] = 1
        S = S_buf[index]
        if index == 0:
            deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S) - f(reference.unsqueeze(dim=0)).repeat((batch_size, 1))).squeeze(dim=1)
        else:
            S_ = S_buf[index-1]
            deltas[arange, queue[:, index]] = (f_mask(f, x, reference, S) - f_mask(f, x, reference, S_)).squeeze(dim=1)

        # S_ = S.clone()

    return deltas.mean(dim=0).unsqueeze(dim=0)

    # val_buf = [f(reference.unsqueeze(dim=0)).repeat((batch_size, 1)).squeeze(dim=1)]
    # for index in range(M):
    #     S[arange, queue[:, index]] = 1
    #     val_buf.append(f(S * x + (1 - S) * reference.unsqueeze(dim=0)).squeeze(dim=1))
    #
    # for index in range(M):
    #     deltas[arange, queue[:, index]] = val_buf[index+1] - val_buf[index]
    #     deltas
    #
    # return deltas.mean(dim=0).unsqueeze(dim=0)

# @torch.no_grad()
# def permutation_sample_serial(f, x, reference, batch_size=16, antithetical=False):
#
#     M = x.shape[-1]
#     queue = torch.arange(M).unsqueeze(dim=0).repeat((batch_size, 1))
#
#     for idx in range(batch_size):
#         if antithetical and idx > batch_size//2:
#             queue[idx] = torch.flip(queue[batch_size-1-idx], dims=(0,))
#         else:
#             queue[idx] = queue[idx, torch.randperm(M)]
#
#     arange = torch.arange(batch_size)
#     S = torch.zeros_like(queue).type(torch.long)
#     deltas = torch.zeros_like(queue).type(torch.float)
#     baseline_value = f(reference.unsqueeze(dim=0)).repeat((batch_size, 1))
#     for index in range(M):
#         S[arange, queue[:, index]] = 1
#         # print(S.shape, x.shape, reference.shape)
#         x_mask = S * x + (1-S) * reference.unsqueeze(dim=0)
#         cur_value = f(x_mask)
#         deltas[arange, queue[:, index]] = (cur_value - baseline_value).squeeze(dim=1)
#         baseline_value = cur_value
#
#     return deltas.mean(dim=0).unsqueeze(dim=0)


# @torch.no_grad()
# def sub_permutation_sample(f, x, reference, feature_index, batch_size=16, antithetical=False):
#
# @torch.no_grad()
# def sub_eff_permutation_sample(f, x, reference, inter_index, other_index):
#
#     # other_index_mask = np.random.randint(low=0, high=2, size=(1, other_index.sum()))
#     other_index_mask = torch.randint(low=0, high=2, size=(1, other_index.sum()))
#     # other_index_mask = torch.zeros((1, other_index.sum()))
#     # other_index_mask = torch.ones((1, other_index.sum()))
#
#     @torch.no_grad()
#     def f_mask(x_inter):
#         x_mask = torch.zeros((x_inter.shape[0], x.shape[1]))
#         # print(reference.dtype, x_mask.dtype, other_index_mask.dtype)
#         x_mask[:, other_index] = x[:, other_index] * other_index_mask + reference.unsqueeze(dim=0)[:, other_index] * (1 - other_index_mask)
#         x_mask[:, inter_index] = x_inter
#         return f(x_mask)
#
#     reference_mask = reference[inter_index]
#     x_inter_ = x[:, inter_index]
#     shap_values = brute_force_shapley(f_mask, x_inter_, reference_mask, shap_index=None, batch_size=None)
#
#     # explainer = shap.KernelExplainer(f_mask, np.reshape(reference_mask, (1, len(reference_mask))))
#     # shap_values = explainer.shap_values(x[:, inter_index], n_sample=np.power(2, inter_index.sum()-1))
#
#     return shap_values
#
#
# @torch.no_grad()
# def eff_permutation_sample(f, x, reference, error_matrix, topK_grad, topK_permut):
#
#     M = x.shape[-1]
#     shapley_value = torch.zeros((x.shape[0], M))
#     for index in range(M):
#
#         error_vector = error_matrix[index]
#
#         topK_node = (error_vector.argsort()[::-1])[0:topK_grad]
#         interactions = [(index, node) for node in topK_node]
#
#         allinter = set([])
#         for inter in interactions:
#             if index in inter:
#                 allinter = allinter | set(inter)
#
#         allinter = np.sort(np.array(list(allinter)).astype(np.int))
#
#         # print(error_vector)
#         # print(allinter)
#
#         local_idx = np.where(allinter == index)[0] # .squeeze(axis=0)
#         inter_index = np.zeros(M).astype(np.bool)
#         inter_index[allinter] = True
#         other_index = np.ones(M).astype(np.bool)
#         other_index[allinter] = False
#
#         # inter = np.where(inter_index)[0]
#         # print("==============")
#         # print(index, inter)
#
#         local_shapley_value = sub_eff_permutation_sample(f, x, reference, inter_index, other_index)
#         shapley_value_valid = local_shapley_value[:, local_idx]
#         # print(shapley_value_valid)
#
#         shapley_value[:, index] = shapley_value_valid
#         # break
#
#     return shapley_value



# def random_rank(error_vector):
#     base = 1
#     error_vector_normalized = (error_vector + base) / (error_vector + base).sum()
#
#     N = 100
#     rd_float = np.random.rand(M, N)
#     rd_01 = (rd_float < error_vector_normalized.reshape(-1, 1)).astype(np.int)
#     rd_score = rd_01.sum(axis=1) / N


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