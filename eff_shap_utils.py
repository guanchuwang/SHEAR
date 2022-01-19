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


class Efficient_shap:
    def __init__(self, f, reference, topK, noise=0):

        self.f = f
        self.topK = min(topK, reference.shape[-1]-1)
        self.M = reference.shape[-1]
        self.M_inter = topK + 1
        self.reference = reference
        self.noise = noise
        self.local_index_buf = torch.zeros(self.M,).type(torch.int)
        self.inter_index_buf = torch.zeros(self.M,self.M).type(torch.bool)
        self.other_index_buf = torch.zeros(self.M,self.M).type(torch.bool)
        mask_dec = torch.arange(0, 2 ** (self.M_inter - 1))
        self.mask = binary(mask_dec, self.M_inter - 1)

    @torch.no_grad()
    def update_attribute(self, **kwargs):

        # print(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # print(self.topK)
        # print(self.reference.shape[-1])

        self.M = self.reference.shape[-1]
        self.topK = min(self.topK, self.reference.shape[-1] - 1)
        self.M_inter = self.topK + 1
        self.local_index_buf = torch.zeros(self.M, ).type(torch.int)
        self.inter_index_buf = torch.zeros(self.M, self.M).type(torch.bool)
        self.other_index_buf = torch.zeros(self.M, self.M).type(torch.bool)
        mask_dec = torch.arange(0, 2 ** (self.M_inter - 1))
        self.mask = binary(mask_dec, self.M_inter - 1)

        # if self.topK > self.reference.shape[-1] - 1:
        #     self.topK = self.reference.shape[-1] - 1
        #     self.M_inter = self.topK + 1
        #     mask_dec = torch.arange(0, 2 ** (self.M_inter - 1))
        #     self.mask = binary(mask_dec, self.M_inter - 1)

    @torch.no_grad()
    def feature_selection(self, error_matrix):

        # print(error_matrix)
        for index in range(self.M):

            error_vector = error_matrix[index]

            topK_node = (error_vector.argsort()[::-1])[0:self.topK]
            interactions = [(index, node) for node in topK_node]

            allinter = set([index])
            for inter in interactions:
                if index in inter:
                    allinter = allinter | set(inter)

            allinter = np.sort(np.array(list(allinter)).astype(np.int))

            local_idx = np.where(allinter == index)[0]  # .squeeze(axis=0)
            inter_index = np.zeros(self.M).astype(np.bool)
            inter_index[allinter] = True
            other_index = np.ones(self.M).astype(np.bool)
            other_index[allinter] = False

            self.local_index_buf[index] = torch.from_numpy(local_idx).type(torch.int)
            self.inter_index_buf[index] = torch.from_numpy(inter_index).type(torch.bool)
            self.other_index_buf[index] = torch.from_numpy(other_index).type(torch.bool)

            # print(self.local_index_buf[index])
            # print(torch.where(self.inter_index_buf[index])[0])
            # print(torch.where(self.other_index_buf[index])[0])
        # stop
            # print(error_vector)
            # print(topK_node)
            # print(inter_index)

    @torch.no_grad()
    def brute_force_forward(self):

        set0 = torch.cat((self.mask, torch.zeros((self.mask.shape[0], 1)).byte()), dim=1)
        set1 = torch.cat((self.mask, torch.ones((self.mask.shape[0], 1)).byte()), dim=1)
        # set0[:, [self.local_index, -1]] = set0[:, [-1, self.local_index]]
        # set1[:, [self.local_index, -1]] = set1[:, [-1, self.local_index]]
        set01 = torch.cat((set0, set1), dim=0)
        set01[:, [self.local_index, -1]] = set01[:, [-1, self.local_index]]
        S = set0.sum(dim=1)
        weights = 1. / torch.from_numpy(comb(self.M_inter-1, S)).type(torch.float)
        # print(set0)
        # print(set1)
        # print(set01.shape)
        # print(self.x_inter.shape)
        # print(self.reference_inter.shape)

        # f_set0 = self.f_mask(set0 * self.x_inter + (1-set0) * self.reference_inter.unsqueeze(dim=0))
        # f_set1 = self.f_mask(set1 * self.x_inter + (1-set1) * self.reference_inter.unsqueeze(dim=0))
        f_set01 = self.f_mask(set01 * self.x_inter + (1-set01) * self.reference_inter.unsqueeze(dim=0))
        N = set0.shape[0]
        f_set0, f_set1 = f_set01[:N], f_set01[N:]
        shapley_value = 1./self.M_inter * weights.unsqueeze(dim=0).mm(f_set1 - f_set0)
        # print(self.M)
        # print(weights)

        return shapley_value

    # sub_eff_shap(f, x, reference, feature_idx, inter_index, other_index):

    @torch.no_grad()
    def f_mask(self, x_inter):
        x_mask = torch.zeros((x_inter.shape[0], self.x.shape[1])).type(self.x.dtype)
        # print(self.x[:, self.other_index].shape, self.other_index_mask.shape)
        # print(reference.dtype, x_mask.dtype, other_index_mask.dtype, x.dtype)
        x_mask[:, self.other_index] = self.x[:, self.other_index] * self.other_index_mask_double + self.reference.unsqueeze(dim=0)[:, self.other_index] * (1 - self.other_index_mask_double)
        # x_mask[:, self.other_index] = self.x[:, self.other_index] * self.other_index_mask + self.reference.unsqueeze(dim=0)[:, self.other_index] * (1 - self.other_index_mask)
        x_mask[:, self.inter_index] = x_inter
        # print(x_mask)
        # print(self.f(x_mask))
        return self.f(x_mask)

    @torch.no_grad()
    def sub_forward(self):

        K = self.inter_index.sum()
        if K > 1:
            other_index_mask_half = torch.randint(low=0, high=2, size=(2 ** (K - 2), self.other_index.sum()))
            self.other_index_mask = torch.cat((other_index_mask_half, 1 - torch.flip(other_index_mask_half, dims=(0,)))).type(torch.int)
            # print(self.other_index_mask.shape)

            if self.noise > 0:
                noise = (torch.rand(self.other_index_mask.shape) < self.noise).type(torch.int)
                self.other_index_mask = ((self.other_index_mask + noise) % 2).type(torch.int)

        else:
            # other_index_mask = torch.randint(low=0, high=2, size=(1, self.other_index.sum())).type(torch.int)
            self.other_index_mask = torch.ones(1, self.other_index.sum()).type(torch.int)
            # other_index_mask = torch.zeros(1, self.other_index.sum()).type(torch.int)

        # print(self.other_index_mask)
        # stop

        self.other_index_mask_double = torch.cat((self.other_index_mask, self.other_index_mask), dim=0)

        self.reference_inter = self.reference[self.inter_index]
        self.x_inter = self.x[:, self.inter_index]
        shap_value = self.brute_force_forward()

        # explainer = shap.KernelExplainer(f_mask, np.reshape(reference_mask, (1, len(reference_mask))))
        # shap_values = explainer.shap_values(x[:, inter_index], n_sample=np.power(2, inter_index.sum()-1))

        return shap_value

    @torch.no_grad()
    def forward(self, x):

        shapley_value_buf = torch.zeros((x.shape[0], self.M))
        # print(self.topK)
        # print(self.reference.shape[-1])

        for index in range(self.M):

            # self.index = index
            self.local_index = self.local_index_buf[index]
            self.inter_index = self.inter_index_buf[index]
            self.other_index = self.other_index_buf[index]
            self.x = x

            # print(self.inter_index)
            # print(self.other_index)

            shapley_value = self.sub_forward() # [self.local_index])
            # print(shapley_value)

            shapley_value_buf[:, index] = shapley_value
            # break

        return shapley_value_buf




def binary(x, bits):
    mask = 2 ** torch.arange(bits) # .to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
