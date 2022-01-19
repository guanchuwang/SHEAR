import torch
import numpy as np


def pearsonr_corr(x, y):
    # x, y = x.reshape(-1), y.reshape(-1)
    # return torch.mean((x - x.mean())*(y - y.mean()))/(x.std(unbiased=False) * y.std(unbiased=False))

    corr = torch.mean((x - x.mean(dim=0))*(y - y.mean(dim=0)), dim=0)/(x.std(dim=0, unbiased=False) * y.std(dim=0, unbiased=False))
    return corr.mean()

def pearsonr_evaluate(feature_weights, ground_truth_weights):
    corr = [np.corrcoef(a, b)[0, 1] for a, b in zip(feature_weights, ground_truth_weights)]
    corr = np.nan_to_num(corr, nan=0, posinf=0, neginf=0)
    return np.mean(corr)