import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import sys, os
sys.path.append("./credit-dataset")

from credit_model import mlp, Model_for_shap
from train_credit import load_data, validate, save_checkpoint
# from captum.attr import *
from shap_utils import brute_force_shapley


def brute_force_kernel_shap(model, data_loader, reference):
    # model.eval()
    val_acc, _ = validate(model.model, data_loader, None, 0)
    print("Testing ACC: {}".format(val_acc))

    # print(reference)
    # reference = reference.numpy()

    shapley_value_buf = []
    shapley_rank_buf = []
    for index, (_, y, x) in enumerate(data_loader):

        # x = x[0].unsqueeze(dim=0)
        # print(x, reference)
        attr = brute_force_shapley(model.forward_1, x, reference)

        shapley_value_buf.append(attr)
        shapley_rank_buf.append(attr.argsort(axis=1))
        print(index)
        # print(attr)
        if index == 100:
            break

    shapley_value_buf = torch.cat(shapley_value_buf, dim=0)
    shapley_rank_buf = torch.cat(shapley_rank_buf, dim=0)

    return shapley_value_buf, shapley_rank_buf

if __name__ == "__main__":

    # datasets_torch, cate_attrib_book, dense_feat_index, sparse_feat_index = load_data('./adult-dataset/adult.csv', val_size=0.2, test_size=0.2, run_num=0) #
    # x_train, y_train, z_train, x_val, y_val, z_val, x_test, y_test, z_test = datasets_torch
    # print(x_train.shape)

    # print(cate_attrib_book)

    model_checkpoint_fname = "./credit-dataset/model_credit_m_1_r_0.pth.tar"

    checkpoint = torch.load(model_checkpoint_fname)
    # print(checkpoint)

    print("Testing dataset: {} samples.".format(len(checkpoint["test_data_x"])))

    x_test = checkpoint["test_data_x"]
    y_test = checkpoint["test_data_y"]
    z_test = checkpoint["test_data_z"]
    dense_feat_index = checkpoint["dense_feat_index"]
    sparse_feat_index = checkpoint["sparse_feat_index"]
    cate_attrib_book = checkpoint["cate_attrib_book"]

    model = mlp(input_dim= checkpoint["input_dim"] ,
                output_dim=checkpoint["output_dim"],
                layer_num= checkpoint["layer_num"] ,
                hidden_dim=checkpoint["hidden_dim"],
                activation=checkpoint["activation"])

    model.load_state_dict(checkpoint["state_dict"])

    model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

    data_loader = DataLoader(TensorDataset(x_test, y_test, z_test), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    reference_dense = x_test[:, dense_feat_index].mean(dim=0)
    reference_sparse = -torch.ones_like(sparse_feat_index).type(torch.long) # -1 for background noise
    # reference_sparse = torch.zeros_like(sparse_feat_index).type(torch.long)
    reference = torch.cat((reference_dense, reference_sparse), dim=0)

    # print(x_test[:, 0:5])
    # print(z_test[:, 0:5])

    checkpoint["reference"] = reference
    checkpoint["cate_attrib_book"] = cate_attrib_book
    checkpoint["dense_feat_index"] = dense_feat_index
    checkpoint["sparse_feat_index"] = sparse_feat_index

    shapley_value, shapley_rank = brute_force_kernel_shap(model_for_shap, data_loader, reference)

    checkpoint["test_shapley_value"] = shapley_value
    checkpoint["test_shapley_ranking"] = shapley_rank

    # save_checkpoint(model_checkpoint_fname,
    #                 round_index=checkpoint["round_index"],
    #                 state_dict=checkpoint["state_dict"],
    #                 layer_num =checkpoint["layer_num"],
    #                 input_dim =checkpoint["input_dim"],
    #                 hidden_dim=checkpoint["hidden_dim"],
    #                 output_dim=checkpoint["output_dim"],
    #                 activation=checkpoint["activation"],
    #                 test_data_x=checkpoint["test_data_x"],
    #                 test_data_y=checkpoint["test_data_y"],
    #                 test_data_z=checkpoint["test_data_z"],
    #                 reference=checkpoint["reference"],
    #                 test_shapley_value=checkpoint["test_shapley_value"],
    #                 test_shapley_ranking=checkpoint["test_shapley_ranking"],
    #                 cate_attrib_book=cate_attrib_book,
    #                 dense_feat_index=dense_feat_index,
    #                 sparse_feat_index=sparse_feat_index,
    #                 )



