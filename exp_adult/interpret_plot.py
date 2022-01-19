import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd

import numpy as np

import sys, os
sys.path.append("../")
sys.path.append("../adult-dataset")

import tqdm
from adult_model import mlp, Model_for_shap
from train_adult import load_data, save_checkpoint
from captum.attr import *
import shap
from shap_utils import eff_shap
from eff_shap_utils import Efficient_shap
import argparse
import time
from shap_utils import brute_force_shapley

import matplotlib.pyplot as plt

# sys.path.append("../DeepCTR-Torch/deepctr_torch/models")


def get_second_order_grad(model, x, device=None):

    # x = torch.FloatTensor(x).to(device)
    # print(x.dtype)

    with torch.set_grad_enabled(True):

        if x.nelement() < 2:
            return np.array([])

        x.requires_grad = True

        y = model(x)
        grads = autograd.grad(y, x, create_graph=True)[0].squeeze()

        grad_list = []
        for j, grad in enumerate(grads):
            grad2 = autograd.grad(grad, x, retain_graph = True)[0].squeeze()
            grad_list.append(grad2)

        grad_matrix = torch.stack(grad_list)
        return torch.abs(grad_matrix)


@torch.no_grad()
def distance_estimation(x, reference, dense_feat_index, cate_attrib_book):

    reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
    dense_feat_num = dense_feat_index.shape[0]
    reference = torch.cat((reference[0:dense_feat_num], reference_sparse), dim=0)

    distance_vector = torch.abs(x - reference.unsqueeze(dim=0)).squeeze(dim=0)

    distance_i, distance_j = torch.meshgrid(distance_vector, distance_vector)
    distance_matrix = distance_i * distance_j

    return distance_matrix


@torch.no_grad()
def error_term_estimation(mlp, x, reference, dense_feat_index, sparse_feat_index, cate_attrib_book):
    interaction_scores = {}

    # if grad_gpu == -1:
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda:" + str(grad_gpu))
    #
    # mlp = mlp.to(device)

    device = torch.device("cpu")

    grad_matrix_1 = get_second_order_grad(mlp, x, device)
    distance_matrix = distance_estimation(x, reference, dense_feat_index, cate_attrib_book)

    # reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
    # dense_feat_num = dense_feat_index.shape[0]
    # reference = torch.cat((reference[0:dense_feat_num], reference_sparse), dim=0)

    error_matrix = grad_matrix_1 * distance_matrix

    dense_feat_num = dense_feat_index.shape[0]
    sparse_feat_num = sparse_feat_index.shape[0]

    error_matrix_comb = torch.zeros((dense_feat_num + sparse_feat_num,
                                    dense_feat_num + sparse_feat_num))

    error_matrix_comb[0:dense_feat_num, 0:dense_feat_num] = error_matrix[0:dense_feat_num, 0:dense_feat_num]

    block_len = torch.tensor([1 for x in dense_feat_index] + [x.shape[-1] for x in cate_attrib_book]).type(torch.int)
    for feature_index_i in range(dense_feat_num + sparse_feat_num):

        for feature_index_j in range(dense_feat_num + sparse_feat_num):

            if feature_index_i == feature_index_j:
                error_matrix_comb[feature_index_i, feature_index_j] = 0
                continue

            elif feature_index_i < dense_feat_num and feature_index_j < dense_feat_num:
                continue

            feature_i_dim = block_len[feature_index_i]
            feature_j_dim = block_len[feature_index_j]

            index_strt_i = block_len[0:feature_index_i].sum()
            index_strt_j = block_len[0:feature_index_j].sum()

            error_matrix_comb[feature_index_i, feature_index_j] = torch.max(error_matrix[index_strt_i:index_strt_i+feature_i_dim,
                                                                           index_strt_j:index_strt_j+feature_j_dim])
    # print(error_matrix_comb)

    return error_matrix_comb


@torch.no_grad()
def efficient_shap(model_for_shap, data_loader, reference, dense_feat_index, sparse_feat_index, cate_attrib_book): # , shapley_value_gt, shapley_ranking_gt):
    # model.eval()

    shapley_value_buf = []
    shapley_rank_buf = []
    topK = torch.log2(torch.tensor(args.sample_num)).type(torch.long) - 1
    MSE_buf = []
    mAP_buf = []
    total_time = 0

    eff_shap_agent = Efficient_shap(model_for_shap, reference, topK)

    for index, (x, y, z, sh_gt, rk_gt, error_matrix) in enumerate(data_loader):

        y_pred = model_for_shap(z).detach().numpy()
        print(y_pred)

        if y_pred[0] < 1:
            continue

        x = x[0].unsqueeze(dim=0)
        z = z[0].unsqueeze(dim=0)
        shapley_gt = sh_gt
        ranking_gt = rk_gt
        feat_num = z.shape[-1]
        error_matrix = error_matrix.squeeze(dim=0).numpy()

        eff_shap_agent.feature_selection(error_matrix)

        # error_matrix = error_term_estimation(model_grad, x, reference, dense_feat_index, sparse_feat_index, cate_attrib_book).detach().numpy()

        # shapley_groundtruth = torch.tensor([[ 2.1027e-03,  2.5598e-04,  9.2007e-03,  1.3041e-04,  2.0145e-03,
        #   1.2062e-02,  8.6793e-02, -3.4935e-01, -6.9856e-02, -2.1126e-01,
        #   1.8807e-02,  5.5653e-02, -3.4564e-01]])

        # print(z)
        # print(shapley_gt)
        # print(shapley_groundtruth)

        shapley_value_gt = brute_force_shapley(model_for_shap, z, reference)

        attr_buf = []
        for idx in range(args.circ_num):
            attr = eff_shap_agent.forward(z)
            attr_buf.append(attr)

        attr_buf = torch.cat(attr_buf, dim=0)
        attr = attr_buf.mean(dim=0).reshape(1, -1)

        print(z)
        print(attr)
        print(attr.sum())

        expected_value = model_for_shap(reference.unsqueeze(dim=0)).detach().numpy()
        print(expected_value)
        print(shapley_value_gt)

        x_label = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country", "sex"]

        bar_width = 0.2
        index_shap_gt = torch.arange(feat_num) * 5 * bar_width
        index_eff_shap = index_shap_gt + bar_width
        # shap.force_plot(float(expected_value), attr[0, :].detach().numpy(), z[0, :].numpy(), matplotlib=True)
        plt.bar(index_shap_gt, shapley_value_gt[0], width=bar_width, color="b", label="GT-Shapley value")
        plt.bar(index_eff_shap, attr[0], width=bar_width, color="r", label="SHAX")

        aa = 0.1
        bb = 0.03
        for a, b in zip(index_shap_gt, shapley_value_gt[0]):
            if b > 0:
                plt.text(a - aa, b + bb, '%.4f' % b, ha='center', va='bottom', fontsize=10, rotation=90)
            else:
                plt.text(a - aa, b - bb - 0.5, '%.4f' % b, ha='center', va='bottom', fontsize=10, rotation=90)

        for a, b in zip(index_eff_shap, attr[0]):
            if b > 0:
                plt.text(a + aa, b + bb, '%.4f' % b, ha='center', va='bottom', fontsize=10, rotation=90)
            else:
                plt.text(a + aa, b - bb - 0.5, '%.4f' % b, ha='center', va='bottom', fontsize=10, rotation=90)

        plt.xticks(index_shap_gt + bar_width / 2, x_label, rotation=270, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim([-1, 1.7])
        plt.ylabel("Feature Contribution", fontsize=12)
        plt.legend(loc='lower right', fontsize=12)

        plt.subplots_adjust(left=0.115 , bottom=0.36 , top=0.97, right=0.99, wspace=0.01)
        plt.savefig("../figure/Interpretation_adult.pdf")
        plt.savefig("../figure/Interpretation_adult.png")

        # plt.show()
        plt.close()

        ranking = attr.argsort(dim=1)
        shapley_value_buf.append(attr)
        shapley_rank_buf.append(ranking)

        rank_mAP = ((ranking == ranking_gt).sum(dim=1).type(torch.float)/ranking_gt.shape[-1]).mean(dim=0)
        mAP_buf.append(rank_mAP)

        MSE_sum = torch.square(attr - shapley_gt).sum(dim=1).mean(dim=0)  # #
        MSE_buf.append(MSE_sum)
        MMSE = torch.tensor(MSE_buf).mean(dim=0)

        print("Index: {}, mAP: {}, MMSE {}".format(index, rank_mAP, MMSE))
        # print("Index: {}, Rank Precision: {}, mAP: {}, mAP std: {}".format(index, rank_mAP, mAP, mAP_std))


        # stop
        # if index == 100:
        break

    return shapley_value_buf, shapley_rank_buf, total_time



parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, help="number of samples", default=16) # for good shapley value ranking
parser.add_argument("--circ_num", type=int, help="number of circle for average", default=1) # for approaching y
parser.add_argument('--antithetical', action='store_true', help='antithetical sampling.')
parser.add_argument("--softmax", action='store_true', help="softmax model output.")
parser.add_argument("--save", action='store_true', help="save estimated shapley value.")
args = parser.parse_args()


if __name__ == "__main__":

    # datasets_torch, cate_attrib_book, dense_feat_index, sparse_feat_index = load_data('./adult-dataset/adult.csv', val_size=0.2, test_size=0.2, run_num=0) #
    # x_train, y_train, z_train, x_val, y_val, z_val, x_test, y_test, z_test = datasets_torch
    # print(x_train.shape)

    # print(cate_attrib_book)

    if args.softmax:
        checkpoint = torch.load("../adult-dataset/model_softmax_adult_m_1_r_0.pth.tar")
    else:
        checkpoint = torch.load("../adult-dataset/model_adult_m_1_r_0.pth.tar")

    # checkpoint = torch.load("./adult-dataset/model_adult_m_1_l_5_r_0.pth.tar")
    # print(checkpoint)
    dense_feat_index  = checkpoint["dense_feat_index"]
    sparse_feat_index = checkpoint["sparse_feat_index"]
    cate_attrib_book  = checkpoint["cate_attrib_book"]

    model = mlp(input_dim=checkpoint["input_dim"],
                output_dim=checkpoint["output_dim"],
                layer_num=checkpoint["layer_num"],
                hidden_dim=checkpoint["hidden_dim"],
                activation=checkpoint["activation"])

    model.load_state_dict(checkpoint["state_dict"])

    model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

    x_test = checkpoint["test_data_x"]
    y_test = checkpoint["test_data_y"]
    z_test = checkpoint["test_data_z"]
    shapley_value_gt = checkpoint["test_shapley_value"]
    shapley_ranking_gt = checkpoint["test_shapley_ranking"]
    reference = checkpoint["reference"]
    error_matrix = checkpoint["error_matrix"]

    N = error_matrix.shape[0]
    data_loader = DataLoader(TensorDataset(x_test[0:N], y_test[0:N], z_test[0:N], shapley_value_gt[0:N], shapley_ranking_gt[0:N], error_matrix[0:N]),
                             batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    # # data_loader = DataLoader(TensorDataset(x_test, y_test, z_test, shapley_value_gt, shapley_ranking_gt), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    # reference_dense = x_test[:, dense_feat_index].mean(dim=0)
    # reference_sparse = -torch.ones_like(sparse_feat_index).type(torch.long) # -1 for background noise
    # # reference_sparse = torch.zeros_like(sparse_feat_index).type(torch.long) # 0 for background noise
    # reference = torch.cat((reference_dense, reference_sparse), dim=0)

    print(x_test[:, 0:5])
    print(z_test[:, 0:5])

    shapley_value, shapley_rank, total_time = efficient_shap(model_for_shap.forward_wo_sigmoid, data_loader, reference,
                                                 dense_feat_index, sparse_feat_index, cate_attrib_book)

