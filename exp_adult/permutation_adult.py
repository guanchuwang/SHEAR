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
from shapreg import removal, games, shapley, shapley_unbiased, shapley_sampling, stochastic_games
from shapreg.utils import crossentropyloss

from shap_utils import permutation_sample_parallel
import argparse
import time
# sys.path.append("../DeepCTR-Torch/deepctr_torch/models")

@torch.no_grad()
def shapreg_shapley(model, data_loader, reference): # , shapley_value_gt, shapley_ranking_gt):

    shapley_value_buf = []
    shapley_rank_buf = []
    MSE_buf = []
    mAP_buf = []
    total_time = 0


    for index, (x, y, z, sh_gt, rk_gt) in enumerate(data_loader):
        x = x # .numpy()
        y = y # .numpy()
        z = z # .numpy()
        shapley_gt = sh_gt.numpy()
        ranking_gt = rk_gt.numpy()
        feat_num = z.shape[-1]
        attr_buf = []


        # rank_mAP_buf = []
        for idx in range(args.circ_num):
            t0 = time.time()
            attr = permutation_sample_parallel(model, z, reference, batch_size=args.sample_num, antithetical=args.antithetical)
            # t1 = time.time()
            # print(index, t1-t0)
            total_time += time.time() - t0
            attr_buf.append(attr.reshape(1, -1))

        attr_buf = np.concatenate(attr_buf, axis=0)
        attr = attr_buf.mean(axis=0).reshape(1, -1)

        ranking = attr.argsort(axis=1)
        shapley_value_buf.append(attr)
        shapley_rank_buf.append(ranking)

        rank_mAP = ((ranking == ranking_gt).astype(np.float).sum(axis=1)/ranking_gt.shape[-1]).mean(axis=0)
        mAP_buf.append(rank_mAP)

        mAP = np.array(mAP_buf).mean(axis=0)
        mAP_std = np.array(mAP_buf).std(axis=0)

        # print("Index: {}, MSE: {}, MMSE: {}".format(index, MSE_sum, MMSE))
        print("Index: {}, Rank Precision: {}, mAP: {}, mAP std: {}".format(index, rank_mAP, mAP, mAP_std))
        # stop

        # if index == 100:
        #     break

    shapley_value_buf = np.concatenate(shapley_value_buf, axis=0)
    shapley_rank_buf = np.concatenate(shapley_rank_buf, axis=0)

    shapley_value_buf = torch.from_numpy(shapley_value_buf).type(torch.float)
    shapley_rank_buf = torch.from_numpy(shapley_rank_buf).type(torch.int)

    return shapley_value_buf, shapley_rank_buf, total_time

        # attr_buf = []
        #
        # for index in range(10):
        #     attr = interaware_kernel_shap(model.forward, z, reference, error_matrix)
        #     # print(attr)
        #     # print(attr.argsort(axis=1))
        #     attr_buf.append(attr.reshape(1, -1))
        #
        # attr_buf = torch.cat(attr_buf, dim=0)
        # MSE = torch.square(attr_buf - shapley_gt).mean(dim=0) # .sum(dim=1) #
        # MSE_sum = torch.square(attr_buf - shapley_gt).sum(dim=1).mean(dim=0) #  #
        # print("MSE: {}".format(MSE_sum))


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

    N = shapley_value_gt.shape[0]
    data_loader = DataLoader(TensorDataset(x_test[0:N], y_test[0:N], z_test[0:N], shapley_value_gt, shapley_ranking_gt), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    # data_loader = DataLoader(TensorDataset(x_test, y_test, z_test, shapley_value_gt, shapley_ranking_gt), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    # print(x_test[:, 0:5])
    # print(z_test[:, 0:5])

    if args.softmax:
        shapley_value, shapley_rank, total_time = shapreg_shapley(model_for_shap.forward_softmax_1, data_loader, reference) #, shapley_value_gt, shapley_ranking_gt)
        save_checkpoint_name = "../adult-dataset/softmax/permutation_" + "att_"*args.antithetical + "adult_m_1_s_" + str(args.sample_num) + "_r_" + str(checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"

    else:
        shapley_value, shapley_rank, total_time = shapreg_shapley(model_for_shap.forward_1, data_loader, reference) #, shapley_value_gt, shapley_ranking_gt)
        save_checkpoint_name = "../adult-dataset/wo_softmax/permutation_" + "att_"*args.antithetical + "adult_m_1_s_" + str(args.sample_num) + "_r_" + str(checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"

    if args.save:
        save_checkpoint(save_checkpoint_name,
                        round_index=checkpoint["round_index"],
                        shapley_value_gt=shapley_value_gt,
                        shapley_ranking_gt=shapley_ranking_gt,
                        shapley_value=shapley_value,
                        shapley_rank=shapley_rank,
                        sample_num=args.sample_num,
                        total_time=total_time,
                        )
