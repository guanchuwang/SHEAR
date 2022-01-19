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
from shapreg import removal, games, shapley
import argparse
import time

# sys.path.append("../DeepCTR-Torch/deepctr_torch/models")


@torch.no_grad()
def shapreg_shapley(imputer, data_loader): # , shapley_value_gt, shapley_ranking_gt):
    # model.eval()

    shapley_value_buf = []
    shapley_rank_buf = []
    MSE_buf = []
    mAP_buf = []
    total_time = 0

    for index, (x, y, z, sh_gt, rk_gt) in enumerate(data_loader):
        x = x.numpy()
        y = y.numpy()
        z = z.numpy()
        shapley_gt = sh_gt.numpy()
        ranking_gt = rk_gt.numpy()
        feat_num = z.shape[-1]

        attr_buf = []
        for idx in range(args.circ_num):
            t0 = time.time()
            game = games.PredictionGame(imputer, z[0])
            # Estimate Shapley values
            attr = shapley.ShapleyRegression(game, paired_sampling=False, detect_convergence=False,
                                             n_samples=args.sample_num*feat_num, batch_size=args.sample_num,
                                             bar=False)
            total_time += time.time() - t0

            attr_buf.append(attr.values.reshape(1, -1))
            # print(attr.values.reshape(1, -1))
            # print(shapley_gt)

        attr_buf = np.concatenate(attr_buf, axis=0)
        attr = attr_buf.mean(axis=0).reshape(1, -1)

        ranking = attr.argsort(axis=1)
        shapley_value_buf.append(attr)
        shapley_rank_buf.append(ranking)

        # print(attr)
        # print(ranking)
        # print(ranking_gt)

        rank_mAP = ((ranking == ranking_gt).astype(np.float).sum(axis=1)/ranking_gt.shape[-1]).mean(axis=0)
        mAP_buf.append(rank_mAP)

        MSE = np.square(attr - shapley_gt).mean(axis=0)  # .sum(dim=1) #
        MSE_sum = np.square(attr - shapley_gt).sum(axis=1).mean(axis=0)  # #
        MSE_buf.append(MSE_sum)
        MMSE = np.array(MSE_buf).mean(axis=0)
        mAP = np.array(mAP_buf).mean(axis=0)

        # print("Index: {}, MSE: {}, MMSE: {}".format(index, MSE_sum, MMSE))
        print("Index: {}, Rank Precision: {}, mAP: {}".format(index, rank_mAP, mAP))

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

    data_loader = DataLoader(TensorDataset(x_test, y_test, z_test, shapley_value_gt, shapley_ranking_gt), batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
    reference_dense = x_test[:, dense_feat_index].mean(dim=0)
    reference_sparse = -torch.ones_like(sparse_feat_index).type(torch.long) # -1 for background noise
    # reference_sparse = torch.zeros_like(sparse_feat_index).type(torch.long) # 0 for background noise
    reference = torch.cat((reference_dense, reference_sparse), dim=0).unsqueeze(dim=0).numpy()

    # print(x_test[:, 0:5])
    # print(z_test[:, 0:5])

    if args.softmax:
        imputer = removal.MarginalExtension(reference, model_for_shap.forward_softmax_1_np)
        save_checkpoint_name = "../adult-dataset/softmax/soft_kernelshap_adult_m_1_s_" + str(args.sample_num) + "_r_" + str(checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"

    else:
        imputer = removal.MarginalExtension(reference, model_for_shap.forward_1_np)
        save_checkpoint_name = "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_" + str(args.sample_num) + "_r_" + str(checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"

    shapley_value, shapley_rank, total_time = shapreg_shapley(imputer, data_loader)

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
