
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

import sys, os
sys.path.append("../")
sys.path.append("../adult-dataset")
from train_adult import load_data, save_checkpoint

from adult_model import mlp, Model_for_shap
from evaluation import pearsonr_corr, pearsonr_evaluate
import seaborn as sns

checkpoint_buf_wo_softmax = {
    "SHEAR":[
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
    "KernelShap":[
        # "../adult-dataset/wo_softmax/kernelshap_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
    "KS-WF":[
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_256_r_0_c_1.pth.tar",
    ],
    "KS-Pair":[
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_16_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_36_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_40_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_48_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_56_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
    "PS":[
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
    "APS":[
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_128_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_256_r_0_c_1.pth.tar",
    ],
    "Unbiased-KS":[
        # "../adult-dataset/wo_softmax/unbiased_kernelshap_pair_adult_m_1_s_128_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/unbiased_kernelshap_pair_adult_m_1_s_256_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/unbiased_kernelshap_pair_adult_m_1_s_512_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/unbiased_kernelshap_pair_adult_m_1_s_1024_r_0_c_1.pth.tar",
    ],
    "Sage":[
        "../adult-dataset/wo_softmax/sage_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/sage_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/sage_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/sage_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/sage_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
}


algorithm_buf = ["KernelShap", "KS-WF", "KS-Pair", "PS", "APS", "SHEAR"] # , "Soft-KS"] # , "Unbiased-KS"
marker_buf = ['^', '>', '<', 'v', 's', 'o', 'p', 'h']
color_buf = ["blue", "orange", "black", "magenta", "green", "red", "#5d1451", "#a87900"] #

checkpoint = torch.load("../adult-dataset/model_adult_m_1_r_0.pth.tar")

model = mlp(input_dim=checkpoint["input_dim"],
                output_dim=checkpoint["output_dim"],
                layer_num=checkpoint["layer_num"],
                hidden_dim=checkpoint["hidden_dim"],
                activation=checkpoint["activation"])

model.load_state_dict(checkpoint["state_dict"])


reference = checkpoint["reference"]
z_test = checkpoint["test_data_z"]
y_test = checkpoint["test_data_y"]
dense_feat_index  = checkpoint["dense_feat_index"]
sparse_feat_index = checkpoint["sparse_feat_index"]
cate_attrib_book  = checkpoint["cate_attrib_book"]
model_for_shap = Model_for_shap(model, dense_feat_index, sparse_feat_index, cate_attrib_book)

sample_num = 32
topK = 2

important_feature_buf = []
feature_inportance_checkpoint = {}
instance_num = z_test.shape[0]
feature_num = z_test.shape[-1]
mask_buf = torch.randint(0, 2, (topK, instance_num * sample_num, feature_num))
y_test_buf = y_test.unsqueeze(dim=1)

monot_buf_buf = []
faithful_buf_buf = []
n_sample_buf_buf = []
faithful_sample_buf_buf = []
monot_sample_buf_buf = []

for alg_index, alg in enumerate(algorithm_buf):

    print(alg)
    fname_buf = checkpoint_buf_wo_softmax[alg]

    monot_buf = np.zeros((len(fname_buf),))
    faithful_buf = np.zeros((len(fname_buf),))
    n_sample_buf = np.zeros((len(fname_buf),))
    faithful_sample_buf = []
    monot_sample_buf = []

    for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
        checkpoint = torch.load(checkpoint_fname)
        shapley_value = checkpoint["shapley_value"]
        shapley_rank = torch.abs(shapley_value).argsort(dim=1, descending=True)
        # shapley_rank = shapley_value.argsort(dim=1)
        n_sample = checkpoint["sample_num"]


        # feature_index_buf = torch.arange(feature_num).repeat((instance_num*sample_num, 1))
        # z_test_buf = z_test.repeat((1, sample_num)).reshape((instance_num*sample_num, feature_num))
        # reference_buf = reference.repeat((instance_num*sample_num, 1))

        mask = torch.zeros_like(shapley_value)
        arange = torch.arange(instance_num)
        baseline_value = model_for_shap.forward_1(reference.unsqueeze(dim=0))
        deltas = torch.zeros_like(shapley_value).type(torch.float)

        for rank_index in range(feature_num):
            mask[arange, shapley_rank[:, rank_index]] = 1
            x_mask = mask * z_test + (1 - mask) * reference.unsqueeze(dim=0)
            cur_value = model_for_shap.forward_1(x_mask)
            deltas[arange, shapley_rank[:, rank_index]] = torch.abs(cur_value - baseline_value).squeeze(dim=1)
            baseline_value = cur_value.clone()

        delta_rank = deltas.argsort(dim=1, descending=True)
        faithful = pearsonr_corr(shapley_value.abs(), deltas)

        sub_set_num = 200
        faithful_sample = torch.zeros(sub_set_num)
        subset_size = deltas.shape[0]//sub_set_num
        index_rand_ranking = torch.arange(deltas.shape[0]) # torch.randperm(deltas.shape[0])
        for idx in range(sub_set_num):

            index_subset = index_rand_ranking[idx*subset_size:(idx+1)*subset_size]
            shapley_value_tmp = shapley_value[index_subset]
            deltas_tmp = deltas[idx*subset_size:(idx+1)*subset_size]
            faithful_tmp = pearsonr_corr(shapley_value_tmp.abs(), deltas_tmp)
            faithful_sample[idx] = faithful_tmp.detach()

        # print(faithful_sample)

        monot_score = torch.zeros((instance_num,))
        for rank_index in range(feature_num-1):

            delta1 = deltas[arange, shapley_rank[:, rank_index]]
            delta2 = deltas[arange, shapley_rank[:, rank_index+1]]

            monot_score += (delta1 >= delta2).byte()

        monot_sample = monot_score/(feature_num-1)
        monot = (monot_score/(feature_num-1)).mean()
        # print(delta_rank[0:10])
        # print(shapley_rank[0:10])
        # print(((shapley_rank == delta_rank).byte().sum(dim=1)*1./feature_num))

        # monot = ((shapley_rank == delta_rank).byte().sum(dim=1)*1./feature_num).mean()

        # print(monot)

        monot_buf[checkpoint_index] = monot
        faithful_buf[checkpoint_index] = faithful
        n_sample_buf[checkpoint_index] = n_sample
        faithful_sample_buf.append(faithful_sample.unsqueeze(dim=0))
        monot_sample_buf.append(monot_sample.unsqueeze(dim=0))

    print(monot_buf)
    print(faithful_buf)

    monot_sample_buf = torch.cat(monot_sample_buf, dim=0)
    faithful_sample_buf = torch.cat(faithful_sample_buf, dim=0)

    monot_buf_buf.append(monot_buf)
    faithful_buf_buf.append(faithful_buf)
    n_sample_buf_buf.append(n_sample_buf)
    monot_sample_buf_buf.append(monot_sample_buf)
    faithful_sample_buf_buf.append(faithful_sample_buf)


# print(faithful_buf_buf)

for alg_index, alg in enumerate(algorithm_buf):
    # MSE_buf = MSE_buf_buf[alg_index]
    faithful_buf = faithful_buf_buf[alg_index]
    n_sample_buf = n_sample_buf_buf[alg_index]
    faithful_sample_buf = faithful_sample_buf_buf[alg_index].T

    # print(mAP_std_buf)
    # print(rank_mAP_buf)

    faithful_buf[faithful_buf < 0.9] = torch.nan

    if alg == "KS-Pair":

        nan_num_buf = torch.isnan(faithful_sample_buf).type(torch.int).sum(dim=1)
        valid_row = (nan_num_buf == 0)
        faithful_sample_buf = faithful_sample_buf[valid_row]
        n_sample_buf = n_sample_buf[1:]
        faithful_sample_buf = faithful_sample_buf[:, 1:]


    # plt.plot(np.log2(n_sample_buf), faithful_buf, marker=marker_buf[alg_index],
    #          label=algorithm_buf[alg_index],
    #          linewidth=1.0, markersize=10,
    #          color=color_buf[alg_index])

    sns.tsplot(time=np.log2(n_sample_buf), data=faithful_sample_buf,
               marker=marker_buf[alg_index],
               condition=algorithm_buf[alg_index],
               linewidth=0.5, markersize=8,
               color=color_buf[alg_index],
               ci=[50],
               )

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("Faithfulness", fontsize=18)
plt.legend(loc='lower left', fontsize=18, frameon=True)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(left=0.19 , bottom=0.13, top=0.99, right=0.99, wspace=0.01)
plt.savefig("../figure/faithful_vs_n_sample_adult.pdf")

# plt.show()
plt.close()


# print(monot_buf_buf)

for alg_index, alg in enumerate(algorithm_buf):
    # MSE_buf = MSE_buf_buf[alg_index]
    monot_buf = monot_buf_buf[alg_index]
    n_sample_buf = n_sample_buf_buf[alg_index]
    monot_sample_buf = monot_sample_buf_buf[alg_index].T.numpy()
    # print(mAP_std_buf)
    # print(rank_mAP_buf)

    # plt.plot(np.log2(n_sample_buf), monot_buf, marker=marker_buf[alg_index],
    #          label=algorithm_buf[alg_index],
    #          linewidth=1.0, markersize=10,
    #          color=color_buf[alg_index])

    sns.tsplot(time=np.log2(n_sample_buf), data=monot_sample_buf,
               marker=marker_buf[alg_index],
               condition=algorithm_buf[alg_index],
               linewidth=0.5, markersize=8,
               color=color_buf[alg_index],
               ci=[50],
               )

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("Monotonicity", fontsize=18)
plt.legend(loc='lower left', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(left=0.18 , bottom=0.13, top=0.99, right=0.99, wspace=0.01)
plt.savefig("../figure/monot_vs_n_sample_adult.pdf")

# plt.show()
plt.close()

# alg_index_buf = torch.arange(len(algorithm_buf))
# for alg_index, alg in enumerate(algorithm_buf):
#     other_alg_index = (alg_index_buf != alg_index)
#     print(other_alg_index)
#     score_alg = ((prediction_gap[:, :, alg_index] >= prediction_gap[:, :, :].max(dim=2)[0]).sum(dim=0)/instance_num).mean()
#     print(score_alg)


# score_eff_shap = ((prediction_gap[:, :, 1] > prediction_gap[:, :, 0]).sum(dim=0)/instance_num).mean()

# print(score_aps)
# print(score_eff_shap)

# print(feature_inportance_checkpoint)

# save_checkpoint("../adult-dataset/wo_softmax/adult_feature_indication_top_" + str(topK) + "_s_" + str(sample_num) + ".pth.tar",
#     feature_inportance_checkpoint=feature_inportance_checkpoint
# )