
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append("../")
sys.path.append("../adult-dataset")
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
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_256_r_0_c_1.pth.tar",
    ],
    "KS-Pair":[
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_32_r_0_c_1.pth.tar",
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


# checkpoint_buf_softmax = {
#     "KernelShap":[
#         "./adult-dataset/softmax/kernelshap_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "Soft-KS":[
#         "./adult-dataset/softmax/soft_kernelshap_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/soft_kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/soft_kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/soft_kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/soft_kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "KS-Pair":[
#         # "./adult-dataset/softmax/kernelshap_pair_adult_m_1_s_8_r_0_c_1.pth.tar",
#         # "./adult-dataset/softmax/kernelshap_pair_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_pair_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_pair_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/kernelshap_pair_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "Efficient-Shap":[
#         "./adult-dataset/softmax/efficient_shap_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/efficient_shap_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/efficient_shap_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/efficient_shap_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/efficient_shap_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "PS":[
#         "./adult-dataset/softmax/permutation_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "APS":[
#         "./adult-dataset/softmax/permutation_att_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_att_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_att_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_att_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/permutation_att_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
#     "Unbiased-KS":[
#         "./adult-dataset/softmax/unbiased_kernelshap_pair_adult_m_1_s_128_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/unbiased_kernelshap_pair_adult_m_1_s_256_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/unbiased_kernelshap_pair_adult_m_1_s_512_r_0_c_1.pth.tar",
#         "./adult-dataseo_softmax/unbiased_kernelshap_pair_adult_m_1_s_1024_r_0_c_1.pth.tar",
#     ],
#     "Sage":[
#         "./adult-dataset/softmax/sage_adult_m_1_s_8_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/sage_adult_m_1_s_16_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/sage_adult_m_1_s_32_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/sage_adult_m_1_s_64_r_0_c_1.pth.tar",
#         "./adult-dataset/softmax/sage_adult_m_1_s_128_r_0_c_1.pth.tar",
#     ],
# }



algorithm_buf = ["KernelShap", "KS-WF", "KS-Pair", "PS", "APS", "SHEAR"] # , "Sage" , "Soft-KS"] # , "Unbiased-KS"
marker_buf = ['^', '>', '<', 'v', 's', 'o', 'p', 'h']
color_buf = ["blue", "orange", "black", "magenta", "green", "red", "#a87900", "#5d1451"]

# checkpoint = torch.load("./adult-dataset/kernelshap_adult_s_16_r_0.pth.tar")
# checkpoint = torch.load("./adult-dataset/efficient_shap_adult_s_16_r_0.pth.tar")


AE_buf_buf = []
AE_std_buf_buf = []
kl_dis_buf_buf = []
rank_mAP_buf_buf = []
mAP_std_buf_buf = []
n_sample_buf_buf = []
time_buf_buf = []
corr_value_buf_buf = []
AE_sample_buf_buf = []
mAP_sample_buf_buf = []

for alg_index, alg in enumerate(algorithm_buf):

    fname_buf = checkpoint_buf_wo_softmax[alg]
    # fname_buf = checkpoint_buf_softmax[alg]
    AE_buf = np.zeros((len(fname_buf),))
    AE_std_buf = np.zeros((len(fname_buf),))
    kl_dis_buf = np.zeros((len(fname_buf),))
    rank_mAP_buf = np.zeros((len(fname_buf),))
    mAP_std_buf = np.zeros((len(fname_buf),))
    mAP_max_buf = np.zeros((len(fname_buf),))
    n_sample_buf = np.zeros((len(fname_buf),))
    time_buf = np.zeros((len(fname_buf),))
    corr_value_buf = np.zeros((len(fname_buf),))
    AE_sample_buf = []
    mAP_sample_buf = []

    for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
        checkpoint = torch.load(checkpoint_fname)
        n_sample = checkpoint["sample_num"]
        shapley_value = checkpoint["shapley_value"]
        # shapley_rank = checkpoint["shapley_rank"]
        shapley_rank = shapley_value.argsort(dim=1)
        # total_time = checkpoint["total_time"]*1./shapley_value.shape[0]

        # print(total_time)

        N = shapley_value.shape[0]
        feature_num = shapley_value.shape[-1]
        shapley_value_gt = checkpoint["shapley_value_gt"][0:N]
        shapley_ranking_gt = checkpoint["shapley_ranking_gt"][0:N]

        absolute_error = torch.abs(shapley_value - shapley_value_gt).sum(dim=1)
        corr_value = pearsonr_corr(shapley_value, shapley_value_gt)
        # np.corrcoef(x, y)

        mAP_weight = torch.tensor([[1./(feature_num-x) for x in range(feature_num)]])
        # rank_mAP = ((shapley_rank == shapley_ranking_gt).sum(dim=1).type(torch.float)/shapley_ranking_gt.shape[-1])
        rank_mAP = torch.sum((shapley_rank == shapley_ranking_gt).type(torch.float)*mAP_weight, dim=1)/mAP_weight.sum()

        shapley_value_logstd = torch.log(torch.abs(shapley_value) / torch.abs(shapley_value).sum(dim=1).unsqueeze(dim=1) + 1e-8)
        shapley_value_gt_std = torch.abs(shapley_value_gt) / torch.abs(shapley_value_gt).sum(dim=1).unsqueeze(dim=1)
        kl_dis = torch.nn.functional.kl_div(shapley_value_logstd, shapley_value_gt_std).detach()

        AE_buf[checkpoint_index] = absolute_error.mean(dim=0)
        AE_std_buf[checkpoint_index] = absolute_error.std(dim=0)
        kl_dis_buf[checkpoint_index] = kl_dis
        rank_mAP_buf[checkpoint_index] = rank_mAP.mean(dim=0)
        mAP_std_buf[checkpoint_index] = rank_mAP.std(dim=0)
        mAP_max_buf[checkpoint_index] = rank_mAP.min(dim=0)[0]
        n_sample_buf[checkpoint_index] = n_sample
        corr_value_buf[checkpoint_index] = corr_value
        # time_buf[checkpoint_index] = total_time
        AE_sample_buf.append(absolute_error.unsqueeze(dim=0))
        mAP_sample_buf.append(rank_mAP.unsqueeze(dim=0))

        # if alg == "SHEAR":
        #     np.savetxt("rank_mAP.txt", rank_mAP)
        #     stop

        # mse = torch.square(shapley_value - shapley_value_gt).sum(dim=1)
        # print(mse)
        # print((mse>1e-3).sum())
        # print(mse.mean(dim=0))

    print(alg)
    print(AE_buf)
    # print(AE_std_buf)
    # print(kl_dis_buf)
    print(rank_mAP_buf)
    # print(mAP_std_buf)
    # print(mAP_max_buf)
    # print(corr_value_buf)

    AE_sample_buf = torch.cat(AE_sample_buf, dim=0)
    mAP_sample_buf = torch.cat(mAP_sample_buf, dim=0)
    AE_buf_buf.append(AE_buf)
    AE_std_buf_buf.append(AE_std_buf)
    kl_dis_buf_buf.append(kl_dis_buf)
    rank_mAP_buf_buf.append(rank_mAP_buf)
    mAP_std_buf_buf.append(mAP_std_buf)
    n_sample_buf_buf.append(n_sample_buf)
    time_buf_buf.append(time_buf)
    corr_value_buf_buf.append(corr_value_buf)
    AE_sample_buf_buf.append(AE_sample_buf)
    mAP_sample_buf_buf.append(mAP_sample_buf)

for alg_index, alg in enumerate(algorithm_buf):

    # if alg == "Sage":
    #     continue

    MSE_buf = AE_buf_buf[alg_index]
    MSE_buf[MSE_buf > 1] = np.nan
    AE_sample_buf = AE_sample_buf_buf[alg_index].T.numpy()
    n_sample_buf = n_sample_buf_buf[alg_index]

    if alg == "KS-Pair":
        AE_sample_buf = AE_sample_buf[:, 4:]
        n_sample_buf = n_sample_buf[4:]

    corr_value_buf = corr_value_buf_buf[alg_index]
    corr_value_buf[corr_value_buf < 0.9] = np.nan

    # plt.plot(np.log2(n_sample_buf), MSE_buf, marker=marker_buf[alg_index],
    #          label=algorithm_buf[alg_index],
    #          linewidth=1.0, markersize=10,
    #          color=color_buf[alg_index])

    sns.tsplot(time=np.log2(n_sample_buf), data=AE_sample_buf,
               marker=marker_buf[alg_index],
               condition=algorithm_buf[alg_index],
               linewidth=0.5, markersize=8,
               color=color_buf[alg_index],
               ci=[100]
               )

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("AE of Estimated Shapley Value", fontsize=18)
plt.legend(loc='upper right', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
plt.gca().ticklabel_format(style='sci', scilimits=(0, -3), axis='y')

# plt.xticks(fontsize=15)
plt.yticks(fontsize=18)
plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(left=0.15 , bottom=0.13, top=0.97, right=0.99, wspace=0.01)
plt.savefig("../figure/AE_vs_n_sample_adult.pdf")
plt.savefig("../figure/AE_vs_n_sample_adult.png")

# plt.show()
plt.close()

for alg_index, alg in enumerate(algorithm_buf):
    # MSE_buf = MSE_buf_buf[alg_index]
    rank_mAP_buf = rank_mAP_buf_buf[alg_index]
    mAP_std_buf = mAP_std_buf_buf[alg_index]
    n_sample_buf = n_sample_buf_buf[alg_index]
    mAP_sample_buf = mAP_sample_buf_buf[alg_index].T.numpy()

    # print(mAP_std_buf)
    # print(rank_mAP_buf)

    # plt.plot(np.log2(n_sample_buf), rank_mAP_buf, marker=marker_buf[alg_index],
    #          label=algorithm_buf[alg_index],
    #          linewidth=1.0, markersize=10,
    #          color=color_buf[alg_index])

    sns.tsplot(time=np.log2(n_sample_buf), data=mAP_sample_buf,
               marker=marker_buf[alg_index],
               condition=algorithm_buf[alg_index],
               linewidth=0.5, markersize=8,
               color=color_buf[alg_index],
               ci=[100],
               )


plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("ACC of Feature Importance Ranking", fontsize=18)
plt.legend(loc='lower right', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(left=0.18, bottom=0.13, top=0.99, right=0.99, wspace=0.01)
plt.savefig("../figure/mAP_vs_n_sample_adult.pdf")
plt.savefig("../figure/mAP_vs_n_sample_adult.png")

# plt.show()
plt.close()



# checkpoint = torch.load("./adult-dataset/kernelshap_adult_r_0.pth.tar")
# # checkpoint = torch.load("./adult-dataset/efficient_shap_adult_r_0.pth.tar")
#
# # shapley_value = checkpoint["shapley_value"]
# # shapley_rank = checkpoint["shapley_rank"]
# # shapley_value_gt = checkpoint["shapley_value_gt"]
# # shapley_ranking_gt = checkpoint["shapley_ranking_gt"]
# # MSE = torch.square(shapley_value - shapley_value_gt).sum(dim=1).mean(dim=0)
# # rank_mAP = (shapley_rank == shapley_ranking_gt).sum(dim=1).type(torch.float).mean(dim=0)/shapley_ranking_gt.shape[-1]
# #
# # print(MSE)
# # print(rank_mAP)
#
#
# checkpoint["sample_num"] = 16
#
# save_checkpoint("./adult-dataset/kernelshap_adult_s_" + str(checkpoint["sample_num"]) + "_r_" + str(checkpoint["round_index"]) + ".pth.tar",
#                     round_index=checkpoint["round_index"],
#                     shapley_value_gt=checkpoint["shapley_value_gt"],
#                     shapley_ranking_gt=checkpoint["shapley_ranking_gt"],
#                     shapley_value=checkpoint["shapley_value"],
#                     shapley_rank=checkpoint["shapley_rank"],
#                     sample_num=checkpoint["sample_num"],
#                     )


