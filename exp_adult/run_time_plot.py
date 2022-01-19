import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

checkpoint_buf_wo_softmax = {
    "SHEAR":[
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_4_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_16_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_128_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/efficient_shap_adult_m_1_s_256_r_0_c_1.pth.tar",
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
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_64_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_128_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/soft_kernelshap_adult_m_1_s_256_r_0_c_1.pth.tar",
    ],
    "KS-Pair":[
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_8_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_36_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_40_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_48_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_56_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_64_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/kernelshap_pair_adult_m_1_s_128_r_0_c_1.pth.tar",
    ],
    "PS":[
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_128_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_adult_m_1_s_192_r_0_c_1.pth.tar",
    ],
    "APS":[
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_8_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_16_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_32_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_64_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_128_r_0_c_1.pth.tar",
        "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_192_r_0_c_1.pth.tar",
        # "../adult-dataset/wo_softmax/permutation_att_adult_m_1_s_256_r_0_c_1.pth.tar",
    ],
}


algorithm_buf = ["KS-WF", "KS-Pair", "PS", "APS", "SHEAR"] # , "Soft-KS" "KernelShap", "Soft-KS", "KS-Pair", "PS", "APS", "Sage",
marker_buf = ['>', '<', 'v', 's', 'o', 's', 'p', 'h']
color_buf = ["orange", "black", "magenta", "green", "red", "#a87900", "#5d1451"]

# checkpoint = torch.load("./adult-dataset/kernelshap_adult_s_16_r_0.pth.tar")
# checkpoint = torch.load("./adult-dataset/efficient_shap_adult_s_16_r_0.pth.tar")


AE_buf_buf = []
AE_std_buf_buf = []
kl_dis_buf_buf = []
rank_mAP_buf_buf = []
mAP_std_buf_buf = []
n_sample_buf_buf = []
time_buf_buf = []

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

    for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
        checkpoint = torch.load(checkpoint_fname)
        n_sample = checkpoint["sample_num"]
        shapley_value = checkpoint["shapley_value"]
        # shapley_rank = checkpoint["shapley_rank"]
        shapley_rank = shapley_value.argsort(dim=1)
        total_time = checkpoint["total_time"]*1./shapley_value.shape[0]

        # print(total_time)

        N = shapley_value.shape[0]
        shapley_value_gt = checkpoint["shapley_value_gt"][0:N]
        shapley_ranking_gt = checkpoint["shapley_ranking_gt"][0:N]

        # absolute_error = torch.abs(shapley_value - shapley_value_gt).sum(dim=1)
        rank_mAP = ((shapley_rank == shapley_ranking_gt).sum(dim=1).type(torch.float)/shapley_ranking_gt.shape[-1])

        # shapley_value_logstd = torch.log(torch.abs(shapley_value) / torch.abs(shapley_value).sum(dim=1).unsqueeze(dim=1) + 1e-8)
        # shapley_value_gt_std = torch.abs(shapley_value_gt) / torch.abs(shapley_value_gt).sum(dim=1).unsqueeze(dim=1)
        # kl_dis = torch.nn.functional.kl_div(shapley_value_logstd, shapley_value_gt_std).detach()

        # AE_buf[checkpoint_index] = absolute_error.mean(dim=0)
        # AE_std_buf[checkpoint_index] = absolute_error.std(dim=0)
        # kl_dis_buf[checkpoint_index] = kl_dis
        rank_mAP_buf[checkpoint_index] = rank_mAP.mean(dim=0)
        mAP_std_buf[checkpoint_index] = rank_mAP.std(dim=0)
        mAP_max_buf[checkpoint_index] = rank_mAP.min(dim=0)[0]
        n_sample_buf[checkpoint_index] = n_sample
        time_buf[checkpoint_index] = total_time

        # mse = torch.square(shapley_value - shapley_value_gt).sum(dim=1)
        # print(mse)
        # print((mse>1e-3).sum())
        # print(mse.mean(dim=0))

    print(alg)
    print(rank_mAP_buf)
    print(time_buf)
    # print(mAP_std_buf)
    # print(mAP_max_buf)

    AE_buf_buf.append(AE_buf)
    AE_std_buf_buf.append(AE_std_buf)
    kl_dis_buf_buf.append(kl_dis_buf)
    rank_mAP_buf_buf.append(rank_mAP_buf)
    mAP_std_buf_buf.append(mAP_std_buf)
    n_sample_buf_buf.append(n_sample_buf)
    time_buf_buf.append(time_buf)



# for alg_index, alg in enumerate(algorithm_buf):
#     rank_mAP_buf = rank_mAP_buf_buf[alg_index]
#     runtime = time_buf_buf[alg_index]
#     fname_buf = checkpoint_buf_wo_softmax[alg]
#
#     time_buf = np.zeros((len(fname_buf),))
#     for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
#         print(checkpoint_fname)
#         checkpoint = torch.load(checkpoint_fname)
#         shapley_value = checkpoint["shapley_value"]
#         # print(checkpoint)
#         total_time = checkpoint["total_time"]*1./shapley_value.shape[0]
#         time_buf[checkpoint_index] = total_time
#         print(total_time)
#         time_buf[checkpoint_index] = total_time
#
#     rank_mAP_buf = rank_mAP_buf_buf[alg_index]
#     mAP_std_buf = mAP_std_buf_buf[alg_index]
#     # total_time = time_buf_buf[alg_index]
#
#     # print(mAP_std_buf)
#     # print(rank_mAP_buf)
#
#     plt.plot(time_buf, rank_mAP_buf,
#          marker=marker_buf[alg_index],
#          label=algorithm_buf[alg_index],
#          linewidth=1.0, markersize=10,
#          color=color_buf[alg_index])
#
# plt.xlabel("Runing Time per Sample (s)", fontsize=15)
# plt.ylabel("mAP of Ranking", fontsize=15)
# plt.legend(loc='lower right', fontsize=14, frameon=True)
# plt.gca().ticklabel_format(style='sci', scilimits=(0, -4), axis='x')
#
# # plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=15)
#
# plt.yticks(fontsize=15)
# plt.grid()
# plt.subplots_adjust(left=0.15 , bottom=0.12, top=0.99, right=0.99, wspace=0.01)
# plt.savefig("../figure/mAP_vs_time_adult.pdf")
#
# plt.show()



for alg_index, alg in enumerate(algorithm_buf):
    rank_mAP_buf = rank_mAP_buf_buf[alg_index]
    runtime = time_buf_buf[alg_index]
    fname_buf = checkpoint_buf_wo_softmax[alg]

    throughtput_buf = np.zeros((len(fname_buf),))
    for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
        print(checkpoint_fname)
        checkpoint = torch.load(checkpoint_fname)
        shapley_value = checkpoint["shapley_value"]
        # print(checkpoint)
        throughtput = shapley_value.shape[0]*1./checkpoint["total_time"]
        throughtput_buf[checkpoint_index] = throughtput
        print(throughtput)

    rank_mAP_buf = rank_mAP_buf_buf[alg_index]
    mAP_std_buf = mAP_std_buf_buf[alg_index]
    # total_time = time_buf_buf[alg_index]

    # print(mAP_std_buf)
    # print(rank_mAP_buf)

    plt.plot(rank_mAP_buf, throughtput_buf,
         marker=marker_buf[alg_index],
         label=algorithm_buf[alg_index],
         linewidth=1.0, markersize=10,
         color=color_buf[alg_index])

plt.ylabel("Throughput", fontsize=18)
plt.xlabel("ACC of Feature Importance Ranking", fontsize=18)
plt.legend(loc='best', fontsize=18, frameon=True)
# plt.xscale("log")
# plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(int(x)) for x in np.log2(n_sample_buf)], fontsize=15)
#
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
# plt.gca().ticklabel_format(style='sci', scilimits=(-5, 1), axis='x')
# plt.ticklabel_format(style='sci', scilimits=(-1, 0), axis='both')
plt.grid()
plt.subplots_adjust(left=0.14, bottom=0.13, top=0.99, right=0.97, wspace=0.01)
plt.savefig("../figure/Throughput_vs_ACC_adult.pdf")
plt.savefig("../figure/Throughput_vs_ACC_adult.png")

# plt.show()
plt.close()