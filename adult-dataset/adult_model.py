import torch
import torch.nn as nn


class mlp(nn.Module):

    def __init__(self, input_dim, output_dim, layer_num=1, hidden_dim=64, activation=None):
        super(mlp, self).__init__()

        self.mlp = nn.ModuleList()
        self.layer_num = layer_num
        self.activation = eval(activation)
        # self.dropout = nn.Dropout(p=0.5)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_str = activation

        if layer_num == 1:
            layer1 = nn.Linear(input_dim, output_dim)
            # layer1.weight.data.mul_(1e-3)
            # nn.init.constant_(layer1.bias.data, 0.)
            self.mlp.append(layer1)

        else:
            for layer_index in range(layer_num):
                if layer_index == 0:
                    layer1 = nn.Linear(input_dim, hidden_dim)
                elif layer_index == layer_num - 1:
                    layer1 = nn.Linear(hidden_dim, output_dim)
                else:
                    layer1 = nn.Linear(hidden_dim, hidden_dim)

                # print(layer1.weight.shape)
                # layer1.weight.data.mul_(1e-3)
                # nn.init.constant_(layer1.bias.data, 0.)
                self.mlp.append(layer1)

    def forward(self, x):
        for layer_index in range(self.layer_num - 1):

            layer = self.mlp[layer_index]
            # print(layer.weight.shape)
            if self.activation == None:
                x = layer(x)
                # x = self.dropout(x)
            else:
                x = layer(x)
                # x = self.dropout(x)
                x = self.activation(x)

        layer_lst = self.mlp[-1]

        return layer_lst(x)
        # return self.activation(layer_lst(x))

    def forward_softmax(self, x):
        return torch.softmax(self.forward(x), dim=1)

    def forward_1(self, x):
        return self.forward(x)[:, 1].unsqueeze(dim=1)

    def forward_softmax_1(self, x):
        return self.forward_softmax(x)[:, 1].unsqueeze(dim=1)

    def forward_wo_sigmoid(self, x):
        return self.forward_softmax(x)[:, 1].unsqueeze(dim=1) - self.forward_softmax(x)[:, 0].unsqueeze(dim=1)

    @torch.no_grad()
    def predict_proba(self, x):

        return torch.softmax(self.forward(x), dim=1)[:, 1] # .unsqueeze(dim=1)




class Model_for_shap(nn.Module):

    def __init__(self, model, dense_index, sparse_index, sparse_codebook):

        super(Model_for_shap, self).__init__()
        self.model = model
        self.dense_feature_index = dense_index
        self.sparse_feature_index = sparse_index
        self.sparse_codebook = sparse_codebook

        # print(dense_index)
        # print(sparse_index)
        # print(sparse_codebook)

        # for x in sparse_codebook:
        #     print(x.shape)

    def forward(self, x):

        # print(x.shape)

        x_dense_feature = x[:, self.dense_feature_index]
        x_sparse_feature = x[:, self.sparse_feature_index]

        sf_onehot_buf = []
        for sf_index in range(x_sparse_feature.shape[1]):
            feature_i = x_sparse_feature[:, sf_index].type(torch.long)
            sparse_codebookping = self.sparse_codebook[sf_index]

            feature_i[feature_i == -1] = sparse_codebookping.shape[0]-1 # -1 for backgruond noise
            index = feature_i.unsqueeze(dim=1).repeat((1, sparse_codebookping.shape[-1]))

            # print(index)
            feature_onehot = torch.gather(sparse_codebookping, 0, index)
            sf_onehot_buf.append(feature_onehot)
        sf_onehot_buf_torch = torch.cat(sf_onehot_buf, dim=1)
        x_ = torch.cat((x_dense_feature, sf_onehot_buf_torch), dim=1)

        return self.model(x_)

    def forward_softmax(self, x):
        return torch.softmax(self.forward(x), dim=1)

    def forward_1(self, x):
        # print(x_.shape)
        return self.forward(x)[:, 1].unsqueeze(dim=1)

    def forward_softmax_1(self, x):
        return self.forward_softmax(x)[:, 1].unsqueeze(dim=1)

    def forward_wo_sigmoid(self, x):
        y = self.forward(x)
        return y[:, 1].unsqueeze(dim=1) - y[:, 0].unsqueeze(dim=1)


    def forward_1_np(self, x):
        x_tensor = torch.from_numpy(x).type(torch.float)
        return self.forward_1(x_tensor).detach().numpy()

    def forward_softmax_1_np(self, x):
        x_tensor = torch.from_numpy(x).type(torch.float)
        return self.forward_softmax_1(x_tensor).detach().numpy()



class masked_Model_for_shap:

    def __init__(self, model, reference):

        self.model = model
        # self.model.eval()

        self.reference = reference
        # print(self.mask)
        # print(self.reference)

    def update_mask(self, mask, local_x):
        self.mask = mask
        self.local_x = local_x

    # @torch.no_grad()
    def forward(self, x):

        x_model = torch.zeros(x.shape[0], self.mask.shape[0]).type(x.dtype)
        # print(x_model[:, self.mask].shape)
        x_model[:, self.mask] = x[:, self.mask]
        # print(x.dtype, x_model.dtype, self.reference.dtype)
        x_model[:, ~self.mask] = self.local_x[:, ~self.mask].type(x.dtype)
        # x_model[:, ~self.mask] = self.reference[~self.mask].type(x.dtype)
        # mask2 = torch.randint(low=0, high=2, size=((~self.mask).sum(),))
        # x_model[:, ~self.mask] = mask2 * self.local_x[:, ~self.mask].type(x.dtype) + (1-mask2) * self.reference[~self.mask].type(x.dtype)
        # print(x_model)
        # print(x_model.shape)

        return self.model(x_model)

    def forward_np(self, x):

        x_torch = torch.from_numpy(x)
        return self.forward(x_torch).numpy()