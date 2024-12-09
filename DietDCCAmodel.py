
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from objectives import cca_loss
from DCCAmodel import MlpNet
import math

LEAKYRELU_SLOPE = math.sqrt(5)

class Hyper_net(nn.Module):
    def __init__(self, n_feats, n_hidden_u, param_init, dropout_rate=0.):
        super(Hyper_net, self).__init__()

        self.hidden_1 = nn.Linear(n_feats, n_hidden_u, bias=False)
        if param_init is not None:
            params = np.load(param_init)
            self.hidden_1.weight = torch.nn.Parameter(
                torch.from_numpy(params['w1_aux']))
        else:
            pass
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden_u, affine=False)  # to be consistent with DCCA

        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_u, bias=False)
        if param_init is not None:
            self.hidden_2.weight = torch.nn.Parameter(
                torch.from_numpy(params['w2_aux']))
        else:
            pass
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden_u, affine=False)

        self.hidden_3 = nn.Linear(n_hidden_u, n_hidden_u, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=n_hidden_u, affine=False)

        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x):
        ze1 = self.hidden_1(x)
        ze1 = self.bn1(ze1)

        ae1 = F.leaky_relu(ze1, negative_slope=LEAKYRELU_SLOPE)
        ae1 = self.dropout(ae1)

        ze2 = self.hidden_2(ae1)
        ze2 = self.bn2(ze2)
        ae2 = F.leaky_relu(ze2, negative_slope=LEAKYRELU_SLOPE)
        ae2 = self.dropout(ae2)

        ze3 = self.hidden_3(ae2)

        return ze3


class Target_net(nn.Module):
    """
    modified to take fatLayer_weights as a forward arg.
    Does not have weights for first layer;
    Uses F.linear with passed weights instead
    """

    def __init__(self, input_size, n_feats,
                 n_hidden1_u, n_hidden2_u, n_targets,
                 param_init, input_dropout=0., eps=1e-5, incl_bias=True, incl_softmax=False, dropout_rate=0.):
        super(Target_net, self).__init__()

        if param_init is not None:
            params = np.load(param_init)

        self.bn0 = nn.BatchNorm1d(num_features=input_size, eps=eps)

        self.input_dropout = nn.Dropout(p=input_dropout)

        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps, affine=False)

        self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
        if param_init is not None:
            self.hidden_2.weight = torch.nn.Parameter(
                torch.from_numpy(params['w2_main']))
        else:
            pass
        nn.init.zeros_(self.hidden_2.bias)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u, eps=eps, affine=False)

        self.out = nn.Linear(n_hidden2_u, n_targets)
        if param_init is not None:
            self.out.weight = torch.nn.Parameter(
                torch.from_numpy(params['w3_main']))
        else:
            pass
        nn.init.zeros_(self.out.bias)

        if incl_bias:
            self.fat_bias = nn.Parameter(data=torch.rand(n_hidden1_u), requires_grad=True)
            nn.init.zeros_(self.fat_bias)
        else:
            self.fat_bias = None

        self.dropout = nn.Dropout(p=dropout_rate)

        self.incl_softmax = incl_softmax

    def forward(self, x, fatLayer_weights):
        """
        input size: batch_size x n_feats
        """

        x = self.bn0(x)

        z1 = F.linear(x, fatLayer_weights,
                      bias=self.fat_bias)
        z1 = self.bn1(z1)
        a1 = F.leaky_relu(z1, negative_slope=LEAKYRELU_SLOPE)
        a1 = self.dropout(a1)

        z2 = self.hidden_2(a1)
        z2 = self.bn2(z2)
        a2 = F.leaky_relu(z2, negative_slope=LEAKYRELU_SLOPE)
        a2 = self.dropout(a2)

        out = self.out(a2)
        if self.incl_softmax:
            out = torch.softmax(out, 1)

        return out


class DNModel(nn.Module):
    def __init__(self, input_size, n_feats_emb, n_hidden_u, n_hidden1_u, n_hidden2_u,
                 n_targets, param_init, input_dropout=0., eps=1e-5, incl_bias=True, incl_softmax=False, dropout_rate=0.):
        super(DNModel, self).__init__()

        self.feat_emb = Hyper_net(n_feats_emb, n_hidden_u, param_init, dropout_rate)
        self.disc_net = Target_net(input_size, n_feats_emb, n_hidden1_u, n_hidden2_u,
                                   n_targets, param_init, input_dropout, eps, incl_bias, incl_softmax, dropout_rate)

    def forward(self, emb, x_batch):
        # Forward pass in hyper-net
        feat_emb_model_out = self.feat_emb(emb)
        # Forward pass in target-net
        fatLayer_weights = torch.transpose(feat_emb_model_out, 1, 0)
        discrim_model_out = self.disc_net(x_batch, fatLayer_weights)

        return discrim_model_out


class DDCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, n_feats_emb2, n_targets, outdim_size,
                 use_all_singular_values, device=torch.device('cpu'), dropout_rate1=0., dropout_rate2=0.):
        super(DDCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1, dropout_rate1).double()
        self.model2 = DNModel(input_size2, n_feats_emb2, layer_sizes2['emb_n_hidden_u'],
                              layer_sizes2['discrim_n_hidden1_u'], layer_sizes2['discrim_n_hidden2_u'], n_targets,
                              param_init=None, dropout_rate=dropout_rate2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

        self.w = [None, None]

    def forward(self, x1, x2, emb):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(emb, x2)

        return output1, output2

    def get_w(self):
        """
        getter function for the loading vectors w
        """

        return self.w

    def set_w(self, ww):
        """
        setter function for the loading vectors w
        """

        self.w = ww