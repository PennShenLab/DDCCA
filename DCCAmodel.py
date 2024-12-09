import torch
import torch.nn as nn
from objectives import cca_loss
import math

LEAKYRELU_SLOPE = math.sqrt(5)  # define the negative slope of leakyRelu

class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size, dropout_rate=0.):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                    nn.LeakyReLU(negative_slope=LEAKYRELU_SLOPE),
                ))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        num_layers = len(self.layers)
        l_id = 0
        for layer in self.layers:
            l_id = l_id + 1
            if l_id == num_layers:
                x = layer(x)
            else:
                x = self.dropout(layer(x))

        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values,
                 device=torch.device('cpu'), dropout_rate1=0., dropout_rate2=0.):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1, dropout_rate1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2, dropout_rate2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

        self.w = [None, None]

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

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




