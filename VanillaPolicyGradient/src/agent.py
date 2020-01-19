import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, output_size,
                 input_size,
                 layer_sizes=[128, 128],
                 dropout_rate=0.5):
        super(MLPAgent, self).__init__()
        self.input_size = input_size
        layer_sizes.insert(0, self.input_size)
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x)


class CNNAgent(nn.Module):
    def __init__(self, action_size,
                 obs_dims,
                 conv_filters=[(3, 64), (3, 32)],
                 pooling_filters=[2, 2],
                 pooling_strides=[2, 2],
                 layer_sizes=[128, 128],
                 dropout_rate=0.5):

        super(CNNAgent, self).__init__()
        self.cnn_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        obs_c = obs_dims[2]
        for (conv_size, filters), pool_size, pooling_stride in zip(conv_filters,
                                                                   pooling_filters,
                                                                   pooling_strides):
            self.cnn_layers.append(nn.Conv2d(obs_c, filters,
                                             conv_size,
                                             padding=(conv_size - 1) // 2))
            self.cnn_layers.append(nn.BatchNorm2d(filters))
            self.cnn_layers.append(nn.ReLU(inplace=True))
            self.cnn_layers.append(nn.MaxPool2d(pool_size, stride=pooling_stride))
            obs_c = filters

        layer_sizes.insert(0, self.linear_input_size(obs_dims, obs_c,
                                                     pooling_filters,
                                                     pooling_strides))

        for i in range(len(layer_sizes) - 1):
            self.linear_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.linear_layers.append(nn.ReLU(inplace=True))
        self.linear_layers.append(nn.Dropout(p=dropout_rate))
        self.linear_layers.append(nn.Linear(layer_sizes[-1], action_size))

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.linear_layers:
            x = layer(x)
        return F.softmax(x, dim=1)

    def linear_input_size(self, obs_dims, obs_c, pooling_filters, pooling_strides):
        obs_w = obs_dims[0]
        obs_h = obs_dims[1]
        for pool, stride in zip(pooling_filters, pooling_strides):
            obs_w = (obs_w - pool) / stride + 1
            obs_h = (obs_h - pool) / stride + 1
        return int(obs_w * obs_h * obs_c)
