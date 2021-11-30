#!/usr/bin/env python3
import torch
import torch.nn as nn

class Neural_blocks():
    def __init__(self):
        pass

    @classmethod
    def conv_module(cls, in_channel, out_channel, kernel_size, pooling, activation, pool_method='max', padding=None, batch_norm=True):
        module_list = []

        # batch norm
        if batch_norm:
            module_list.append(nn.BatchNorm2d(num_features=in_channel))

        # conv module
        if padding is not None:
            module_list.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding))
        else:
            module_list.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=kernel_size // 2))

        # if pooling then add pooling
        if pooling > 0:
            if pool_method == 'max':
                module_list.append(nn.MaxPool2d(kernel_size=pooling))
            else:
                module_list.append(nn.AvgPool2d(kernel_size=pooling))

        # add in the activation function
        module_list.append(cls.get_activation_function(activation))

        return nn.Sequential(*module_list)


    @classmethod
    def linear_module(cls, in_features, out_features, activation, batch_norm=True, bias=True):
        module_list = []

        # batch norm
        if batch_norm:
            module_list.append(nn.BatchNorm1d(num_features=in_features))

        # linear module
        module_list.append(nn.Linear(in_features, out_features, bias))

        # add in the activation function
        module_list.append(cls.get_activation_function(activation))

        return nn.Sequential(*module_list)


    @classmethod
    def deconv_module(cls, in_channel, out_channel, kernel_size, padding, stride, activation):
        module_list = []

        # batch norm
        module_list.append(nn.BatchNorm2d(num_features=in_channel))

        # conv module
        module_list.append(nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding, stride=stride))

        # add in the activation function
        module_list.append(cls.get_activation_function(activation))

        return nn.Sequential(*module_list)


    @classmethod
    def cond_gated_masked_conv_module(cls, channel_size, kernel_size, mask_type):
        module = CondGatedMaskedConv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, mask_type=mask_type)

        return module


    @classmethod
    def get_activation_function(cls, activation):
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'lrelu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'identity':
            return nn.Identity()
        else:
            raise Exception(f'Unknown activation function "{activation}"')


    @classmethod
    def generate_conv_stack(cls, channels_description, kernels_description, pooling_description, activation_description, padding=None, batch_norm=True):
        """
        conv -> conv -> conv -> ...
        """
        network_depth = len(channels_description) - 1

        assert(
            network_depth == len(kernels_description) and
            network_depth == len(activation_description) and
            network_depth == len(pooling_description)
            ), 'All network descriptions must be of the same size'

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.conv_module
                (
                    channels_description[i],
                    channels_description[i+1],
                    kernels_description[i],
                    pooling_description[i],
                    activation_description[i],
                    padding=padding,
                    batch_norm=batch_norm
                )
            )

        return nn.Sequential(*module_list)


    @classmethod
    def generate_deconv_stack(cls, channels_description, kernels_description, padding_description, stride_description, activation_description):
        """
        conv -> conv -> conv -> ...
        """
        network_depth = len(channels_description) - 1

        assert(
            network_depth == len(kernels_description) and
            network_depth == len(activation_description) and
            network_depth == len(padding_description) and
            network_depth == len(stride_description)
            ), 'All network descriptions must be of the same size'

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.deconv_module
                (
                    channels_description[i],
                    channels_description[i+1],
                    kernels_description[i],
                    padding_description[i],
                    stride_description[i],
                    activation_description[i]
                )
            )

        return nn.Sequential(*module_list)


    @classmethod
    def generate_cond_gated_masked_conv_stack(cls, channels_description, masks_description, kernels_description):
        """
         |       |       |
         v       v       v
        conv -> conv -> conv -> ...
        """

        network_depth = len(masks_description)

        module_list = nn.ModuleList()
        for i in range(network_depth):
            module_list.append(
                cls.cond_gated_masked_conv_module
                (
                    channels_description,
                    kernels_description,
                    masks_description[i]
                )
            )

        return module_list


    @classmethod
    def generate_conv_res_stack(cls, layers_per_block, channels_description, kernels_description, pooling_description, activation_description):

        blocks = []
        intermediates = []

        for i in range(len(channels_description) - 1):
            _kernels = []
            _activation = []

            # generate the intermediate layers
            _channels = [channels_description[i]] + [channels_description[i+1]]
            _kernels = [kernels_description]
            _pooling = [pooling_description]
            _activation = [activation_description]
            intermediate = cls.generate_conv_stack(_channels, _kernels, _pooling, _activation)

            # these operations are on python lists so + means concat and * means repeat
            _channels = [channels_description[i+1]] * (layers_per_block + 1)
            _kernels = [kernels_description] * layers_per_block
            _pooling = [0] * layers_per_block
            _activation = [activation_description] * layers_per_block

            # generate the conv layers
            block = cls.generate_conv_stack(_channels, _kernels, _pooling, _activation)

            blocks = blocks + [block]
            intermediates = intermediates + [intermediate]

        return nn.ModuleList(blocks), nn.ModuleList(intermediates)


    @classmethod
    def generate_conv_cascade(cls, channels_description, kernels_description, pooling_description, activation_description):
        """
          |
          V
        conv ->
          |
          V
        conv ->
          |
          V
        conv ->
          |
          V
        conv ->
        """
        network_depth = len(channels_description) - 1

        assert(
            network_depth == len(kernels_description) and
            network_depth == len(activation_description) and
            network_depth == len(pooling_description)
            ), 'All network descriptions must be of the same size'

        # a place to strore our cascade
        cascade = []

        # the cascade is made up of module, so we use this to store the module
        module_list = []
        for i in range(network_depth):
            module_list.append
            (
                cls.conv_module
                (
                    channels_description[i],
                    channels_description[i+1],
                    kernels_description[i],
                    pooling_description[i],
                    activation_description[i]
                    )
                )

            cascade.append(nn.Sequential(*module_list))
            module_list = []

        return nn.ModuleList(cascade)


    @classmethod
    def generate_conv_parallel(cls, channels_description, kernels_description, pooling_description, activation_description):
        """
        -> conv ->
        -> conv ->
        -> conv ->
        -> conv ->
        """
        network_depth = len(channels_description)

        assert(
            network_depth == len(kernels_description) and
            network_depth == len(activation_description) and
            network_depth == len(pooling_description)
            ), 'All network descriptions must be of the same size'

        # a place to strore our cascade
        cascade = []

        # the cascade is made up of module, so we use this to store the module
        module_list = []
        for i in range(network_depth):
            module_list.append
            (
                cls.conv_module
                (
                    channels_description[i],
                    channels_description[i],
                    kernels_description[i],
                    pooling_description[i],
                    activation_description[i]
                )
            )

            cascade.append(nn.Sequential(*module_list))
            module_list = []

        return nn.ModuleList(cascade)


    @classmethod
    def generate_linear_stack(cls, features_description, activation_description, batch_norm=True, bias=True):
        """
        linear -> linear -> linear -> ...
        """
        network_depth = len(features_description) - 1

        assert(
            network_depth == len(activation_description)
            ), 'All network descriptions must be of the same size'

        module_list = []
        for i in range(network_depth):
            module_list.append(
                cls.linear_module
                (
                    features_description[i],
                    features_description[i+1],
                    activation_description[i],
                    batch_norm=batch_norm,
                    bias=bias
                )
            )

        return nn.Sequential(*module_list)


    @classmethod
    def generate_gated_linear_stack(cls, features_description, batch_norm=True):
        """
        gated_linear -> gated_linear -> gated_linear -> ...
        """
        network_depth = len(features_description) - 1

        module_list = []
        for i in range(network_depth):
            if batch_norm:
                module_list.append(nn.BatchNorm1d(features_description[i]))

            module_list.append(GatedLinear(features_description[i], features_description[i+1]))

        return nn.Sequential(*module_list)


    @classmethod
    def forward_cascade(cls, cascade, input):
        # fast way to compute the output from a cascade
        output = [input]
        for i in range(len(cascade)):
            output.append(cascade[i](output[i]))

        return output


    @classmethod
    def forward_parallel(cls, cascade, input):
        # parallel run the modules
        output = [None] * len(cascade)

        for i in range(len(cascade)):
            output[i] = cascade[i](input[i])

        return output



class MaskedConv2d(nn.Conv2d):
    """
    Implementation by jzbontar/pixelcnn-pytorch

    mask_type: must be 'A' or 'B' (see [1])
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', torch.ones(self.weight.data.shape), persistent=False)

        # so VIM syntax highlighting doesn't freak out
        self.mask = self.mask

        h = self.weight.data.shape[2]
        w = self.weight.data.shape[3]
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)



class CondGatedMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type):
        super().__init__()
        self.masked_conv_1 = MaskedConv2d( \
                                          mask_type=mask_type,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2
                                          )
        self.masked_conv_2 = MaskedConv2d( \
                                          mask_type=mask_type,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2
                                          )
        self.cond_conv_1 = nn.Conv2d( \
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2
                                     )
        self.cond_conv_2 = nn.Conv2d( \
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2
                                     )

    def forward(self, x, h):
        """
        x: input
        h: conditional input (should have the same shape as input)
        """
        inp = torch.tanh(self.masked_conv_1(x) + self.cond_conv_1(h))
        gate = torch.sigmoid(self.masked_conv_2(x) + self.cond_conv_2(h))
        return inp * gate



class GatedLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.value = nn.Linear(*args, **kwargs)
        self.gate = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.value(x)

