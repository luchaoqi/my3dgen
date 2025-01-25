# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import copy
import torch.nn.functional as F
import torch.nn as nn
import math

# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
original_conv_lora = False
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity(p=lora_dropout)
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

@persistence.persistent_class
class FullyConnectedLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, fully_connected_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.fully_connected_layer = fully_connected_layer
        in_features = self.fully_connected_layer.in_features
        out_features = self.fully_connected_layer.out_features
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        if r > 0:
            self.lora_A = nn.Parameter(self.fully_connected_layer.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.fully_connected_layer.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.fully_connected_layer.weight.requires_grad = False
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        self.requires_grad_(True)
        self.fully_connected_layer.weight.requires_grad_(False)
        w = self.fully_connected_layer.weight.to(x.dtype) * self.fully_connected_layer.weight_gain
        b = self.fully_connected_layer.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.fully_connected_layer.bias_gain != 1:
                b = b * self.fully_connected_layer.bias_gain

        if self.fully_connected_layer.activation == 'linear' and b is not None:
            result = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            result = result.matmul(w.t())
            result = bias_act.bias_act(result, b, act=self.fully_connected_layer.activation)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    
@persistence.persistent_class
class AdaFullyConnectedLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, fully_connected_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.fully_connected_layer = fully_connected_layer
        in_features = self.fully_connected_layer.in_features
        out_features = self.fully_connected_layer.out_features
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        if r > 0:
            self.lora_A = nn.Parameter(self.fully_connected_layer.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.fully_connected_layer.weight.new_zeros((out_features, r)))
            self.lora_E = nn.Parameter(self.fully_connected_layer.weight.new_zeros((r,1)))
            self.ranknum = nn.Parameter(self.fully_connected_layer.weight.new_zeros((1)), requires_grad=False)
            self.ranknum.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)  
            # Freezing the pre-trained weight matrix
            self.fully_connected_layer.weight.requires_grad = False
            self.ranknum.requires_grad = False
            torch.nn.init.normal_(self.lora_A, mean=0, std=0.02)
            torch.nn.init.normal_(self.lora_B, mean=0, std=0.02)
            torch.nn.init.zeros_(self.lora_E)

    def forward(self, x):
        self.requires_grad_(True)
        self.fully_connected_layer.weight.requires_grad_(False)
        self.ranknum.requires_grad_(False)
        w = self.fully_connected_layer.weight.to(x.dtype) * self.fully_connected_layer.weight_gain
        b = self.fully_connected_layer.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.fully_connected_layer.bias_gain != 1:
                b = b * self.fully_connected_layer.bias_gain

        if self.fully_connected_layer.activation == 'linear' and b is not None:
            result = torch.addmm(b.unsqueeze(0), x, w.t())
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
        else:
            result = result.matmul(w.t())
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            result = bias_act.bias_act(result, b, act=self.fully_connected_layer.activation)
        return result

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.kernel_size = kernel_size

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

@persistence.persistent_class
class Conv2dLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, conv2d_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.conv2d_layer = conv2d_layer
        in_channels = self.conv2d_layer.in_channels
        out_channels = self.conv2d_layer.out_channels
        kernel_size = self.conv2d_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)

        if r > 0:
            if original_conv_lora:
                self.lora_A = torch.nn.Parameter(
                    self.conv2d_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.conv2d_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
                )
            else:
                self.lora_A = torch.nn.Parameter(
                    self.conv2d_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.conv2d_layer.weight.new_zeros((out_channels, r*kernel_size))
                )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv2d_layer.weight.requires_grad_(False)
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, x, gain=1):
        self.requires_grad_(True)
        self.conv2d_layer.weight.requires_grad_(False)
        # w = self.conv2d_layer.weight * self.conv2d_layer.weight_gain
        w = (self.conv2d_layer.weight + (self.lora_B @ self.lora_A).view(self.conv2d_layer.weight.shape) * self.scaling) * self.conv2d_layer.weight_gain
        # w = (self.conv2d_layer.weight + (self.lora_B @ self.lora_A).view(self.conv2d_layer.out_channels, self.conv2d_layer.kernel_size, self.conv2d_layer.in_channels, self.conv2d_layer.kernel_size).permute(0, 2, 1, 3) * self.scaling) * self.conv2d_layer.weight_gain
        b = self.conv2d_layer.bias.to(x.dtype) if self.conv2d_layer.bias is not None else None
        flip_weight = (self.conv2d_layer.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.conv2d_layer.resample_filter, up=self.conv2d_layer.up, down=self.conv2d_layer.down, padding=self.conv2d_layer.padding, flip_weight=flip_weight)

        act_gain = self.conv2d_layer.act_gain * gain
        act_clamp = self.conv2d_layer.conv_clamp * gain if self.conv2d_layer.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.conv2d_layer.activation, gain=act_gain, clamp=act_clamp)
        return x
@persistence.persistent_class
class AdaConv2dLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, conv2d_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.conv2d_layer = conv2d_layer
        in_channels = self.conv2d_layer.in_channels
        out_channels = self.conv2d_layer.out_channels
        kernel_size = self.conv2d_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)

        if r > 0:
            self.lora_A = torch.nn.Parameter(
                self.conv2d_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_E = torch.nn.Parameter(
                self.conv2d_layer.weight.new_zeros((r*kernel_size, 1))
            )
            self.lora_B = torch.nn.Parameter(
                self.conv2d_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r

            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.ranknum.requires_grad = False

            # Freezing the pre-trained weight matrix
            self.conv2d_layer.weight.requires_grad_(False)
            torch.nn.init.normal_(self.lora_A, mean=0, std=0.02)
            torch.nn.init.normal_(self.lora_B, mean=0, std=0.02)
            torch.nn.init.zeros_(self.lora_E)
    def forward(self, x, gain=1):
        self.requires_grad_(True)
        self.ranknum.requires_grad_(False)
        self.conv2d_layer.weight.requires_grad_(False)
        # w = self.conv2d_layer.weight * self.conv2d_layer.weight_gain
        w = (self.conv2d_layer.weight + (self.lora_B @ (self.lora_A * self.lora_E)).view(self.conv2d_layer.weight.shape) * self.scaling / (self.ranknum+1e-5)) * self.conv2d_layer.weight_gain
        b = self.conv2d_layer.bias.to(x.dtype) if self.conv2d_layer.bias is not None else None
        flip_weight = (self.conv2d_layer.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.conv2d_layer.resample_filter, up=self.conv2d_layer.up, down=self.conv2d_layer.down, padding=self.conv2d_layer.padding, flip_weight=flip_weight)

        act_gain = self.conv2d_layer.act_gain * gain
        act_clamp = self.conv2d_layer.conv_clamp * gain if self.conv2d_layer.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.conv2d_layer.activation, gain=act_gain, clamp=act_clamp)
        return x
#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.kernel_size = kernel_size

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])
            

@persistence.persistent_class
class SynthesisLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, synthesis_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.synthesis_layer = synthesis_layer
        in_channels = self.synthesis_layer.in_channels
        out_channels = self.synthesis_layer.out_channels
        kernel_size = self.synthesis_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        self.synthesis_layer.affine = FullyConnectedLayerWrapper(self.synthesis_layer.affine, r, lora_alpha, lora_dropout)
        if r > 0:
            if original_conv_lora:
                self.lora_A = torch.nn.Parameter(
                    self.synthesis_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.synthesis_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
                )
            else:
                self.lora_A = torch.nn.Parameter(
                    self.synthesis_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.synthesis_layer.weight.new_zeros((out_channels, r*kernel_size))
                )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.synthesis_layer.weight.requires_grad_(False)
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        self.requires_grad_(True)
        self.synthesis_layer.weight.requires_grad_(False)
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.synthesis_layer.resolution // self.synthesis_layer.up
        misc.assert_shape(x, [None, self.synthesis_layer.in_channels, in_resolution, in_resolution])
        styles = self.synthesis_layer.affine(w)

        noise = None
        if self.synthesis_layer.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.synthesis_layer.resolution, self.synthesis_layer.resolution], device=x.device) * self.synthesis_layer.noise_strength
        if self.synthesis_layer.use_noise and noise_mode == 'const':
            noise = self.synthesis_layer.noise_const * self.synthesis_layer.noise_strength

        flip_weight = (self.synthesis_layer.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=(self.synthesis_layer.weight + (self.lora_B @ self.lora_A).view(self.synthesis_layer.weight.shape) * self.scaling), styles=styles, noise=noise, up=self.synthesis_layer.up, padding=self.synthesis_layer.padding, resample_filter=self.synthesis_layer.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.synthesis_layer.act_gain * gain
        act_clamp = self.synthesis_layer.conv_clamp * gain if self.synthesis_layer.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.synthesis_layer.bias.to(x.dtype), act=self.synthesis_layer.activation, gain=act_gain, clamp=act_clamp)
        return x


@persistence.persistent_class
class AdaSynthesisLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, synthesis_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.synthesis_layer = synthesis_layer
        in_channels = self.synthesis_layer.in_channels
        out_channels = self.synthesis_layer.out_channels
        kernel_size = self.synthesis_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        self.synthesis_layer.affine = AdaFullyConnectedLayerWrapper(self.synthesis_layer.affine, r, lora_alpha, lora_dropout)
        if r > 0:
            self.lora_A = torch.nn.Parameter(
                self.synthesis_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = torch.nn.Parameter(
                self.synthesis_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.lora_E = torch.nn.Parameter(
                self.synthesis_layer.weight.new_zeros((r*kernel_size,1))
            )
            self.ranknum = nn.Parameter(self.synthesis_layer.weight.new_zeros((1)), requires_grad=False)
            self.ranknum.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)  
            # Freezing the pre-trained weight matrix
            self.synthesis_layer.weight.requires_grad = False
            self.ranknum.requires_grad = False 
            # Freezing the pre-trained weight matrix
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        self.requires_grad_(True)
        self.ranknum.requires_grad_(False)
        self.synthesis_layer.weight.requires_grad_(False)
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.synthesis_layer.resolution // self.synthesis_layer.up
        misc.assert_shape(x, [None, self.synthesis_layer.in_channels, in_resolution, in_resolution])
        styles = self.synthesis_layer.affine(w)

        noise = None
        if self.synthesis_layer.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.synthesis_layer.resolution, self.synthesis_layer.resolution], device=x.device) * self.synthesis_layer.noise_strength
        if self.synthesis_layer.use_noise and noise_mode == 'const':
            noise = self.synthesis_layer.noise_const * self.synthesis_layer.noise_strength

        flip_weight = (self.synthesis_layer.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=(self.synthesis_layer.weight + (self.lora_B @ (self.lora_A * self.lora_E)).view(self.synthesis_layer.weight.shape) * self.scaling / (self.ranknum+1e-5)), styles=styles, noise=noise, up=self.synthesis_layer.up, padding=self.synthesis_layer.padding, resample_filter=self.synthesis_layer.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.synthesis_layer.act_gain * gain
        act_clamp = self.synthesis_layer.conv_clamp * gain if self.synthesis_layer.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.synthesis_layer.bias.to(x.dtype), act=self.synthesis_layer.activation, gain=act_gain, clamp=act_clamp)
        return x
#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.kernel_size = kernel_size

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

@persistence.persistent_class
class ToRGBLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, to_rgb_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.to_rgb_layer = to_rgb_layer
        in_channels = self.to_rgb_layer.in_channels
        out_channels = self.to_rgb_layer.out_channels
        kernel_size = self.to_rgb_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        self.to_rgb_layer.affine = FullyConnectedLayerWrapper(self.to_rgb_layer.affine, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if r > 0:
            if original_conv_lora:
                self.lora_A = torch.nn.Parameter(
                    self.to_rgb_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.to_rgb_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
                )
            else:
                self.lora_A = torch.nn.Parameter(
                    self.to_rgb_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size*kernel_size))
                )
                self.lora_B = torch.nn.Parameter(
                    self.to_rgb_layer.weight.new_zeros((out_channels, r*kernel_size))
                )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.to_rgb_layer.weight.requires_grad_(False)
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)
    def forward(self, x, w, fused_modconv=True):
        self.requires_grad_(True)
        self.to_rgb_layer.weight.requires_grad_(False)
        styles = self.to_rgb_layer.affine(w) * self.to_rgb_layer.weight_gain
        x = modulated_conv2d(x=x, weight=(self.to_rgb_layer.weight + (self.lora_B @ self.lora_A).view(self.to_rgb_layer.weight.shape) * self.scaling), styles=styles, demodulate=False, fused_modconv=fused_modconv)
        # x = modulated_conv2d(x=x, weight=(self.to_rgb_layer.weight + (self.lora_B @ self.lora_A).view(self.to_rgb_layer.out_channels, self.to_rgb_layer.kernel_size, self.to_rgb_layer.in_channels, self.to_rgb_layer.kernel_size).permute(0, 2, 1, 3) * self.scaling), styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.to_rgb_layer.bias.to(x.dtype), clamp=self.to_rgb_layer.conv_clamp)
        return x

@persistence.persistent_class
class AdaToRGBLayerWrapper(torch.nn.Module, LoRALayer):
    def __init__(self, to_rgb_layer, r, lora_alpha=1, lora_dropout=0):
        torch.nn.Module.__init__(self)
        self.to_rgb_layer = to_rgb_layer
        in_channels = self.to_rgb_layer.in_channels
        out_channels = self.to_rgb_layer.out_channels
        kernel_size = self.to_rgb_layer.kernel_size
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=False)
        self.to_rgb_layer.affine = AdaFullyConnectedLayerWrapper(self.to_rgb_layer.affine, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        if r > 0:
            self.lora_A = torch.nn.Parameter(
                self.to_rgb_layer.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = torch.nn.Parameter(
                self.to_rgb_layer.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.lora_E = torch.nn.Parameter(
                self.to_rgb_layer.weight.new_zeros((r*kernel_size,1))
            )
            self.ranknum = nn.Parameter(self.to_rgb_layer.weight.new_zeros((1)), requires_grad=False)
            self.ranknum.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)  
            # Freezing the pre-trained weight matrix
            self.to_rgb_layer.weight.requires_grad = False
            self.ranknum.requires_grad = False 
            # Freezing the pre-trained weight matrix
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def forward(self, x, w, fused_modconv=True):
        self.requires_grad_(True)
        self.ranknum.requires_grad_(False)
        self.to_rgb_layer.weight.requires_grad_(False)
        styles = self.to_rgb_layer.affine(w) * self.to_rgb_layer.weight_gain
        x = modulated_conv2d(x=x, weight=(self.to_rgb_layer.weight + (self.lora_B @ (self.lora_A * self.lora_E)).view(self.to_rgb_layer.weight.shape) * self.scaling / (self.ranknum+1e-5) ), styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.to_rgb_layer.bias.to(x.dtype), clamp=self.to_rgb_layer.conv_clamp)
        return x
#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, zero_conv=False, featurenet=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        
        # # ToRGB.
        # if img is not None:
        #     misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
        #     img = upfirdn2d.upsample2d(img, self.resample_filter)
        # if self.is_last or self.architecture == 'skip':
        #     y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
        #     y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        #     img = img.add_(y) if img is not None else y

        # ToRGB.
        if (img is not None) and (not zero_conv):
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if featurenet is not False:
            if featurenet is not None:
                featurenet_up = upfirdn2d.upsample2d(featurenet, self.resample_filter)
                featurenet_up = self.feature_net(torch.cat([featurenet_up, x], dim=1))
            else:
                featurenet_up = x
            
            if self.is_last or self.architecture == 'skip':
                y = self.torgb(featurenet_up, next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
            assert x.dtype == dtype
            assert img is None or img.dtype == torch.float32
            return x, img, featurenet_up
           

        else:
            if self.is_last or self.architecture == 'skip':
                y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                if zero_conv:
                    y = self.zero_conv1(y)
                img = img.add_(y) if img is not None else y

            assert x.dtype == dtype
            assert img is None or img.dtype == torch.float32
            return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------
# @persistence.persistent_class
# class ConstWrapper(torch.nn.Module, LoRALayer):
#     def __init__(self, const, r, lora_alpha=1, lora_dropout=0):
#         torch.nn.Module.__init__(self)
#         self.const = const
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=False)
#         if r > 0:
#             self.lora_A = nn.Parameter(self.fully_connected_layer.weight.new_zeros((r, in_features)))
#             self.lora_B = nn.Parameter(self.fully_connected_layer.weight.new_zeros((out_features, r)))
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.const.requires_grad_(False)
#             torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             torch.nn.init.zeros_(self.lora_B)

#     def forward(self, x):
#         self.requires_grad_(True)
#         self.const.requires_grad_(False)
#         return self.const + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

@persistence.persistent_class
class SynthesisBlockWrapper(torch.nn.Module):
    def __init__(self, synthesis_block,  r, lora_alpha=1, lora_dropout=0):
        super().__init__()
        self.synthesis_block = synthesis_block
        self.synthesis_block.requires_grad_(True)
        if hasattr(self.synthesis_block, "conv0"):
            self.synthesis_block.conv0 = SynthesisLayerWrapper(self.synthesis_block.conv0,  r, lora_alpha, lora_dropout)
        self.synthesis_block.conv1 = SynthesisLayerWrapper(self.synthesis_block.conv1,  r, lora_alpha, lora_dropout)
        if hasattr(self.synthesis_block, "torgb"):
            self.synthesis_block.torgb = ToRGBLayerWrapper(self.synthesis_block.torgb, r, lora_alpha, lora_dropout)
        if hasattr(self.synthesis_block, "skip"):
            self.synthesis_block.skip = Conv2dLayerWrapper(self.synthesis_block.skip, r, lora_alpha, lora_dropout)
        self.num_conv = self.synthesis_block.num_conv
        self.num_torgb = self.synthesis_block.num_torgb
    def forward(self, *args, **kwargs):
        return self.synthesis_block(*args, **kwargs)

@persistence.persistent_class
class AdaSynthesisBlockWrapper(torch.nn.Module):
    def __init__(self, synthesis_block,  r, lora_alpha=1, lora_dropout=0):
        super().__init__()
        self.synthesis_block = synthesis_block
        self.synthesis_block.requires_grad_(True)
        if hasattr(self.synthesis_block, "conv0"):
            self.synthesis_block.conv0 = AdaSynthesisLayerWrapper(self.synthesis_block.conv0,  r, lora_alpha, lora_dropout)
        self.synthesis_block.conv1 = AdaSynthesisLayerWrapper(self.synthesis_block.conv1,  r, lora_alpha, lora_dropout)
        if hasattr(self.synthesis_block, "torgb"):
            self.synthesis_block.torgb = AdaToRGBLayerWrapper(self.synthesis_block.torgb, r, lora_alpha, lora_dropout)
        if hasattr(self.synthesis_block, "skip"):
            self.synthesis_block.skip = AdaConv2dLayerWrapper(self.synthesis_block.skip, r, lora_alpha, lora_dropout)
        self.num_conv = self.synthesis_block.num_conv
        self.num_torgb = self.synthesis_block.num_torgb
    def forward(self, *args, **kwargs):
        return self.synthesis_block(*args, **kwargs)

# ---------------------------------------------------------------------------- #
#                                custom wrapper                                #
    
@persistence.persistent_class
class ControlNet(torch.nn.Module):
    # controlnet
    def __init__(self, synthesis):
        super().__init__()
        self.synthesis = synthesis # locked 
        self.synthesis_unlocked = copy.deepcopy(synthesis) # unlocked
        # controlnet: add zero_conv for each resolution block
        for res in self.synthesis_unlocked.block_resolutions:
            block = getattr(self.synthesis_unlocked, f'b{res}')
            # input: same as in_channels unless it is the first block
            in_channels = out_channels = block.in_channels if block.in_channels != 0 else block.const.shape[0] # 512 in this case
            block.zero_conv0 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).cuda()
            torch.nn.init.zeros_(block.zero_conv0.weight)
            torch.nn.init.zeros_(block.zero_conv0.bias)
            # output: channel is always 96 for input/output feature maps
            in_channels = out_channels = self.synthesis_unlocked.img_channels
            block.zero_conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).cuda()
            torch.nn.init.zeros_(block.zero_conv1.weight)
            torch.nn.init.zeros_(block.zero_conv1.bias)
    def forward(self, ws, **block_kwargs):
        block_ws = []
        # freeze the synthesis network
        self.synthesis.requires_grad_(False)
        with torch.autograd.profiler.record_function('controlnet'):
            misc.assert_shape(ws, [None, self.synthesis.num_ws, self.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.synthesis.block_resolutions:
                block = getattr(self.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = x_unlocked = img = None
        for res, cur_ws in zip(self.synthesis.block_resolutions, block_ws):
            block = getattr(self.synthesis, f'b{res}')
            block_unlocked = getattr(self.synthesis_unlocked, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            x_unlocked, img = block_unlocked(x_unlocked, img, cur_ws, zero_conv=True, **block_kwargs)
        return img

@persistence.persistent_class
class FeatureNet(torch.nn.Module):
    # extract all intermediate synthesis block features and train a simple MLP to predict the offset
    def __init__(self, synthesis):
        super().__init__()
        self.synthesis = synthesis # locked
        self.synthesis.requires_grad_(False)
        for idx, res in enumerate(self.synthesis.block_resolutions):
            block = getattr(self.synthesis, f'b{res}')
            block.torgb.requires_grad_(True)
            in_channels, out_channels = block.in_channels, block.out_channels
            # block.feature_net = torch.nn.Sequential(
            #     # torch.nn.Conv2d(out_channels+in_channels, out_channels, kernel_size=1),
            #     torch.nn.LeakyReLU(0.2, inplace=True)
            # ).cuda()
            block.feature_net = Conv2dLayer(out_channels+in_channels, out_channels, 1, bias=True, activation='lrelu').cuda()
            block.feature_net.requires_grad_(True)
    def forward(self, ws, **block_kwargs):
        
        block_ws = []
        # freeze the synthesis network
        self.synthesis.requires_grad_(False)
        # unfreeze feature_conv
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            block.feature_net.requires_grad_(True)
            block.torgb.requires_grad_(True)
        with torch.autograd.profiler.record_function('featurenet'):
            misc.assert_shape(ws, [None, self.synthesis.num_ws, self.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.synthesis.block_resolutions:
                block = getattr(self.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        x_side = None
        for res, cur_ws in zip(self.synthesis.block_resolutions, block_ws):
            block = getattr(self.synthesis, f'b{res}')
            x, img, x_side = block(x, img, cur_ws, featurenet=x_side, **block_kwargs)


        return img

@persistence.persistent_class
class LoraNet(torch.nn.Module):
    def __init__(self, synthesis, r, lora_alpha=1, lora_dropout=0):
        super().__init__()
        self.synthesis = synthesis
        
        self.synthesis.requires_grad_(True)
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            setattr(self.synthesis, f'b{res}', SynthesisBlockWrapper(block, r, lora_alpha, lora_dropout))
    def forward(self, *args, **kwargs):
        return self.synthesis(*args, **kwargs)

@persistence.persistent_class
class AdaLoraNet(torch.nn.Module):
    def __init__(self, synthesis, r, lora_alpha=1, lora_dropout=0):
        super().__init__()
        self.synthesis = synthesis
        
        self.synthesis.requires_grad_(True)
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            setattr(self.synthesis, f'b{res}', AdaSynthesisBlockWrapper(block, r, lora_alpha, lora_dropout))
    def forward(self, *args, **kwargs):
        return self.synthesis(*args, **kwargs)
# ---------------------------------------------------------------------------- #

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------