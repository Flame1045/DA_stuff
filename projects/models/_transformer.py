# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))

except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv-full`` if you need this module. ')
    
from natten import NeighborhoodAttention2D
from .slide_attention import SlideAttention
    
def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)

def build_Adapter(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(nn.Linear(input_dim, hidden_dim))
    A_layers.append(nn.GELU()) # nn.GELU or nn.ReLU()
    A_layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*A_layers)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class MLP_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.natten = NeighborhoodAttention2D(dim=hidden_dim, kernel_size=5, dilation=1, num_heads=4)
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)

        # xs = xs.permute(1, 0, 2)
        # BT, L, C = xs.size()
        # T = 16
        # B = BT // 16
        # H = W = round(math.sqrt(L - 1))
        # assert L - 1 == H * W
        # xs_cls = xs[:, 0:1, :]
        # xs_patch = xs[:, 1:, :]

        # xs_patch = xs_patch.view(BT, H, W, C)
        # print(xs.shape)
        H, W, D = xs.shape
        xs = torch.reshape(xs,(1,H,W,D))
        xs = self.natten(xs)
        xs = torch.reshape(xs,(H,W,D))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLP_Adapter_a(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.natten = NeighborhoodAttention2D(dim=hidden_dim, kernel_size=kernel_size, dilation=1, num_heads=4)
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)

        # xs = xs.permute(1, 0, 2)
        # BT, L, C = xs.size()
        # T = 16
        # B = BT // 16
        # H = W = round(math.sqrt(L - 1))
        # assert L - 1 == H * W
        # xs_cls = xs[:, 0:1, :]
        # xs_patch = xs[:, 1:, :]

        # xs_patch = xs_patch.view(BT, H, W, C)
        # print(xs.shape)
        H, W, D = xs.shape
        xs = torch.reshape(xs,(1,H,W,D))
        xs = self.natten(xs)
        xs = torch.reshape(xs,(H,W,D))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class MLP_Adapter_slide(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True) 
        self.conv_B = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.silde = SlideAttention(dim=hidden_dim, num_heads=8, ka=3).cuda()
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)
        H, W, C = xs.shape
        xs = torch.reshape(xs,(1,C,H,W))
        xs, _ , _ = self.silde(xs)
        xs = torch.reshape(xs,(H,W,C))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class MLP_Adapter_slide8(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True) 
        self.conv_B = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_C = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_D = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_E = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_F = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_G = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_H = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.silde = SlideAttention(dim=hidden_dim, num_heads=8, ka=3).cuda()
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))
        xs = self.conv_D(self.dropout(self.conv_C(xs)))
        xs = self.conv_F(self.dropout(self.conv_E(xs)))
        xs = self.conv_H(self.dropout(self.conv_G(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)
        H, W, C = xs.shape
        xs = torch.reshape(xs,(1,C,H,W))
        xs, _ , _ = self.silde(xs)
        xs = torch.reshape(xs,(H,W,C))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class MLP_Adapter_slide16(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True) 
        self.conv_B = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_C = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_D = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_E = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_F = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_G = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_H = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_I = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_J = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_K = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_L = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_M = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_N = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_O = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_P = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.silde = SlideAttention(dim=hidden_dim, num_heads=8, ka=3).cuda()
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))
        xs = self.conv_D(self.dropout(self.conv_C(xs)))
        xs = self.conv_F(self.dropout(self.conv_E(xs)))
        xs = self.conv_H(self.dropout(self.conv_G(xs)))
        xs = self.conv_J(self.dropout(self.conv_I(xs)))
        xs = self.conv_L(self.dropout(self.conv_K(xs)))
        xs = self.conv_N(self.dropout(self.conv_M(xs)))
        xs = self.conv_P(self.dropout(self.conv_O(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)
        H, W, C = xs.shape
        xs = torch.reshape(xs,(1,C,H,W))
        xs, _ , _ = self.silde(xs)
        xs = torch.reshape(xs,(H,W,C))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class MLP_Adapter_slide32(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True) 
        self.conv_B = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_C = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_D = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_E = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_F = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_G = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_H = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_I = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_J = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_K = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_L = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_M = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_N = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_O = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_P = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_Q = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_R = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_S = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_T = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_U = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_V = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_W = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_X = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_Y = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_Z = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_A_ = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_B_ = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_C_ = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_D_ = nn.Conv1d(64, 64, 1, groups=1, bias=True)
        self.conv_E_ = nn.Conv1d(64, 64, 1, groups=1, bias=True) 
        self.conv_F_ = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        # self.drop_path = nn.Identity()
        self.silde = SlideAttention(dim=hidden_dim, num_heads=8, ka=3).cuda()
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))
        xs = self.conv_D(self.dropout(self.conv_C(xs)))
        xs = self.conv_F(self.dropout(self.conv_E(xs)))
        xs = self.conv_H(self.dropout(self.conv_G(xs)))
        xs = self.conv_J(self.dropout(self.conv_I(xs)))
        xs = self.conv_L(self.dropout(self.conv_K(xs)))
        xs = self.conv_N(self.dropout(self.conv_M(xs)))
        xs = self.conv_P(self.dropout(self.conv_O(xs)))
        xs = self.conv_R(self.dropout(self.conv_Q(xs)))
        xs = self.conv_T(self.dropout(self.conv_S(xs)))
        xs = self.conv_V(self.dropout(self.conv_U(xs)))
        xs = self.conv_X(self.dropout(self.conv_W(xs)))
        xs = self.conv_Z(self.dropout(self.conv_Y(xs)))
        xs = self.conv_B_(self.dropout(self.conv_A_(xs)))
        xs = self.conv_D_(self.dropout(self.conv_C_(xs)))
        xs = self.conv_F_(self.dropout(self.conv_E_(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)
        H, W, C = xs.shape
        xs = torch.reshape(xs,(1,C,H,W))
        xs, _ , _ = self.silde(xs)
        xs = torch.reshape(xs,(H,W,C))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLP_Adapter_ratio(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, ratio=0.25):
        super().__init__()
        self.skip_connect = skip_connect
        # D_hidden_features = int(input_dim * mlp_ratio)
        # self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        # self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(input_dim, hidden_dim)
        self.D_fc2 = nn.Linear(hidden_dim, output_dim)
        self.conv_A = nn.Conv1d(hidden_dim, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, hidden_dim, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        if int(hidden_dim*ratio) % 2 == 0:
            kernel_size = int(hidden_dim*ratio) + 1
        print("kernel_size:",kernel_size)
        # self.drop_path = nn.Identity()
        self.natten = NeighborhoodAttention2D(dim=hidden_dim, kernel_size=kernel_size, dilation=1, num_heads=4)
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)

        # xs = xs.permute(1, 0, 2)
        # BT, L, C = xs.size()
        # T = 16
        # B = BT // 16
        # H = W = round(math.sqrt(L - 1))
        # assert L - 1 == H * W
        # xs_cls = xs[:, 0:1, :]
        # xs_patch = xs[:, 1:, :]

        # xs_patch = xs_patch.view(BT, H, W, C)
        # print(xs.shape)
        H, W, D = xs.shape
        xs = torch.reshape(xs,(1,H,W,D))
        xs = self.natten(xs)
        xs = torch.reshape(xs,(H,W,D))
        # xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch))) # or xs = self.natten(xs)
        # xs_patch = xs_patch.view(BT, L - 1, C)
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    

def build_AdapterV2(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter(input_dim, hidden_dim, output_dim))
    return nn.Sequential(*A_layers)

def build_AdapterV2a7x7(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=7))
    return nn.Sequential(*A_layers)

def build_AdapterV2a13x13(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=13))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5_slide(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_slide(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5_slide8(input_dim, hidden_dim, output_dim):
    print("build_AdapterV2a5x5_slide8")
    A_layers = list()
    A_layers.append(MLP_Adapter_slide8(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5_slide16(input_dim, hidden_dim, output_dim):
    print("build_AdapterV2a5x5_slide16")
    A_layers = list()
    A_layers.append(MLP_Adapter_slide16(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5_slide32(input_dim, hidden_dim, output_dim):
    print("build_AdapterV2a5x5_slide32")
    A_layers = list()
    A_layers.append(MLP_Adapter_slide32(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2a3x3(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, kernel_size=3))
    return nn.Sequential(*A_layers)


def build_AdapterV2a7x7R(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.ReLU, skip_connect=True, kernel_size=7))
    return nn.Sequential(*A_layers)

def build_AdapterV2a13x13R(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.ReLU, skip_connect=True, kernel_size=13))
    return nn.Sequential(*A_layers)

def build_AdapterV2a5x5R(input_dim, hidden_dim, output_dim):
    A_layers = list()
    A_layers.append(MLP_Adapter_a(input_dim, hidden_dim, output_dim, act_layer=nn.ReLU, skip_connect=True, kernel_size=5))
    return nn.Sequential(*A_layers)

def build_AdapterV2r(input_dim, hidden_dim, output_dim, ratio):
    A_layers = list()
    A_layers.append(MLP_Adapter_ratio(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, skip_connect=True, ratio=ratio))
    return nn.Sequential(*A_layers) 

def learnable_scalar():
    r = torch.nn.ParameterList([nn.Parameter(torch.tensor(1, dtype=torch.float32))]).to("cuda:0")
    return r

@TRANSFORMER_LAYER.register_module()
class BaseTransformerLayer_(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer_, self).__init__(init_cfg)

        self.batch_first = batch_first

        # assert set(operation_order) & set(
        #     ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
        #     set(operation_order), f'The operation_order of' \
        #     f' {self.__class__.__name__} should ' \
        #     f'contains all four operation type ' \
        #     f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = sum(1 for x in operation_order if 'attn' in x)
        print("num_attn", num_attn)
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'slide_attn', 'cross_attn', 'cross_attn_res_adapter', 
                                  'cross_attn_res_adapterV213x13', 'cross_attn_res_adapterV25x5', 
                                  'cross_attn_res_adapterV27x7', 'cross_attn_seq_adapterV25x5', 'cross_attn_seq_adapterV25x5_slide', 
                                  'cross_attn_seq_adapterV25x5_slide8', 'cross_attn_seq_adapterV25x5_slide16', 'cross_attn_seq_adapterV25x5_slide32']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1
                self.embed_dims = self.attentions[0].embed_dims
            if operation_name in ['slide_attn']:
                self.slideatten = SlideAttention(dim=256, num_heads=8, ka=3).cuda()
                # self.embed_dims = 256
                # self.adapter = build_Adapter(self.embed_dims, self.embed_dims // 2,self.embed_dims)
                # self.scalar = learnable_scalar()
            if operation_name in ['adapter_natten']:
                self.na2d = NeighborhoodAttention2D(dim=256, kernel_size=7, dilation=2, num_heads=8).cuda()
                self.embed_dims = 256
                self.adapter = build_Adapter(self.embed_dims, self.embed_dims // 2,self.embed_dims)
                self.scalar = learnable_scalar()
            if operation_name in ['adapter']:
                self.adapter = build_Adapter(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['res_adapter']:
                self.adapter_res = build_AdapterV2a5x5(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar = learnable_scalar()
            
            if operation_name in ['seq_adapter']:
                self.adapter_seq = build_Adapter(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['cross_attn_res_adapter']:
                self.adapter_att = build_Adapter(self.embed_dims, self.embed_dims // 2,self.embed_dims)
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_res_adapterV25x5']:
                self.adapter_att = build_AdapterV2a5x5(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_res_adapterV27x7']:
                self.adapter_att = build_AdapterV2a7x7(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_res_adapterV213x13']:
                self.adapter_att = build_AdapterV2a13x13(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_seq_adapterV25x5']:
                self.adapter_att = build_AdapterV2a5x5(self.embed_dims, self.embed_dims // 4,self.embed_dims)
                self.scalar_att = learnable_scalar()
            
            if operation_name in ['cross_attn_seq_adapterV25x5_slide']:
                self.adapter_att = build_AdapterV2a5x5_slide(self.embed_dims, self.embed_dims // 4,self.embed_dims) 
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_seq_adapterV25x5_slide8']:
                self.adapter_att = build_AdapterV2a5x5_slide8(self.embed_dims, self.embed_dims // 4,self.embed_dims) 
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_seq_adapterV25x5_slide16']:
                self.adapter_att = build_AdapterV2a5x5_slide16(self.embed_dims, self.embed_dims // 4,self.embed_dims) 
                self.scalar_att = learnable_scalar()

            if operation_name in ['cross_attn_seq_adapterV25x5_slide32']:
                self.adapter_att = build_AdapterV2a5x5_slide32(self.embed_dims, self.embed_dims // 4,self.embed_dims) 
                self.scalar_att = learnable_scalar()

            if operation_name in ['adapter_natten_V2']:
                self.adapter = build_AdapterV2(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a7x7']:
                self.adapter = build_AdapterV2a7x7(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a13x13']:
                self.adapter = build_AdapterV2a13x13(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a3x3']:
                self.adapter = build_AdapterV2a3x3(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a7x7R']:
                self.adapter = build_AdapterV2a7x7R(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a13x13R']:
                self.adapter = build_AdapterV2a13x13R(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a5x5R']:
                self.adapter = build_AdapterV2a5x5R(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a5x5']:
                self.adapter = build_AdapterV2a5x5(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2a13x13R']:
                self.adapter = build_AdapterV2a13x13R(self.embed_dims, self.embed_dims // 4, self.embed_dims)
                self.scalar = learnable_scalar()
                
            if operation_name in ['adapter_natten_V2r_0_25']:
                self.adapter = build_AdapterV2r(self.embed_dims, self.embed_dims // 4, self.embed_dims, 0.25)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2r_0_50']:
                self.adapter = build_AdapterV2r(self.embed_dims, self.embed_dims // 4, self.embed_dims, 0.50)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2r_0_75']:
                self.adapter = build_AdapterV2r(self.embed_dims, self.embed_dims // 4, self.embed_dims, 0.75)
                self.scalar = learnable_scalar()

            if operation_name in ['adapter_natten_V2r_0_10']:
                self.adapter = build_AdapterV2r(self.embed_dims, self.embed_dims // 4, self.embed_dims, 0.10)
                self.scalar = learnable_scalar()

                

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                # print("self_attn")
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'Nattn': 
                # print("Nattn")
                H, W, D = query.shape
                query = torch.reshape(query,(1,H,W,D))
                query = self.na2d(query)
                query = torch.reshape(query,(H,W,D))
                identity = query

            elif layer == 'norm':
                # print("norm")
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # print("cross_attn")
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif 'cross_attn_res_adapter' in layer:
                # print("cross_attn_res_adapter")
                before_res = query
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                res = self.adapter_att(before_res)
                query = res * self.scalar_att[0] + query
                identity = query
                # print('cross_attn_res_adapter')
            
            elif 'cross_attn_seq_adapter' in layer:
                # print("cross_attn_seq_adapter")
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                query = self.adapter_att(query)
                identity = query
                # print('cross_attn_res_adapter')

            elif layer == 'ffn':
                # print("ffn")
                identity_parallel = query
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

            elif 'seq_adapter' in layer: # seq identity is from self_attn
                # print("seq_adapter")
                seq = self.adapter_seq(identity)
                query = seq * self.scalar[0] + identity

            elif 'res_adapter' in layer: # parallel identity_parallel is from ffn
                # print("res_adapter")
                res = self.adapter_res(identity_parallel)
                query = res * self.scalar[0] + query

            elif 'adapter' in layer: # parallel
                # print("adapter")
                res = self.adapter(identity_parallel)
                query = res * self.scalar[0] + query

            elif layer == 'slide_attn':
                # print("slide_attn")
                H, W, C = query.shape
                query = torch.reshape(query,(1,C,H,W))
                query, _ , _ = self.slideatten(query)
                query = torch.reshape(query,(H,W,C))
                attn_index += 1
                identity = query

            

            # elif layer == 'adapter_natten': # parallel
            #     H, W, D = identity_adapter.shape
            #     queryna2 = identity_adapter
            #     queryna2 = torch.reshape(queryna2,(1,H,W,D))
            #     queryna2 = self.na2d(queryna2)
            #     queryna2 = torch.reshape(queryna2,(H,W,D))

            #     res = self.adapter(identity_adapter)

            #     query = res * self.scalar[0] + query + queryna2
                

        return query