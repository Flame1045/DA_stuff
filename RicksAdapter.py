from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmaction.utils import get_root_logger
from einops import rearrange
from ..builder import BACKBONES
from torch.nn import init
import random
import math
from natten import NeighborhoodAttention2D, NeighborhoodAttention3D, NeighborhoodAttention1D

# na1d = NeighborhoodAttention1D(dim=128, kernel_size=7, dilation=3, num_heads=4)
# na2d = NeighborhoodAttention2D(dim=128, kernel_size=7, dilation=3, num_heads=4)
# na3d = NeighborhoodAttention3D(dim=128, kernel_size=7, dilation=3, num_heads=4)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature = 4, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(1, 1)
        self.ln_1 = LayerNorm(1)

    def forward(self, x, mask=None):
        # k = 255 - k##new add
        q, k, v = x, x, x
        q, k, v = self.ln_1(q), self.ln_1(k), self.ln_1(v)
        attn = torch.matmul(q / self.temperature, k.transpose(-1, -2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = self.proj(output) + output

        return output

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class AttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # BN, T, C = x.shape
        B, N, T, C = x.shape

        # qkv = self.qkv(x).reshape(BN, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, T, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(BN, T, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Shift_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2):
        BT, N, C = x.shape
        BT, N, C = x2.shape
        # x2 = self.shift_tk(x)

        q = self.q(x).reshape(BT, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(BT, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BT, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        # assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws
        self.inner_attn = ScaledDotProductAttention(temperature=4)
        # self.ln_1 = LayerNorm(1)

    def forward(self, x, H = 16, W = 16):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        # q = self.v(x2).reshape(B, total_groups, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn + self.inner_attn(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, key_channels = 256, value_channels = 256, height = 16, width = 16, head_count=1, proj_drop=0.):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width
        self.proj = nn.Linear(key_channels, key_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reprojection = nn.Conv2d(value_channels, value_channels, 1)
        # self.norm = nn.LayerNorm(value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        # aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, N)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, D, N).permute(0, 2, 1)
        reprojected_value = self.proj(reprojected_value)
        reprojected_value = self.proj_drop(reprojected_value)        
        # reprojected_value = self.norm(reprojected_value)

        return reprojected_value

class Cross_Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        BT, N, C = x2.shape
        qk = self.qkv(x2).reshape(BT, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = x1.reshape(BT, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BT, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# #################################### Baseline(all)
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)        
        xs = self.D_fc1(x)

        xs = self.act(xs)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class Con1d_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1

    def forward(self, x):
        # x       
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()
        xs = self.act(xs)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
# #################################### Baseline(all)

################# Improved Adapter (all) +Cross Attention (spatial/joint) + Weighted Temporal Attention (temporal)
class Spatial_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_features = D_features
        self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        self.drop_path = nn.Identity()
        self.natten = NeighborhoodAttention2D(dim=D_hidden_features, kernel_size=5, dilation=1, num_heads=4)
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)

        xs = xs.permute(1, 0, 2)
        BT, L, C = xs.size()
        T = 16
        B = BT // T
        H = W = round(math.sqrt(L))
        assert L == H * W

        xs = xs.view(BT, H, W, C)
        xs = xs + self.drop_path(self.natten(self.ln_1(xs)))
        xs = xs.view(BT, L, C)
        xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
            # n, bt, d = x.size()
            # x = rearrange(x, 'n (b t) d -> b d t n', b = B, t = T, n = n).contiguous().view(B, d, T, H, W)
            x = x.permute(1, 0, 2).contiguous().view(BT, H, W, self.D_features)
            x = x.view(B, T, H, W, self.D_features).permute(0, 4, 1, 2, 3)
        else:
            x = xs
            x = x.permute(1, 0, 2).contiguous().view(BT, H, W, self.D_features)
            x = x.view(B, T, H, W, self.D_features).permute(0, 4, 1, 2, 3)
        return x

class Temporal_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_features = D_features
        self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        self.ln_1 = LayerNorm(D_hidden_features)
        self.drop_path = nn.Identity()
        self.natten3d = NeighborhoodAttention3D(dim=D_hidden_features, kernel_size=3, dilation=1, num_heads=4)

    def forward(self, x):
        # x is t (b n) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()
        
        xs = self.act(xs)

        ##scale dot product attention
        xs = xs.permute(1, 0, 2)
        BN, T, C = xs.size()
        N = 257
        B = BN // N
        H = W = round(math.sqrt(N - 1))
        assert N - 1 == H * W
        xs = xs.view(B, N, T, C)
        xs_cls = xs[:, 0:1, :, :]
        xs_patch = xs[:, 1:, :, :]

        xs_patch = xs_patch.view(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        xs_patch = xs_patch + self.drop_path(self.natten3d(self.ln_1(xs_patch)))
        xs_patch = xs_patch.permute(0, 2, 3, 1, 4).contiguous().view(B, N-1, T, C)
        xs = torch.cat((xs_cls, xs_patch), dim=1)
        xs = xs.view(BN, T, C).permute(1, 0, 2)
        ##scale dot product attention
        
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
            # x = x.permute(1, 0, 2).contiguous().view(B, N, T, self.D_features)
            # x = x.view(B, H, W, T, self.D_features).permute(0, 4, 3, 1, 2)
        else:
            x = xs
            # x = x.permute(1, 0, 2).contiguous().view(B, N, T, self.D_features)
            # x = x.view(B, H, W, T, self.D_features).permute(0, 4, 3, 1, 2)
        return x

class MLP_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_hidden_features = D_hidden_features
        self.act = act_layer()
        self.ln_1 = LayerNorm(D_hidden_features)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        self.drop_path = nn.Identity()
        self.natten = NeighborhoodAttention2D(dim=D_hidden_features, kernel_size=5, dilation=1, num_heads=4)
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()

        xs = self.act(xs)

        xs = xs.permute(1, 0, 2)
        BT, L, C = xs.size()
        T = 16
        B = BT // 16
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        xs_cls = xs[:, 0:1, :]
        xs_patch = xs[:, 1:, :]

        xs_patch = xs_patch.view(BT, H, W, C)
        xs_patch = xs_patch + self.drop_path(self.natten(self.ln_1(xs_patch)))
        xs_patch = xs_patch.view(BT, L - 1, C)
        xs = torch.cat((xs_cls, xs_patch), dim=1)
        xs = xs.permute(1, 0, 2)

        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class Spatial_Adapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        self.drop_path = nn.Identity() 
        self.ln_1 = LayerNorm(D_hidden_features)
        # self.attn_cross1 = CrossAttention(D_hidden_features, num_heads=8)
        self.group_attn  = GroupAttention(dim = D_hidden_features)
        # self.ch_attn = ChannelAttention(dim = D_hidden_features)
        # self.cross = Cross_Attention()
    
    def forward(self, x):
        # x is n (b t) d
        xs = self.D_fc1(x)

        xs2 = xs.transpose(1,2)
        xs2 = self.conv_B(self.dropout(self.conv_A(xs2)))*self.scale + xs2
        xs2 = xs2.transpose(1,2).contiguous()

        xs2 = self.act(xs2)

        # xs = self.ln_1(xs)
        # xs2 = self.ln_1(xs2)
        xs2 = xs2.permute(1, 0, 2)
        xs2_cls = xs2[:, 0:1, :]
        xs2_patch = xs2[:, 1:, :]

        # xs = xs.permute(1, 0, 2)
        # xs_patch = xs[:, 1:, :]
        
        # xs2_patch = xs2_patch + 0.5 * self.drop_path(self.cross(self.ln_1(xs2_patch), self.ln_1(xs_patch)))
        # # xs_cls = xs_cls + self.drop_path(self.attn_cross1(self.ln_1(xs_cls)))
        xs2_patch = xs2_patch + self.drop_path(self.group_attn(self.ln_1(xs2_patch)))
        xs2 = torch.cat((xs2_cls, xs2_patch), dim = 1)
        # xs = xs + self.drop_path(self.attn_cross1(self.ln_1(xs2)))
        # xs = 0.5 * xs + 0.5 * self.drop_path(self.ch_attn(self.ln_1(xs)))
        xs2 = xs2.permute(1, 0, 2)

        # xs = xs.permute(1, 0, 2)
        # xs_cls = xs[:, 0:1, :]
        # xs_patch = xs[:, 1:, :]
        # xs_patch = 0.5 * xs_patch + 0.5 * self.drop_path(self.group_attn(self.ln_1(xs_patch)))
        # xs = torch.cat((xs_cls, xs_patch), dim = 1)
        # xs = xs.permute(1, 0, 2)

        xs2 = self.D_fc2(xs2)

        if self.skip_connect:
            x = x + xs2
        else:
            x = xs2
        return x

class Temporal_Adapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.conv_A = nn.Conv1d(D_hidden_features, 64, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(64, D_hidden_features, 1, groups=1, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1
        self.drop_path = nn.Identity()
        self.ln_1 = LayerNorm(D_hidden_features)
        # self.attention = AttentionModule(dim = D_hidden_features)
        # self.shift_attention = Shift_Attention(dim = D_hidden_features)
        # self.natten3d = NeighborhoodAttention3D(dim=D_hidden_features, kernel_size=3, dilation=1, num_heads=8)
        self.fold_div = 8

    def shift_tk(self, x):
        t = 16
        bt, n, c = x.size()
        b = bt // t
        x = x.view(b, t, n, c) # B, T, N, C

        fold = c // self.fold_div
        out  = torch.zeros_like(x)
        out[:, :-1, 1:, :fold] = x[:, 1:, 1:, :fold] # shift left
        out[:, 1:,  1:, fold:2*fold] = x[:,:-1:, 1:, fold:2*fold]
        # out[:, -1, 1:, :fold] = x[:, -1, 1:, :fold]
        # out[:, 0, 1:, fold:2*fold] = x[:, 0, 1:, fold:2*fold]

        out[:, :, 0, :2*fold] = x[:, :, 0, :2*fold]
        out[:, :, :, 2*fold:] = x[:, :, :, 2*fold:]

        return out.view(bt, n, c)

    def forward(self, x):
        # x is t (b n) d
        xs = self.D_fc1(x)

        xs = xs.transpose(1,2)
        xs = self.conv_B(self.dropout(self.conv_A(xs)))*self.scale+xs
        xs = xs.transpose(1,2).contiguous()
        
        xs = self.act(xs)

        ##scale dot product attention
        xs = rearrange(xs, 't (b n) d -> (b t) n d', n = 257)
        xs2 = self.shift_tk(xs)
        xs = xs2 + self.drop_path(self.shift_attention(self.ln_1(xs), self.ln_1(xs2)))
        xs = rearrange(xs, '(b t) n d -> t (b n) d', t = 16)
        # xs = xs.permute(1, 0, 2)
        # out = self.attn_temporal(xs, xs, xs)
        # xs = (0.2 * xs) + (0.8 * out)
        # xs = xs.permute(1, 0, 2)
        # xs = rearrange(xs, 'b n t d -> t (b n) d')
        # xs = xs.permute(1, 0, 2)
        # xs = rearrange(xs, 't (b n) d -> b n t d', n = 257)
        # xs_cls = xs[:, 0:1, :, :]
        # xs_patch = xs[:, 1:, :, :]
        # xs_cls = 0.5 * xs_cls + 0.5 * self.drop_path(self.attention(self.ln_1(xs_cls)))
        # # xs_cls = self.drop_path(self.attention(xs_cls))
        # xs = torch.cat((xs_cls, xs_patch), dim=1)
        # xs = rearrange(xs, 'b n t d -> t (b n) d')
        # xs = xs.permute(1, 0, 2)
        ##scale dot product attention
        
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

################# Improved Adapter (all) +Cross Attention (spatial/joint) + Weighted Temporal Attention (temporal)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        # self.MLP_Adapter = Spatial_Adapter2(d_model, skip_connect=False)###joint
        # self.MLP_Adapter = MLP_Adapter(d_model, skip_connect=False)###joint
        self.MLP_Adapter = Con1d_Adapter(d_model, skip_connect=False)###joint
        # self.MLP_Adapter = Adapter(d_model, skip_connect=False)###joint

        # self.S_Adapter = Spatial_Adapter2(d_model)###spatial
        # self.S_Adapter = Spatial_Adapter(d_model)###spatial
        self.S_Adapter = MLP_Adapter(d_model)###spatial
        # self.S_Adapter = Con1d_Adapter(d_model)###spatial
        # self.S_Adapter = Adapter(d_model)###spatial
        self.scale = scale

        # self.T_Adapter = Temporal_Adapter2(d_model, skip_connect=False)###temporal
        # self.T_Adapter = Adapter(d_model, skip_connect=False)###temporal
        # self.T_Adapter = Con1d_Adapter(d_model, skip_connect=False)###temporal
        self.T_Adapter = Temporal_Adapter(d_model, skip_connect=False)###temporal
        # self.T_Adapter = Temporal_Adapter(d_model)###temporal

        if num_tadapter == 2:
            # self.T_Adapter_in = Temporal_Adapter2(d_model)###temporal
            # self.T_Adapter_in = Adapter(d_model)###temporal
            # self.T_Adapter_in = Con1d_Adapter(d_model)###temporal
            self.T_Adapter_in = Temporal_Adapter(d_model)###temporal

        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]

        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)

        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))   # Original Andre
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))  # Original Andrew
            # xt = self.attention(self.ln_1(xt))  # Original Andrew
            # xt = rearrange(xt, 't (b n) d -> t b n d', n=n)
            # xt_cls = xt[:, :, 0:1, :]
            # xt_patch = xt[:, :, 1:, :]
            # t1, b1, n1, d1 = xt_patch.shape
            # xt_patch = rearrange(xt_patch, 't1 b1 n1 d1 -> t1 (b1 n1) d1', t1=t1)
            # xt_patch = self.T_Adapter(xt_patch)
            # xt_patch = rearrange(xt_patch, 't1 (b1 n1) d1 -> t1 b1 n1 d1', n1=n1)
            # xt = torch.cat((xt_cls, xt_patch), dim=2)
            # xt = rearrange(xt, 't b n d -> t (b n) d', n=n)
            # xt = rearrange(xt, 't (b n) d -> b n t d', n=n)
            # xt_cls = xt[:, 0:1, :, :]
            # xt_patch = xt[:, 1:, :, :]
            # b1, n1, t1, d1 = xt_patch.shape
            # xt_patch = rearrange(xt_patch, 'b1 n1 t1 d1 -> t1 (b1 n1) d1', t1=t1)
            # xt_patch = self.T_Adapter(xt_patch)
            # xt_patch = rearrange(xt_patch, 't1 (b1 n1) d1 -> b1 n1 t1 d1', b1=b1, n1=n1)
            # xt = torch.cat((xt_cls, xt_patch), dim=1)
            # xt = rearrange(xt, 'b n t d -> t (b n) d', n=n)


        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)

        ## spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))  # Original Andrew

        # xs = self.attention(self.ln_1(x))
        # N, BT, D = xs.shape
        # xs_cls = xs[0:1, :, :]
        # xs_patch = xs[1:, :, :]
        
        # xs_patch = self.S_Adapter(xs_patch)
        # xs_patch = xs_patch.permute(3, 4, 0, 2, 1).contiguous().view(N - 1, BT, D)
        # xs = torch.cat((xs_cls, xs_patch), dim=0)
        # x = x + xs
        # x = torch.cat((x_cls, xs_patch), dim=0)

        ## joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@BACKBONES.register_module()
class ViT_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)
        # self.linear = nn.Linear(256, 1)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        ## Space-only
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        x = x[:, 0]

        # x = x[:, 1:]
        # x = self.linear(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        # x = x[:, 0]

        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        # x = rearrange(x[:,1:], '(b t) (h w) d -> b d t h w', b=B, t=T, h=16, w=16)

        return x
