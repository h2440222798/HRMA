# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from torch import Tensor
from torch.types import _size

logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "hrvit_224": _cfg(),
}


class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

# 激活函数采用的是GELU

from torch.nn import Dropout
class MixCFN(nn.Module):
    def __init__(
        self,
        in_features: int,                        # 输入特征维度
        hidden_features: Optional[int] = None,   # 隐层特征维度，若不指定则默认与输入特征维度相同
        out_features: Optional[int] = None,      # 输出特征维度，若不指定则默认与输入特征维度相同
        act_func: nn.Module = nn.GELU,           # 激活函数，使用GELU函数
        with_cp: bool = False,                   # 是否使用checkpoint加速计算
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_cp = with_cp
        # 第一个全连接层，输入特征维度为in_features，输出特征维度为hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dropout = Dropout(0.1)
        # 混合卷积层，输入特征维度为hidden_features，输出特征维度为hidden_features
        # 使用3x3的卷积核，填充为1，组数为hidden_features，空洞率为1，带有偏置项
        # 激活函数，使用GELU函数
        self.act = act_func()
        # 第二个全连接层，输入特征维度为hidden_features，输出特征维度为out_features
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            # 第一个全连接层的前向传播过程
            x = self.fc1(x)
            # B, N, C = x.shape
            # 将第一个全连接层输出的特征向量转换为混合卷积层的输入特征矩阵
            # x = self.conv(x.transpose(1, 2).view(B, C, H, W))
            # 使用激活函数
            x = self.act(x)
            x = self.dropout(x)
            # 第二个全连接层的前向传播过程，将混合卷积层输出的特征矩阵展平后作为全连接层的输入
            x = self.fc2(x)
            x = self.dropout(x)
            return x

        # 判断是否使用checkpoint加速计算
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

# 这是一个HRVit模型中
class HRViTAttention(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,  # 输入的特征维度
        dim: int = 64,  # 注意力层的维度
        heads: int = 2,  # 注意力头的数量
        ws: int = 1,  # 窗口大小
        qk_scale: Optional[float] = None,  # 缩放因子
        proj_drop: float = 0.0,  # 投影层中的dropout概率
        with_cp: bool = False,  # 是否使用checkpointing技术
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.in_dim = in_dim
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads  # 每个头的维度
        self.ws = ws  # 窗口大小
        self.with_cp = with_cp

        self.to_qkv = nn.Linear(in_dim, 2 * dim)  # 输入到Q,K,V的线性投影层

        self.scale = qk_scale or self.dim_head ** -0.5  # 缩放因子

        self.attend = nn.Softmax(dim=-1)  # 注意力计算中的softmax操作

        self.attn_act = nn.Hardswish(inplace=True)  # 激活函数

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),  # 输出线性变换
            nn.Dropout(proj_drop),  # dropout层
        )

        self.attn_bn = nn.BatchNorm1d(
            dim, momentum=1 - 0.9 ** 0.5 if self.with_cp else 0.1  # batch normalization
        )
        nn.init.constant_(self.attn_bn.bias, 0)  # bn层的bias初始化为0
        nn.init.constant_(self.attn_bn.weight, 0)  # bn层的weight初始化为0

        self.parallel_conv = nn.Sequential(
            nn.Hardswish(inplace=False),  # 激活函数
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,  # depth-wise卷积
            ),
        )

        # 增加一些卷积操作
        out_planes = self.in_dim//2
        self.kernel_conv = 3
        self.fc = nn.Conv2d(self.heads*3,self.kernel_conv*self.kernel_conv,kernel_size=1,bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv*self.kernel_conv*self.dim_head//2,out_planes,kernel_size=self.kernel_conv,bias=True,groups=self.dim_head//2,padding=1,stride=1)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.q_conv_336 = torch.nn.Linear(336, 256)
        self.k_conv_336 = torch.nn.Linear(336, 256)
        self.v_conv_336 = torch.nn.Linear(336, 256)



    @lru_cache(maxsize=4)
    def _generate_attn_mask(self, h: int, hp: int, device):
        x = torch.empty(hp, hp, device=device).fill_(-100.0)
        x[:h, :h] = 0
        return x

    def _cross_shaped_attention(
            self,
            q: Tensor,  # 查询张量，大小为[B, N, C]
            k: Tensor,  # 键张量，大小为[B, N, C]
            v: Tensor,  # 值张量，大小为[B, N, C]
            H: int,  # 高度
            W: int,  # 宽度
            HP: int,  # 查询区域的高度
            WP: int,  # 查询区域的宽度
            ws: int,  # 查询区域的步幅
            horizontal: bool = True,  # 是否为水平方向
    ) -> Tensor:
        """
        Args:
            q: 查询张量，大小为[B, N, C]
            k: 键张量，大小为[B, N, C]
            v: 值张量，大小为[B, N, C]
            H: 高度
            W: 宽度
            HP: 查询区域的高度
            WP: 查询区域的宽度
            ws: 查询区域的步幅
            horizontal: 是否为水平方向
        Returns:
            attn: 注意力张量，大小为[B, H_1 * W_1, D]
        """
        B, N, C = q.shape
        #对q,k,v做卷积
        # print('q_shape',q.shape)
        # print('其他参数：H : {}, W : {}, self.heads : {}, self.dim_head : {}'.format(H,W,self.heads,self.dim_head))

        if H*W != N:
            q_test = q.view(B,C,-1)
            k_test = k.view(B,C,-1)
            v_test = v.view(B,C,-1)
            q_conv = self.q_conv_336(q_test)
            k_conv = self.k_conv_336(k_test)
            v_conv = self.v_conv_336(v_test)
            q_conv = q_conv.view(B,H*W,C)
            k_conv = k_conv.view(B,H*W,C)
            v_conv = v_conv.view(B,H*W,C)
        else:
            q_conv = q
            k_conv = k
            v_conv = v

        f_all = self.fc(torch.cat([q_conv.reshape(B,self.heads,self.dim_head//2,H*W),k_conv.reshape(B,self.heads,self.dim_head//2,H*W),v_conv.reshape(B,self.heads,self.dim_head//2,H*W)],1))
        f_conv =  f_all.permute(0,2,1,3).reshape(B,-1,H,W)

        out_conv = self.dep_conv(f_conv).permute(0,2,3,1).reshape(B,-1,C)



        # B, N, C = q.shape
        if C < self.dim_head:  # 如果通道数小于头的维度，则使用C作为头的维度
            dim_head = C
            scale = dim_head ** -0.5 # 根号下d分之1
        else:
            scale = self.scale  # 缩放因子，取值为sqrt(dim_head)
            dim_head = self.dim_head

        if horizontal:
            # 对输入的q、k、v进行reshape、permute和flatten操作
            q, k, v = map(
                lambda y: y.reshape(B, HP // ws, ws, W, C // dim_head, -1)
                .permute(0, 1, 4, 2, 3, 5)
                .flatten(3, 4),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda y: y.reshape(B, H, WP // ws, ws, C // dim_head, -1)
                .permute(0, 2, 4, 3, 1, 5)
                .flatten(3, 4),
                (q, k, v),
            )

        # 计算注意力矩阵
        attn = q.matmul(k.transpose(-2, -1)).mul(
            scale
        )  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),(b1*b2+1)*(ws*ws)]

        # 在softmax之前，需要对填充值进行mask
        if horizontal and HP != H:
            attn_pad = attn[:, -1:]  # 取出填充部分
            mask = self._generate_attn_mask(
                h=(ws - HP + H) * W, hp=attn.size(-2), device=attn.device
            )  # 生成掩码
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        if not horizontal and WP != W:
            attn_pad = attn[:, -1:]  # [B, 1, num_head, ws*H, ws*H]
            mask = self._generate_attn_mask(
                h=(ws - WP + W) * H, hp=attn.size(-2), device=attn.device
            )  # [ws*H, ws*H]
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        attn = self.attend(attn)

        attn = attn.matmul(v)  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),D//h]

        attn = rearrange(
            attn,
            "B H h (b W) d -> B (H b) W (h d)"
            if horizontal
            else "B W h (b H) d -> B H (W b) (h d)",
            b=ws,
        )  # [B,H_1, W_1,D]
        if horizontal and HP != H:
            attn = attn[:, :H, ...]
        if not horizontal and WP != W:
            attn = attn[:, :, :W, ...]
        attn = attn.flatten(1, 2)
        # if attn.shape != out_conv.shape:
        #     print('attn_shape', attn.shape)
        return self.rate1*attn + self.rate2*out_conv

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            B = x.shape[0]
            ws = self.ws
            qv = self.to_qkv(x)
            q, v = qv.chunk(2, dim=-1)

            v_conv = (
                self.parallel_conv(v.reshape(B, H, W, -1).permute(0, 3, 1, 2))
                .flatten(2)
                .transpose(-1, -2)
            )

            qh, qv = q.chunk(2, dim=-1)
            vh, vv = v.chunk(2, dim=-1)
            kh, kv = vh, vv  # share key and value

            # padding to a multple of window size
            if H % ws != 0:
                HP = int((H + ws - 1) / ws) * ws
                qh = (
                    F.pad(
                        qh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vh = (
                    F.pad(
                        vh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kh = vh
            else:
                HP = H

            if W % ws != 0:
                WP = int((W + ws - 1) / ws) * ws
                qv = (
                    F.pad(
                        qv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vv = (
                    F.pad(
                        vv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kv = vv
            else:
                WP = W

            attn_h = self._cross_shaped_attention(qh,kh,vh,H,W,HP,W,ws,horizontal=True,)
            attn_v = self._cross_shaped_attention(qv,kv,vv,H,W,H,WP,ws,horizontal=False,)

            attn = torch.cat([attn_h, attn_v], dim=-1)
            attn = attn.add(v_conv)
            attn = self.attn_act(attn)

            attn = self.to_out(attn)
            attn = self.attn_bn(attn.flatten(0, 1)).view_as(attn)
            return attn

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

    def extra_repr(self) -> str:
        s = f"window_size={self.ws}"
        return s


class HRViTBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ws: int = 1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp

        # build layer normalization
        self.attn_norm = nn.LayerNorm(in_dim)

        # build attention layer
        self.attn = HRViTAttention(
            in_dim=in_dim,
            dim=dim,
            heads=heads,
            ws=ws,
            proj_drop=proj_dropout,
            with_cp=with_cp,
        )

        # build diversity-enhanced shortcut DES
        # build drop path
        self.attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # build layer normalization
        self.ffn_norm = nn.LayerNorm(in_dim)

        # build FFN
        self.ffn = MixCFN(
            in_features=in_dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_func=nn.GELU,
            with_cp=with_cp,
        )

        # build drop path
        self.ffn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # attention block
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, H, W)
        x = self.attn_drop_path(x).add(res)

        # ffn block
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x, H, W)
        x = self.ffn_drop_path(x).add(res)

        return x


class HRViTPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,           # 输入图片的通道数
        patch_size: _size = 3,          # patch 的大小
        stride: int = 1,                # patch 移动的步长
        dim: int = 64,                  # embedding 后的特征维度
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = to_2tuple(patch_size)   # 将 patch_size 转换成二元组形式
        self.dim = dim                            # 特征维度

        self.padding_conv_30 = nn.Conv2d(30, self.in_channels, kernel_size=1, padding=0, stride=1)
        self.padding_conv_63 = nn.Conv2d(63, self.in_channels, kernel_size=1, padding=0, stride=1)
        self.padding_conv_126 = nn.Conv2d(126, self.in_channels, kernel_size=1, padding=0, stride=1)
        self.padding_conv_255 = nn.Conv2d(255, self.in_channels, kernel_size=1, padding=0, stride=1)


        # embedding 的卷积操作，包括 1x1 卷积和 patch 的卷积
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels,dim,kernel_size=1,stride=1,padding=0,),  # 1x1 卷积，调整通道数
            nn.Conv2d(dim,dim,kernel_size=self.patch_size,stride=stride,padding=(self.patch_size[0] // 2, self.patch_size[1] // 2),groups=dim,),  # patch 卷积
        )

        self.norm = nn.LayerNorm(dim)  # 归一化层

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x_dim = x.shape[1]
        if x_dim != self.in_channels:
            if x_dim is 30:
                x = self.padding_conv_30(x)
            elif x_dim is 63:
                x = self.padding_conv_63(x)
            elif x_dim is 126:
                x = self.padding_conv_126(x)
            elif x_dim is 255:
                x = self.padding_conv_255(x)

        x = self.proj(x)            # 前向传播，embedding
        H, W = x.shape[-2:]         # 记录下 embedding 后的 feature map 的长和宽
        x = x.flatten(2).transpose(1, 2)   # 将 feature map 展平，并转置
        x = self.norm(x)            # 归一化操作

        return x, H, W             # 返回特征 embedding、embedding 后的 feature map 的长和宽



class HRViTFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int] = (32, 64, 128, 256),
        out_channels: Tuple[int] = (32, 64, 128, 256),
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_func = act_func
        self.with_cp = with_cp
        self.n_outputs = len(out_channels)
        self._build_fuse_layers()

    def _build_fuse_layers(self):
        self.blocks = nn.ModuleList([])
        n_inputs = len(self.in_channels)
        for i, outc in enumerate(self.out_channels):
            blocks = nn.ModuleList([])
            start = 0
            end = n_inputs
            print("outc_",str(outc),"end-start",str(end-start))
            outc = outc // (end - start)
            for j in range(start, end):
                inc = self.in_channels[j]
                if j == i:
                    blocks.append(nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ))
                elif j < i:
                    block = [
                        nn.Conv2d(
                            inc,
                            inc,
                            kernel_size=2 ** (i - j) + 1,
                            stride=2 ** (i - j),
                            dilation=1,
                            padding=2 ** (i - j) // 2,
                            groups=inc,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inc),
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),
                    ]

                    blocks.append(nn.Sequential(*block))

                else:
                    block = [nn.Conv2d(
                        inc,
                        outc,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        padding=0,
                        groups=1,
                        bias=True,
                    ), nn.BatchNorm2d(outc), nn.Upsample(
                        scale_factor=2 ** (j - i),
                        mode="nearest",
                    )]

                    blocks.append(nn.Sequential(*block))
            self.blocks.append(blocks)

        self.act = nn.ModuleList([self.act_func() for _ in self.out_channels])

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:

        out = [None] * len(self.blocks)
        n_inputs = len(x)

        for i, (blocks, act) in enumerate(zip(self.blocks, self.act)):
            start = 0
            end = n_inputs
            for j, block in zip(range(start, end), blocks):
                # if out[i] is not None:
                #     print(n_inputs)
                #     print("out_i_shape",out[i].shape)
                # print("block_j_shape",block(x[j]).shape)
                out[i] = block(x[j]) if out[i] is None else torch.cat([out[i],block(x[j])],dim=1)
            out[i] = act(out[i])

        return out


class HRViTStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 4,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        stride = (stride[0]//2, stride[1]//2)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [
            (dilation[i] * (kernel_size[i] - 1) + 1) // 2
            for i in range(len(kernel_size))
        ]


        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        feature = []
        feature.append(x)
        x = self.act1(self.bn1(self.conv1(x)))
        feature.append(x)
        x = self.act2(self.bn2(self.conv2(x)))
        feature.append(x)
        return x,feature


class HRViTStage(nn.Module):
    def __init__(
        self,
        #### Patch Embed Config ####
        in_channels: Tuple[
            int,
        ] = (32, 64, 128, 256),
        out_channels: Tuple[
            int,
        ] = (32, 64, 128, 256),
        block_list: Tuple[
            int,
        ] = (1, 1, 6, 2),
        #### HRViTAttention Config ####
        dim_head: int = 32,
        ws_list: Tuple[
            int,
        ] = (1, 2, 7, 7),
        proj_dropout: float = 0.0,
        drop_path_rates: Tuple[float] = (
            0.0,
        ),  # different droprate for different attn/mlp
        #### MixCFN Config ####
        mlp_ratio_list: Tuple[
            int,
        ] = (4, 4, 4, 4),
        dropout: float = 0.0,
        #### Gradient Checkpointing #####
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.ModuleList(
            [
                HRViTPatchEmbed(
                    in_channels=inc,
                    patch_size=3,
                    stride=1,
                    dim=outc,
                )
                for inc, outc in zip(in_channels, out_channels)
            ]
        )  # one patch embedding for each branch

        ## we arrange blocks in stages/layers
        n_inputs = len(out_channels)

        self.branches = nn.ModuleList([])
        for i, n_blocks in enumerate(block_list[:n_inputs]):
            blocks = []
            for j in range(n_blocks):
                blocks += [
                    HRViTBlock(
                        in_dim=out_channels[i],
                        dim=out_channels[i],
                        heads=out_channels[i] // dim_head,  # automatically derive heads
                        proj_dropout=proj_dropout,
                        mlp_ratio=mlp_ratio_list[i],
                        drop_path=drop_path_rates[j],
                        ws=ws_list[i],
                        with_cp=with_cp,
                    )
                ]

            blocks = nn.ModuleList(blocks)
            self.branches.append(blocks)
        self.norm = nn.ModuleList([nn.LayerNorm(outc) for outc in out_channels])

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:
        B = x[0].shape[0]
        x = list(x)
        H, W = [], []
        ## patch embed
        for i, (xx, embed) in enumerate(zip(x, self.patch_embed)):
            xx, h, w = embed(xx)
            x[i] = xx
            H.append(h)
            W.append(w)

        ## HRViT blocks
        for i, (branch, h, w) in enumerate(zip(self.branches, H, W)):
            for block in branch:
                x[i] = block(x[i], h, w)

        ## LN at the end of each stage
        for i, (xx, norm, h, w) in enumerate(zip(x, self.norm, H, W)):
            xx = norm(xx)
            xx = xx.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = xx
        return x


class HRViT(nn.Module):
    def __init__(
        self,
        #### HRViT Stem Config ####
        in_channels: int = 3,
        stride: int = 4,
        channels: int = 64,
        #### Branch Config ####
        channel_list: Tuple[Tuple[int,],] = (
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
        ),
        block_list: Tuple[Tuple[int]] = (
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
        ),
        #### HRViTAttention Config ####
        dim_head: int = 32,
        ws_list: Tuple[
            int,
        ] = (1, 2, 7),
        proj_dropout: float = 0.0,
        drop_path_rate: float = 0.0,  # different droprate for different attn/mlp
        #### HRViTFeedForward Config ####
        mlp_ratio_list: Tuple[
            int,
        ] = (4, 4, 4),
        dropout: float = 0.0,
        #### Classification Head Config ####
        num_classes: int = 1000,
        head_dropout: float = 0.1,
        #### Gradient Checkpointing #####
        with_cp: bool = False,
    ) -> None:
        super().__init__()

        self.features = []
        self.ws_list = ws_list
        self.head_dropout = head_dropout
        self.with_cp = with_cp

        # calculate drop path rates
        total_blocks = sum(max(b) for b in block_list)

        total_drop_path_rates = (
            torch.linspace(0, drop_path_rate, total_blocks).numpy().tolist()
        )

        cur = 0
        self.channel_list = channel_list = [[channels]] + list(channel_list)

        # build stem
        self.stem = HRViTStem(
            in_channels=in_channels, out_channels=channels, kernel_size=3, stride=4
        )

        # build backbone
        for i, blocks in enumerate(block_list):
            inc, outc = channel_list[i : i + 2]
            depth_per_stage = max(blocks)

            self.features.extend(
                [
                    HRViTFusionBlock(
                        in_channels=inc,
                        out_channels=inc
                        if len(inc) == len(outc)
                        else list(inc) + [outc[-1]],
                        act_func=nn.GELU,
                        with_cp=False,
                    ),
                    HRViTStage(
                        #### Patch Embed Config ####
                        in_channels=inc
                        if len(inc) == len(outc)
                        else list(inc) + [outc[-1]],
                        out_channels=outc,
                        block_list=blocks,
                        dim_head=dim_head,
                        #### HRViTBlock Config ####
                        ws_list=ws_list,
                        proj_dropout=proj_dropout,
                        drop_path_rates=total_drop_path_rates[
                            cur : cur + depth_per_stage
                        ],  # different droprate for different attn/mlp
                        #### MixCFN Config ####
                        mlp_ratio_list=mlp_ratio_list,
                        dropout=dropout,
                        #### Gradient Checkpointing #####
                        with_cp=with_cp,
                    ),
                ]
            )
            cur += depth_per_stage

        self.features = nn.Sequential(*self.features)
    def forward_features(
        self, x: Tensor
    ) -> Tuple[Tensor,]:
        # stem
        x,feature = self.stem(x)
        # backbone
        x = self.features((x,))
        return x,feature

    def forward(self, x: Tensor) -> Tensor:
        # stem and backbone
        x,feature = self.forward_features(x)
        # classifier
        # x = self.head(x)
        return x,feature


@register_model
def HRViT_b1_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128, 256),
            (32, 64, 128, 256),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 2, 2),
        ),
        dim_head=32,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(4, 4, 4, 4),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model


@register_model
def HRViT_b2_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (48,),
            (48, 96),
            (48, 96, 240),
            (48, 96, 240),
            (48, 96, 240, 384),
            (48, 96, 240, 384),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 6, 2),
        ),
        dim_head=24,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(2, 3, 3, 3),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model


@register_model
def HRViT_b3_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (64,),
            (64, 128),
            (64, 128, 256),
            (64, 128, 256),
            (64, 128, 256, 512),
            (64, 128, 256, 512),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 3),
            (1, 1, 6, 3),
        ),
        dim_head=32,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(2, 2, 2, 2),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model

if __name__ == '__main__':

    model = HRViT_b1_224(False)

    # model = CAN(out_channels=1)
    print(model)

    # x_np = np.ones((2,1,192,192),dtype=np.float32)
    # x_tensor = torch.tensor(x_np)(64, 192,160)
    x_tensor = torch.randn((6, 3, 256, 256))
    y = model(x_tensor)
    print(y)
    # print(y.shape)
    # import torch
    from thop import profile

    #
    # inputs = torch.randn(2, 1, 192, 192)
    # # model = DenseUNet(1, 3)
    #
    flops, params = profile(model, inputs=(x_tensor,))
    print('flops: {}, params: {}'.format(flops, params))