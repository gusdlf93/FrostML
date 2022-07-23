# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vision OutLOoker (VOLO) implementation
"""
import math
import numpy as np
from itertools import chain

import torch.nn.functional as F
from torch.nn.functional import softmax

from torchvision import models

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from segmentation_model.decoder.UperNet.base import BaseModel
from segmentation_model.decoder.UperNet.utils.helpers import initialize_weights

from segmentation_model.utils import *

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'volo': _cfg(crop_pct=0.96),
    'volo_large': _cfg(crop_pct=1.15),
}


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """
    def __init__(self, dim, kernel_size, padding, stride=1,
                 num_heads=1,mlp_ratio=3., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim,
                            self.head_dim * self.num_heads * 2,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(
            B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    """
    get block by name, specifically for class attention block in here
    """
    if block_type == 'ca':
        return ClassBlock(**kargs)


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Layer_View(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

class img_View(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          padding=3, bias=False),  # 112x112
                Layer_View(),
                nn.LayerNorm(hidden_dim),
                img_View(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                Layer_View(),
                nn.LayerNorm(hidden_dim),
                img_View(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                Layer_View(),
                nn.LayerNorm(hidden_dim),
                img_View(),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3,
                     padding=1,stride=1, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                     attn_drop=0, drop_path_rate=0., **kwargs):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding,
                               stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.,
                       qkv_bias=False, qk_scale=None, attn_drop=0,
                       drop_path_rate=0., **kwargs):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            block_fn(dim, num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=block_dpr))

    blocks = nn.Sequential(*blocks)

    return blocks


class VOLO(nn.Module):
    """
    Vision Outlooker, the main class of our model
    --layers: [x,x,x,x], four blocks in two stages, the first block is outlooker, the
              other three are transformer, we set four blocks, which are easily
              applied to downstream tasks
    --img_size, --in_chans, --num_classes: these three are very easy to understand
    --patch_size: patch_size in outlook attention
    --stem_hidden_dim: hidden dim of patch embedding, d1-d4 is 64, d5 is 128
    --embed_dims, --num_heads: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios, --qkv_bias, --qk_scale, --drop_rate: easy to undertand
    --attn_drop_rate, --drop_path_rate, --norm_layer: easy to undertand
    --post_layers: post layers like two class attention layers using [ca, ca],
                  if yes, return_mean=False
    --return_mean: use mean of all feature tokens for classification, if yes, no class token
    --return_dense: use token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --mix_token: mixing tokens as token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --pooling_scale: pooling_scale=2 means we downsample 2x
    --out_kernel, --out_stride, --out_padding: kerner size,
                                               stride, and padding for outlook attention
    """
    def __init__(self, layers, img_size=224, in_chans=3, num_classes=1000, patch_size=8,
                 stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=None,
                 outlook_attention=None, mlp_ratios=None, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 post_layers=None, return_mean=False, return_dense=True, mix_token=True,
                 pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1, model_path=None):

        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size,
                                      in_chans=in_chans, hidden_dim=stem_hidden_dim,
                                      embed_dim=embed_dims[0])

        # inital positional encoding, we add positional encoding after outlooker blocks
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size // pooling_scale,
                        img_size // patch_size // pooling_scale,
                        embed_dims[-1]))

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.UperNet = UperNet(num_classes=4, in_channels=3, backbone='volo_d1', fpn_out=192)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers,
                                         downsample=downsamples[i], num_heads=num_heads[i],
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            else:
                # stage 2
                stage = transformer_blocks(Transformer, i, embed_dims[i], layers,
                                           num_heads[i], mlp_ratio=mlp_ratios[i],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path_rate=drop_path_rate,
                                           attn_drop=attn_drop_rate,
                                           norm_layer=norm_layer)
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.ModuleList(network)

        # set post block, for example, class attention layers
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([
                get_block(post_layers[i],
                          dim=embed_dims[-1],
                          num_heads=num_heads[-1],
                          mlp_ratio=mlp_ratios[-1],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate,
                          drop_path=0.,
                          norm_layer=norm_layer)
                for i in range(len(post_layers))
            ])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=.02)

        # set output type
        self.return_mean = return_mean  # if yes, return mean, not use class token
        self.return_dense = return_dense  # if yes, return class token and all feature tokens
        if return_dense:
            assert not return_mean, "cannot return both mean and dense"
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert return_dense, "return all tokens if mix_token is enabled"
        if return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        try:
            self.load_checkpoint(checkpoint=r'./d1_384_85.2.pth.tar')
        except:
            print(f"model_weight doesn't exists"
                  "you need to download model weights from https://github.com/sail-sg/volo")

        #########################jooly###############################
        self.low_conv = conv3x3(24, 64)
        self.attention_middle = attention(96, 128, 64)
        self.attention_high = attention(320, 512, 128)
        self.middle_conv = conv1x1(96, 64)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
        self.concate_conv = conv1x1(256, 192)
        self.upsampling2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners = True)
        self.conv0 = conv3x3(256, 128)
        self.conv1 = conv3x3(128, 64)
        self.classification = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def load_checkpoint(self, checkpoint=None):
        nw, nh = 44, 44
        state = torch.load(checkpoint, map_location='cpu')

        model_dict = self.state_dict()
        weights = {k: v for k, v in state.items() if k in model_dict}

        model_dict.update(weights)

        model_dict['pos_embed'] = model_dict['pos_embed'].permute(0, 3, 1, 2) # b h w c -> b, c, h , w
        model_dict['pos_embed'] = nn.functional.interpolate(
            model_dict['pos_embed'],
            size=nh,
            mode='bicubic',
            align_corners=True,
        )
        model_dict['pos_embed'] = model_dict['pos_embed'].permute(0, 2, 3, 1)

        del model_dict['aux_head.bias']
        del model_dict['aux_head.weight']
        del model_dict['head.weight']
        del model_dict['head.bias']
        self.load_state_dict(model_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        # patch embedding
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            if idx == 1:
                x1 = x.permute(0, 3, 1, 2)
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = block(x)
            if idx == 2:
                x2 = x.permute(0, 3, 1, 2)
            if idx == 3:
                x3 = x.permute(0, 3, 1, 2)
            if idx == 4:
                x4 = x.permute(0, 3, 1, 2)
        # B, H, W, C = x.shape
        # x = x.reshape(B, -1, C)
        return x1, x2, x3, x4

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward(self, x):
        # step1: patch embedding
        x = self.forward_embeddings(x)

        # # mix token, see token labeling for details.
        # if self.mix_token and self.training:
        #     lam = np.random.beta(self.beta, self.beta)
        #     patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
        #         2] // self.pooling_scale
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
        #     temp_x = x.clone()
        #     sbbx1,sbby1,sbbx2,sbby2=self.pooling_scale*bbx1,self.pooling_scale*bby1,\
        #                             self.pooling_scale*bbx2,self.pooling_scale*bby2
        #     temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
        #     x = temp_x
        # else:
        #     bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        # step2: tokens learning in the two stages
        x1, x2, x3, x4 = self.forward_tokens(x)
        x = x.permute(0, 3, 1, 2)

        output = self.UperNet(x, [x1, x2, x3, x4])
        # step3: post network, apply class attention or not
        # if self.post_network is not None:
        #     x = self.forward_cls(x)
        # x = self.norm(x)
        #
        # if self.return_mean:  # if no class token, return mean
        #     return self.head(x.mean(1))
        #
        # x_cls = self.head(x[:, 0])
        # if not self.return_dense:
        #     return x_cls
        #
        # x_aux = self.aux_head(
        #     x[:, 1:]
        # )  # generate classes in all feature tokens, see token labeling
        #
        # if not self.training:
        #     return x_cls + 0.5 * x_aux.max(1)[0]

        # if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
        #     x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
        #
        #     temp_x = x_aux.clone()
        #     temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
        #     x_aux = temp_x
        #
        #     x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        # return these: 1. class token, 2. classes from all feature tokens, 3. bounding box
        return output


@register_model
def volo_d1(pretrained=False, **kwargs):
    """
    VOLO-D1 model, Params: 27M
    --layers: [x,x,x,x], four blocks in two stages, the first stage(block) is outlooker,
            the other three blocks are transformer, we set four blocks, which are easily
             applied to downstream tasks
    --embed_dims, --num_heads,: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not in four blocks
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios: mlp ratio in four blocks
    --post_layers: post layers like two class attention layers using [ca, ca]
    See detail for all args in the class VOLO()
    """
    layers = [4, 4, 8, 2]  # num of layers in the four blocks
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False] # do downsampling after first block
    outlook_attention = [True, False, False, False ]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)

    model.default_cfg = default_cfgs['volo']
    return model

@register_model
def volo_d2(pretrained=False, **kwargs):
    """
    VOLO-D2 model, Params: 59M
    """
    layers = [6, 4, 10, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model


@register_model
def volo_d3(pretrained=False, **kwargs):
    """
    VOLO-D3 model, Params: 86M
    """
    layers = [8, 8, 16, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model


@register_model
def volo_d4(pretrained=False, **kwargs):
    """
    VOLO-D4 model, Params: 193M
    """
    layers = [8, 8, 16, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo_large']
    return model


@register_model
def volo_d5(pretrained=False, **kwargs):
    """
    VOLO-D5 model, Params: 296M
    stem_hidden_dim=128, the dim in patch embedding is 128 for VOLO-D5
    """
    layers = [12, 12, 20, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 stem_hidden_dim=128,
                 **kwargs)
    model.default_cfg = default_cfgs['volo_large']
    return model

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            Layer_View(),
            nn.LayerNorm(in_channels),
            img_View(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        lv  = Layer_View()
        ln = nn.LayerNorm(out_channels)
        iv = img_View()
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, lv, ln, iv, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]#features must be b, c, h, w
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            Layer_View(),
            nn.LayerNorm(fpn_out),
            img_View(),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet(BaseModel):
    # Implementing only the object path
    def __init__(self, num_classes=4, in_channels=3, backbone='resnet101', pretrained=False, use_aux=True, fpn_out=256,
                 freeze_bn=False, **_):
        super(UperNet, self).__init__()

        if backbone == 'resnet34' or backbone == 'resnet18':
            feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet101':
            feature_channels = [256, 512, 1024, 2048]
        elif backbone == 'volo_d1':
            feature_channels = [192, 384, 384, 384]

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        self.softmax = softmax
        if freeze_bn: self.freeze_bn()
        # if freeze_backbone:
        #     set_trainable([self.backbone], False)

    def forward(self, x, features):
        features[-1] = self.PPN(features[-1])
        x = self.FPN(features)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = self.head(x)
        x = softmax(x, dim=1)
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm): module.eval()