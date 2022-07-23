import torch
import torch.nn as nn
from functools import partial

class MAE_ViT(nn.Module):
    def __init__(self,
                 img_size, patch_size, in_channels,
                 num_layers, num_heads, embed_dim,
                 decoder_num_layers, decoder_num_heads, decoder_embed_dim,
                 mlp_ratio, norm_layer,
                 qkv_bias, layer_scale, num_classes, weight_path=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder
        # --------------------------------------------------------------------------
        self.num_classes=num_classes
        self.weight_path=weight_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches

        # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                              layer_scale=layer_scale, norm_layer=norm_layer)
            for _ in range(num_layers)])

        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder
        # --------------------------------------------------------------------------

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                              layer_scale=layer_scale, norm_layer=norm_layer)
            for _ in range(decoder_num_layers)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_channels * patch_size ** 2, bias=True)

        # --------------------------------------------------------------------------

        self.initialize_weights()

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, dim, 1),
            # nn.Upsample(scale_factor = 2 ** i)
            nn.Upsample(scale_factor=4)
        ) for i, dim in enumerate([192, 192, 384, 768])])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(2 * embed_dim, embed_dim, 1),
            nn.Conv2d(embed_dim, self.num_classes, 1),
        )

    def initialize_weights(self):
        # initialize pos_embed, decoder_pos_embed by sin-cos embedding
        num_patches = self.patch_embed.num_patches

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape, num_patches, cls_token=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape, num_patches,
                                                                cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        P = self.patch_embed.patch_size
        N_H, N_W = H // P, W // P

        x = imgs.reshape(shape=(B, C, N_H, P, N_W, P))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(shape=(B, N_H * N_W, C * P ** 2))
        return x

    def unpatchify(self, x):
        B, N, D = x.shape
        P = self.patch_embed.patch_size
        H = W = int(N ** 0.5)

        x = x.reshape(shape=(B, H, W, P, P, 3))
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(shape=(B, 3, H * P, W * P))
        return imgs

    def load_weight(self):
        state = torch.load(self.weight_path)
        model_dict = state['state_dict']
        #
        img_size = self.img_size
        patch_size = self.patch_size

        model_dict['pos_embed'] = model_dict['pos_embed'].permute(0, 2, 1)  # b, p, c -> b, c, p
        class_token = model_dict['pos_embed'][:, :, 0].reshape(1, self.embed_dim, 1)
        model_dict['pos_embed'] = model_dict['pos_embed'][:, :, 1:].reshape(1, self.embed_dim, 14, 14)  # b, c, h, w
        model_dict['pos_embed'] = nn.functional.interpolate(
            model_dict['pos_embed'],
            size=int(img_size / patch_size),
            mode='bicubic',
            align_corners=True,
        )
        # b, c, h, w -> b, c, p
        model_dict['pos_embed'] = model_dict['pos_embed'].flatten(2)
        model_dict['pos_embed'] = torch.cat([class_token, model_dict['pos_embed']], dim=2)
        model_dict['pos_embed'] = model_dict['pos_embed'].permute(0, 2, 1)  # b, c, p -> b, p, c

        weights = {k for k in model_dict}  # use model1 not model2
        for k in weights.copy():
            if 'decoder' in k:
                del model_dict[k]

        self.load_state_dict(model_dict, strict=False)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape  # (B, num_patches, embed_dim)
        len_keep = int(N * (1 - mask_ratio))

        # get unmasked random idx and original idx
        noise = torch.rand(B, N, device=x.device)
        idx_shuffle = torch.argsort(noise, dim=1)
        idx_restore = torch.argsort(idx_shuffle, dim=1)

        idx_keep = idx_shuffle[:, :len_keep]  # (B, len_keep)
        x_masked = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, D))  # (B, len_keep, D)

        # mask: 0 for keeped idx, 1 for masked idx
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=idx_restore)

        return x_masked, mask, idx_restore

    def shape_change(self, x):
        b, p, c = x.shape
        h = int(np.sqrt(p))
        x = x.reshape(b, h, h, c)  # b, h, w, c
        x = x.permute(0, 3, 1, 2)  # b, c, h, w
        return x

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        pyramid_features = []
        # forward
        for _, block in enumerate(self.blocks):
            x = block(x)
            if _ == 3:
                pyramid_features.append(self.shape_change(x))
            if _ == 6:
                pyramid_features.append(self.shape_change(x))
            if _ == 9:
                pyramid_features.append(self.shape_change(x))
        x = self.norm(x)
        pyramid_features.append(self.shape_change(x))
        return pyramid_features

    def forward(self, imgs, mask_ratio=0.75):
        layer_outputs = self.forward_encoder(imgs, mask_ratio)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        outputs = self.to_segmentation(fused)
        outputs = nn.functional.softmax(outputs, dim=1)
        return outputs

def MAE_VIT_B16(qkv_bias=True, layer_scale=True, img_size=224, num_classes = 4):
    model = MAE_ViT(img_size=img_size, patch_size=16, in_channels=3,
                    num_layers=12, num_heads=12, embed_dim=768,
                    decoder_num_layers=8, decoder_num_heads=16, decoder_embed_dim=512,
                    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    qkv_bias=qkv_bias, layer_scale=layer_scale, num_classes=num_classes,
                    weight_path=r'./TCGA_MAE_ViT_10x_400_50_0399.pth.tar')
    return model


import torch

import numpy as np
import torch.nn as nn


# --------------------------------------------------------
# 2D sine-cosine position embedding
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_shape, num_patches, cls_token=False):
    embed_dim = embed_shape[-1]
    grid_size = int(num_patches ** 0.5)

    grid_w = np.arange(grid_size, dtype=float)  # (grid_size, )
    grid_h = np.arange(grid_size, dtype=float)  # (grid_size, )
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)

    grid = grid.reshape([2, 1, grid_size, grid_size])  # (2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (grid_size ** 2, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (grid_size ** 2, embed_dim // 2)
    embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (grid_size ** 2, embed_dim // 2)

    embed = np.concatenate([embed_h, embed_w], axis=1)  # (grid_size ** 2, embed_dim)
    return embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)  # (embed_dim // 4, )
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)  # (grid_size ** 2,)
    out = np.einsum('m,d->md', pos, omega)  # (grid_size ** 2, embed_dim // 4)

    embed_sin = np.sin(out)  # (grid_size ** 2, embed_dim // 4)
    embed_cos = np.cos(out)  # (grid_size ** 2, embed_dim // 4)

    embed = np.concatenate([embed_sin, embed_cos], axis=1)  # (grid_size ** 2, embed_dim // 2)
    return embed


# --------------------------------------------------------
# Patch Embedding
# --------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, img_size // patch_size, img_size // patch_size)
        x = self.proj(x)

        # (B, embed_dim, img_size//patch_size, img_size//patch_size) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


# --------------------------------------------------------
# Multi-Head Self-Attention
# --------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # (B, N, C) -> (B, N, 3C) -> (B, N, 3, num_heads, C // num_heads) -> (3, B, num_heads, N, C // num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, C // num_heads) -> (B, N, num_heads, C // num_heads) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# --------------------------------------------------------
# MLP
# --------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, bias=True, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --------------------------------------------------------
# utils
# --------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample

    '''

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if (not self.training) or (self.drop_prob == 0.):
            return x
        else:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                mask_tensor.div_(keep_prob)
            return x * mask_tensor


# --------------------------------------------------------
# Multi-Head Self-Attention Block
# --------------------------------------------------------

class Block(nn.Module):
    def __init__(self,
                 embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 mlp_ratio=0., mlp_drop=0.,
                 layer_scale=False, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale(embed_dim) if layer_scale else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), drop=mlp_drop)
        self.ls2 = LayerScale(embed_dim) if layer_scale else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x