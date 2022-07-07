from typing import Any, List

from torch import Tensor, ones
from torch.nn import Module, AvgPool2d, BatchNorm2d, Conv2d, Dropout,\
    GELU, LayerNorm, Linear, MultiheadAttention, Parameter, Sequential
from torch.nn.init import trunc_normal_, constant_


__all__ = [
    'EfficientFormer',
    'efficientformer_l1',
    'efficientformer_l3',
    'efficientformer_l7',
]


class Stem(Module):

    def __init__(self, in_channels: int, out_channels: int, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(Stem, self).__init__()
        self.stem = Sequential(
            Conv2d(in_channels, out_channels // 2, 3, 2, 1, **factory_kwargs),
            BatchNorm2d(out_channels // 2, **factory_kwargs),
            GELU(),
            Conv2d(out_channels // 2, out_channels, 3, 2, 1, **factory_kwargs),
            BatchNorm2d(out_channels, **factory_kwargs),
            GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class PoolMixer(Module):

    def __init__(self, kernel_size: int = 3):
        super(PoolMixer, self).__init__()
        self.pool = AvgPool2d(kernel_size, 1, kernel_size // 2, count_include_pad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x) - x


class MLP4D(Module):

    def __init__(self, in_channels: int, out_channels: int, expansion: int,
                 dropout:float = 0., device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MLP4D, self).__init__()
        self.feedforward = Sequential(
            Conv2d(in_channels, in_channels * expansion, 1, bias=False, **factory_kwargs),
            BatchNorm2d(in_channels * expansion, **factory_kwargs),
            GELU(),
            Dropout(p=dropout),
            Conv2d(in_channels * expansion, out_channels, 1, bias=False, **factory_kwargs),
            BatchNorm2d(out_channels, **factory_kwargs),
            Dropout(p=dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.feedforward(x)


class MLP3D(Module):

    def __init__(self, in_features: int, out_features: int, expansion: int,
                 dropout: float = 0., device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MLP3D, self).__init__()
        self.feedforward = Sequential(
            Linear(in_features, in_features * expansion, bias=True, **factory_kwargs),
            GELU(),
            Dropout(p=dropout),
            Linear(in_features * expansion, out_features, bias=True, **factory_kwargs),
            Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feedforward(x)


class MetaBlock4D(Module):
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(self, embed_dim: int, kernel_size: int = 3, dropout: float = 0.1,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MetaBlock4D, self).__init__()
        self.token_mixer = PoolMixer(kernel_size=kernel_size)
        self.feedforward = MLP4D(embed_dim, embed_dim, self.expansion, dropout, **factory_kwargs)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = Parameter(layer_scale_init * ones((1, embed_dim, 1, 1)), requires_grad=True)
            self.layer_scale_2 = Parameter(layer_scale_init * ones((1, embed_dim, 1, 1)), requires_grad=True)
        else:
            self.register_parameter('layer_scale_1', None)
            self.register_parameter('layer_scale_2', None)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if self.use_layer_scale:
            out = out + self.layer_scale_1 * self.token_mixer(out)
            out = out + self.layer_scale_2 * self.feedforward(out)
        else:
            out = out + self.token_mixer(out)
            out = out + self.feedforward(out)
        return out


class MetaBlock3D(Module):
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(self, embed_dim: int, dropout: float = 0.1,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MetaBlock3D, self).__init__()
        self.token_mixer = MultiheadAttention(embed_dim, 8, dropout, bias=False, add_bias_kv=True, **factory_kwargs)
        self.feedforward = MLP3D(embed_dim, embed_dim, self.expansion, dropout, **factory_kwargs)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = Parameter(layer_scale_init * ones((1, 1, embed_dim)), requires_grad=True)
            self.layer_scale_2 = Parameter(layer_scale_init * ones((1, 1, embed_dim)), requires_grad=True)
        else:
            self.register_parameter('layer_scale_1', None)
            self.register_parameter('layer_scale_2', None)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        nrm = self.norm1(out)
        if self.use_layer_scale:
            out = out + self.layer_scale_1 * self.token_mixer(nrm, nrm, nrm)[0]
            out = out + self.layer_scale_2 * self.feedforward(self.norm2(out))
        else:
            out = out + self.token_mixer(nrm, nrm, nrm)[0]
            out = out + self.feedforward(self.norm2(out))
        return out


class PatchEmbed(Module):

    def __init__(self, patch_size: int, stride: int, padding: int,
                 in_channels: int, out_channels: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(PatchEmbed, self).__init__()
        self.proj = Conv2d(in_channels, out_channels, patch_size, stride, padding, bias=False, **factory_kwargs)
        self.norm = BatchNorm2d(out_channels, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class Reshape(Module):

    def __init__(self, batch_first: bool = False) -> None:
        super(Reshape, self).__init__()
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        x = x.permute(2, 0, 1) if not self.batch_first else x.transpose(-2, -1)
        return x


class EfficientFormer(Module):

    def __init__(self, embed_dims: List[int], layers: List[int], down_sample: List[bool], num_vit: int,
                 num_classes: int = 1000, dropout: float = 0.1,
                 use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(EfficientFormer, self).__init__()
        features = [Stem(3, embed_dims[0], **factory_kwargs)]
        for stage in range(len(layers)):
            meta_blocks = self._make_layer(embed_dims[stage], stage, layers[stage], num_vit, 3,
                                           dropout, use_layer_scale, layer_scale_init, **factory_kwargs)
            features.append(meta_blocks)
            if stage >= len(layers) - 1:
                break
            if down_sample[stage] or embed_dims[stage] != embed_dims[stage + 1]:
                features.append(PatchEmbed(3, 2, 3 // 2, embed_dims[stage], embed_dims[stage + 1]))
        self.features = Sequential(*features)
        self.fc = Sequential(
            LayerNorm(embed_dims[3]),
            Linear(embed_dims[3], num_classes, **factory_kwargs),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, Linear) and m.bias is not None:
                    constant_(m.bias, 0)

    @staticmethod
    def _make_layer(embed_dim: int, stage: int, num_layers: int, num_vit: int, kernel_size: int = 3, dropout: int = 0.,
                    use_layer_scale: bool = True, layer_scale_init: float = 1e-5,
                    device=None, dtype=None) -> Sequential:
        factory_kwargs = dict(device=device, dtype=dtype)

        layers = []
        for block_index in range(num_layers):
            if stage == 3 and num_layers - block_index <= num_vit:
                layers.append(MetaBlock3D(embed_dim, dropout, use_layer_scale, layer_scale_init, **factory_kwargs))
            else:
                layers.append(MetaBlock4D(embed_dim, kernel_size, dropout, use_layer_scale, layer_scale_init, **factory_kwargs))
                if stage == 3 and num_layers - block_index - 1 == num_vit:
                    layers.append(Reshape())

        return Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientformer(pretrained: bool, embed_dim: List[int], layers: List[int],
                     down_sample:List[bool], num_vit: int, **kwargs: Any) -> EfficientFormer:
    model = EfficientFormer(embed_dims=embed_dim, layers=layers, down_sample=down_sample, num_vit=num_vit, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def efficientformer_l1(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L1 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [48, 96, 224, 448], [3, 2, 6, 4], [True, True, True, True], 1, **kwargs)


def efficientformer_l3(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L3 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [64, 128, 320, 512], [4, 4, 12, 6], [True, True, True, True], 3, **kwargs)


def efficientformer_l7(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L7 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [96, 192, 384, 768], [6, 6, 8, 8], [True, True, True, True], 8, **kwargs)
