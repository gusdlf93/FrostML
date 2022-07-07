from typing import Dict, Optional

import copy
import torch

from torch import Tensor
from torch.nn import Module, ModuleList, Conv2d, Dropout, GELU, LayerNorm, Linear, MultiheadAttention, Parameter
from torch.nn.init import xavier_uniform_
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import generalized_box_iou
from frostml.vision.utils.ops import box_cxcywh_to_xyxy


__all__ = ['DETR', 'HungarianMatcher']


class DETR(Module):

    def __init__(self, backbone: Module, transformer: Module,
                 num_classes: int = 91, num_queries: int = 100,
                 backbone_out_channels: int = 512,
                 backbone_out_resolution: int = 49,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETR, self).__init__()
        self.backbone = backbone
        assert hasattr(transformer, 'embed_dim')
        self.bridge = Reshape(backbone_out_channels, transformer.embed_dim, **factory_kwargs)
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pos = Parameter(torch.zeros((backbone_out_resolution, 1, transformer.embed_dim), **factory_kwargs), True)
        self.pos_queries = Parameter(torch.zeros((num_queries, 1, transformer.embed_dim), **factory_kwargs), True)
        self.pred_labels = Linear(transformer.embed_dim, num_classes + 1, **factory_kwargs)
        self.pred_bboxes = Linear(transformer.embed_dim, 4, **factory_kwargs)

    def forward(self, x: Tensor, object_queries: Tensor, mask: Optional[Tensor] = None) -> Dict[Tensor, Tensor]:
        out = x
        out = self.backbone(out)
        out = self.bridge(out)
        out = self.transformer(out, object_queries, mask, self.pos, self.pos_queries)
        out = out.transpose(1, 0)
        out = {
            'labels': self.pred_labels(out),
            'bboxes': self.pred_bboxes(out).sigmoid(),
        }
        return out


class Reshape(Module):

    def __init__(self, in_channels: int, out_channels: int, batch_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(Reshape, self).__init__()
        self.linear = Linear(in_channels, out_channels, **factory_kwargs)
        self.conv = Conv2d(in_channels, out_channels, 1, **factory_kwargs)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = self.linear(x)
            return x
        if len(x.shape) == 4:
            x = self.conv(x)
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w)
            x = x.permute(2, 0, 1) if not self.batch_first else x.transpose(-2, -1)
            return x
        raise ValueError


class Transformer(Module):

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 6, num_decoder_layer: int = 6,
                 dropout: float = 0., prenorm: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(Transformer, self).__init__()
        encoderlayer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, prenorm, **factory_kwargs)
        decoderlayer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout, prenorm, **factory_kwargs)
        encoder_norm = LayerNorm(embed_dim) if prenorm else None
        decoder_norm = LayerNorm(embed_dim) if prenorm else None
        self.encoder = TransformerEncoder(encoderlayer, num_encoder_layer, encoder_norm)
        self.decoder = TransformerDecoder(decoderlayer, num_decoder_layer, decoder_norm)

        self._reset_parameters()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src: Tensor, tgt: Tensor, mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pos_queries: Optional[Tensor] = None) -> Tensor:
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        output = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos, pos_queries=pos_queries)
        return output


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _with_pos_encoding(tensor: Tensor, pos: Optional[Tensor] = None):
    return tensor if pos is None else tensor + pos


class TransformerEncoder(Module):
    
    def __init__(self, layer: Module, num_layers: int, norm: Optional[Module] = None) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoder(Module):

    def __init__(self, layer: Module, num_layers: int, norm: Optional[Module] = None) -> None:
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pos_queries: Optional[Tensor] = None) -> Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(out, memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, pos_queries=pos_queries)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerEncoderLayer(Module):

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0., prenorm: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(TransformerEncoderLayer, self).__init__()
        self.mhsa = MultiheadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = GELU()
        self.prenorm = prenorm

    def _forward_prenorm(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        out = self.norm1(src)
        q = k = _with_pos_encoding(out, pos)
        out = self.mhsa(q, k, out, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(out)
        out = self.norm2(src)
        out = self.linear2(self.dropout2(self.activation(self.linear1(out))))
        src = src + self.dropout3(out)
        return src

    def _forward_postnorm(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        q = k = _with_pos_encoding(src, pos)
        out = self.mhsa(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(out)
        src = self.norm1(src)
        out = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(out)
        src = self.norm2(src)
        return src

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        _forward_impl = self._forward_prenorm if self.prenorm else self._forward_postnorm
        return _forward_impl(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(Module):

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0., prenorm: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(TransformerDecoderLayer, self).__init__()
        self.mhsa = MultiheadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.mha  = MultiheadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.activation = GELU()
        self.prenorm = prenorm

    def _forward_prenorm(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pos_queries: Optional[Tensor] = None) -> Tensor:
        out = self.norm1(tgt)
        q = k = _with_pos_encoding(out, pos_queries)
        out = self.mhsa(q, k, out, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(out)
        out = self.norm2(tgt)
        out = self.mha(_with_pos_encoding(out, pos_queries), _with_pos_encoding(memory, pos), memory,
                       attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(out)
        out = self.norm3(tgt)
        out = self.linear2(self.dropout3(self.activation(self.linear1(out))))
        tgt = tgt + self.dropout4(out)
        return tgt

    def _forward_postnorm(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pos_queries: Optional[Tensor] = None) -> Tensor:
        q = k = _with_pos_encoding(tgt, pos_queries)
        out = self.mhsa(q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(out)
        tgt = self.norm1(tgt)
        out = self.mha(_with_pos_encoding(out, pos_queries), _with_pos_encoding(memory, pos), memory,
                       attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(out)
        tgt = self.norm2(tgt)
        out = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(out)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, pos_queries: Optional[Tensor] = None) -> Tensor:
        _forward_impl = self._forward_prenorm if self.prenorm else self._forward_postnorm
        return _forward_impl(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, pos_queries)


class HungarianMatcher(Module):

    def __init__(self, w_labels, w_bboxes, w_generalized_iou):
        super(HungarianMatcher, self).__init__()
        self.w_labels = w_labels
        self.w_bboxes = w_bboxes
        self.w_geniou = w_generalized_iou

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, N = outputs['labels'].shape[:2]

        outputs_labels = outputs['labels'].flatten(0, 1).softmax(-1)
        outputs_bboxes = outputs['bboxes'].flatten(0, 1)
        targets_labels = torch.cat([v['labels'] for v in targets])
        targets_bboxes = torch.cat([v['bboxes'] for v in targets])

        cost_labels = 1 - outputs_labels[:, targets_labels]
        cost_bboxes = torch.cdist(outputs_bboxes, targets_bboxes, p=1)
        cost_geniou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(outputs_bboxes), box_cxcywh_to_xyxy(targets_bboxes))
        cost_matrix = self.w_labels * cost_labels + self.w_bboxes * cost_bboxes + self.w_geniou * cost_geniou
        cost_matrix = cost_matrix.view(B, N, -1).detach()
        num_objects = [len(v['bboxes']) for v in targets]

        matching = [linear_sum_assignment(cost[idx]) for idx, cost in enumerate(cost_matrix.split(num_objects, -1))]
        return [(torch.as_tensor(i, dtype=torch.int32), torch.as_tensor(j, dtype=torch.int32)) for i, j in matching]
