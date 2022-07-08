from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image

import copy
import torch
import torchvision.transforms.functional as F

from frostml.vision.utils.ops import box_xyxy_to_cxcywh


def get_size_with_aspect_ratio(image_size, size: int, max_size: int = None):
    w, h = image_size
    if max_size is not None:
        size = min(size, max_size)
    l = float(max(w, h))
    h = int(h / l * size)
    w = int(w / l * size)
    return h, w


def get_size(image_size, size: Union[int, Tuple], max_size: Union[int, Tuple] = None):
    return size if isinstance(size, (list, tuple)) else get_size_with_aspect_ratio(image_size, size, max_size)


def resize(image: Image, target: Optional[Dict[Any, Any]], size: Union[int, Tuple], max_size: Union[int, Tuple] = None):
    # size can be a size of longer edge (scalar) or a spatial resolution h x w (tuple)
    rescale_factor = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, rescale_factor)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(rescaled) / float(original) for rescaled, original in zip(rescaled_image.size, image.size))
    ratio_w, ratio_h = ratios

    target = copy.deepcopy(target)

    if 'bboxes' in target:
        bboxes = target['bboxes']
        rescaled_bboxes = bboxes * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        target['bboxes'] = rescaled_bboxes

    if 'area' in target:
        area = target['area']
        rescaled_area = area * (ratio_w * ratio_h)
        target['area'] = rescaled_area

    h, w = rescale_factor
    target['size'] = torch.tensor([h, w])

    return rescaled_image, target


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:

    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = copy.deepcopy(target)
        h, w = image.shape[-2:]
        if 'bboxes' in target:
            bboxes = target['bboxes']
            bboxes = box_xyxy_to_cxcywh(bboxes)
            bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target['bboxes'] = bboxes
        return image, target


class Resize:

    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image, target):
        return resize(image, target, self.size, self.max_size)
