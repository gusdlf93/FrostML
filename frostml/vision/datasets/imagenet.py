from typing import Callable, Optional

import os.path

from torchvision.datasets import ImageFolder


__all__ = ['ImageNet']


class ImageNet(ImageFolder):

    def __init__(self, root: str, train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        root = os.path.join(root, 'train') if train else os.path.join(root, 'val')
        super(ImageNet, self).__init__(root, transform, target_transform)
