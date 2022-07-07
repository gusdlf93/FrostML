from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

import os.path
import torch

from torchvision.datasets import VisionDataset

torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['COCODetection', 'coco_collate_fn']


def _make_spatial_shape_to_square(tensor_shapes):
    shape = tensor_shapes[0]
    long = max(shape)
    shape = [shape[0], long, long]
    return shape


def _align_spatial_shape_with_paddings(tensors):
    tensor_shape = _make_spatial_shape_to_square([list(tensor.shape) for tensor in tensors])
    batch_shape = [len(tensors)] + tensor_shape
    padded_tensors = torch.zeros(batch_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    for tensor, padded_tensor in zip(tensors, padded_tensors):
        padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
    return padded_tensors


def coco_collate_fn(batch):
    inputs, targets = list(zip(*batch))
    inputs = _align_spatial_shape_with_paddings(inputs)
    return inputs, targets


class COCODetection(VisionDataset):

    def __init__(self, root: str, annFile: str, transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super(COCODetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = sorted([i for i in self.coco.imgs.keys() if len(self.coco.getAnnIds(i, iscrowd=False)) != 0])

    def __load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self.__load_image(id)
        annotations = self.__load_target(id)

        labels = [annotation['category_id'] for annotation in annotations]
        labels = torch.tensor(labels, dtype=torch.float32)

        bboxes = [annotation['bbox'] for annotation in annotations]
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        bboxes[:, 2:] += bboxes[:, :2]
        bboxes[:, 0::2].clamp_(min=0, max=image.size[0])
        bboxes[:, 1::2].clamp_(min=0, max=image.size[1])

        cond = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])
        target = {
            'image_id': torch.tensor([id]),
            'labels': labels[cond],
            'bboxes': bboxes[cond],
            'area': torch.tensor([annotation['area'] for annotation in annotations])[cond],
            'iscrowd': torch.tensor([annotation['iscrowd'] for annotation in annotations])[cond],
            'size': torch.as_tensor([int(image.size[1]), int(image.size[0])]),
            'orig_size': torch.as_tensor([int(image.size[1]), int(image.size[0])])
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)
