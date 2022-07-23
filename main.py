import torch
from segmentation_model.backbone import *

# 'EfficientNet'
# 'ReResSegNet'
# 'EfficientFormer'
# 'volo'
# 'MobileNet'
# 'MAE_ViT'
# 'Wide_ResNet'
# 'UNet'
# 'resnet_18'
# 'resnet_50'

net = load_model("EfficientNet", num_classes=1000, img_size=704, pretrained=False)

img = torch.randn((2, 3, 704, 704))
output = net(img)

print(f"debug")