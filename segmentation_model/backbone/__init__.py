from segmentation_model.backbone import EfficientFormer
from segmentation_model.backbone import volo
from segmentation_model.backbone import MAE_ViT
from segmentation_model.backbone import unet
from segmentation_model.backbone import ReResSegNet
from segmentation_model.backbone import MobileNetV2
from segmentation_model.backbone import resnet_18
from segmentation_model.backbone import resnet_50
from segmentation_model.backbone import Wide_ResNet
from segmentation_model.backbone import EfficientNet

def load_model(model_name, num_classes=1000, img_size=704, pretrained=False):
    models = {'EfficientNet' : EfficientNet.EfficientNet,
              'ReResSegNet' : ReResSegNet.ReResSegNet,
              'EfficientFormer' : EfficientFormer.efficientformer_l3,
              'volo' : volo.volo_d1,
              'MobileNet' : MobileNetV2.MobileNetV2,
              'MAE_ViT' : MAE_ViT.MAE_VIT_B16,
              'Wide_ResNet' : Wide_ResNet.Wide_Resnet,
              'UNet' : unet.UNet,
              'resnet_18' : resnet_18.ResNet18,
              'resnet_50' : resnet_50.ResNet50
              }

    try:
        if model_name == "EfficientNet":
            model = models[model_name](num_classes=1000, img_size=704, pretrained=False)
        else:
            model = models[model_name](num_classes=1000, img_size=704)
    except:
        raise("Model load Error")

    return model
