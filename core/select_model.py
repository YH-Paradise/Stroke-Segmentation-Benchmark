import torch.nn as nn

from core.models.Unet3D import unet3d
from core.models.MobileViTbased import mobilevit_s
from core.models.unet_plus_plus import unet_pp
from monai.networks.nets import SwinUNETR


def model_configuration(model_name, device):
    if model_name == 'swin3d':
        model = SwinUNETR(
                img_size=(32, 256, 256),
                in_channels=1,
                out_channels=1,
                feature_size=48,
                use_checkpoint=True,
            ).to(device)

        model = nn.Sequential(model, nn.Sigmoid())

    elif model_name == 'unet3d':
        model = unet3d().to(device)

    elif model_name == 'm_vit_based':
        model = mobilevit_s().to(device)

    else:
        model = unet_pp().to(device)

    return model