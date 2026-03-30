import torch.nn as nn
import torch
import math


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   # nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   # nn.BatchNorm3d(out_channels),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.dconv(x)
        return x



class down_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
                                  double_conv(in_channels, out_channels))

    def forward(self, x):
        x = self.down(x)
        return x


class up_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, small, big):
        small = self.up(small)

        if big.size()[2] != small.size()[2]:
            diff = big.size()[2] - small.size()[2]
            half_diff = int(diff / 2)
            big = big[:, :, half_diff:(big.size()[2] - diff), half_diff:(big.size()[3] - diff)]

        x = torch.cat([big, small], dim=1)
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, model_info, init_weights=True):
        super().__init__()

        stem = []
        for i, layer_info in enumerate(model_info["in_conv"]):
            in_channels = layer_info[0]
            out_channels = layer_info[1]
            stem.append(double_conv(in_channels, out_channels))

        self.stem = nn.Sequential(*stem)

        encoder = []
        for i, layer_info in enumerate(model_info["encoder"]):
            in_channels = layer_info[0]
            out_channels = layer_info[1]
            encoder.append(down_op(in_channels, out_channels))

        self.encoder = nn.ModuleList([*encoder])

        decoder = []
        for i, layer_info in enumerate(model_info["decoder"]):
            in_channels = layer_info[0]
            out_channels = layer_info[1]
            decoder.append(up_op(in_channels, out_channels))

        self.decoder = nn.ModuleList([*decoder])

        self.last = nn.Conv3d(32, 1, 1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x1 = self.stem(x)
        x2 = self.encoder[0](x1)
        x3 = self.encoder[1](x2)
        x4 = self.encoder[2](x3)
        x = self.encoder[3](x4)

        x = self.decoder[0](x, x4)
        x = self.decoder[1](x, x3)
        x = self.decoder[2](x, x2)
        x = self.decoder[3](x, x1)
        x = self.last(x)
        return torch.sigmoid(x)


def unet3d():
    model_info = {
        "in_conv": [
            [2, 32],
        ],
        "encoder": [
            [32, 64],
            [64, 128],
            [128, 256],
            [256, 512],
        ],
        "decoder": [
            [512, 256],
            [256, 128],
            [128, 64],
            [64, 32],
        ]
    }

    return Unet3D(model_info)
