import math
import torch
from torch import nn
from einops import rearrange


def conv1x1_bn(in_channels, out_channels):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
                         nn.GroupNorm(1, out_channels),
                         nn.SiLU())


def basic_conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.GroupNorm(1, out_channels),
                         nn.SiLU())


class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   # nn.BatchNorm3d(out_channels),
                                   nn.SiLU(inplace=True),
                                   nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   # nn.BatchNorm3d(out_channels),
                                   nn.SiLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.dconv(x)
        return x


class up_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.CT_up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # in_dim = int(in_channels / 3)
        in_dim = int(in_channels / 2)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, in_dim, kernel_size=3, padding=1),
                                  nn.GroupNorm(1, in_dim),
                                  nn.SiLU(inplace=True),
                                  nn.Conv3d(in_dim, out_channels, kernel_size=3, padding=1),
                                  nn.GroupNorm(1, out_channels),
                                  nn.SiLU(inplace=True)
                                  )

        self.US_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

    def forward(self, small, big):
        small = self.US_up(small)
        x = torch.cat([big, small], dim=1)
        return self.conv(x)


class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=2):
        super().__init__()

        expand_channels = in_channels * expand_ratio
        self.conv = nn.Sequential(conv1x1_bn(in_channels, expand_channels),
                                  basic_conv_bn(expand_channels, expand_channels, stride=stride, padding=1),
                                  nn.Conv3d(expand_channels, out_channels, 1, 1, bias=False),
                                  nn.GroupNorm(1, out_channels))

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if identity.shape[1] != x.shape[1] or identity.shape[2] != x.shape[2]:
            return x
        else:
            x = x + identity
            return x


class LayerNorm_w_func(nn.Module):
    def __init__(self, channels, func):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
        self.func = func

    def forward(self, x, **kwargs):
        return self.func(self.layernorm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNorm_w_func(dim, Attention(dim, heads, dim_head, dropout)),
                LayerNorm_w_func(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ps, self.ph, self.pw = patch_size

        self.conv1 = basic_conv_bn(channel, channel, kernel_size)
        self.conv2 = conv1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv1x1_bn(dim, channel)
        self.conv4 = basic_conv_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, t, s, h, w = x.shape
        x = rearrange(x, 'b t (s ps) (h ph) (w pw) -> b (ps ph pw) (s h w) t', ph=self.ph, pw=self.pw, ps=self.ps)
        x = self.transformer(x)
        x = rearrange(x, 'b (ps ph pw) (s h w) t -> b t (s ps) (h ph) (w pw)',
                      s=s // self.ps, h=h // self.ph, w=w // self.pw, ps=self.ps, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, dims, model_info, init_weights=True):
        super().__init__()

        depth = [2, 4, 3]

        encoder = []
        encoder.append(basic_conv_bn(2, 16, 3, (1, 2, 2), 1))  # 1x1x52x256x256 -> 1x16x52x128x128 $$$
        encoder.append(MV2Block(16, 32, 1))  # 1x16x52x128x128 -> 1x32x52x128x128

        encoder.append(MV2Block(32, 64, (1, 2, 2)))  # 1x32x52x128x128 -> 1x48x52x64x64 $$$
        encoder.append(MV2Block(64, 64, 1))  # 1x48x52x64x64 -> 1x48x52x64x64

        encoder.append(MV2Block(64, 96, (1, 2, 2)))  # 1x48x52x64x64 -> 1x64x52x32x32 $$$
        encoder.append(MobileViTBlock(dims[0], depth[0], 96, 3, (2, 2, 2), int(dims[0] * 2)))  # 토큰이 그럼 27x16x16개?

        encoder.append(MV2Block(96, 128, (1, 2, 2)))  # 1x64x52x32x32 -> 1x80x52x16x16 $$$
        encoder.append(MobileViTBlock(dims[1], depth[1], 128, 3, (2, 2, 2), int(dims[1] * 4)))  # 27x8x8

        encoder.append(MV2Block(128, 160, (1, 2, 2)))  # 1x80x52x16x16 -> 1x96x52x8x8
        encoder.append(MobileViTBlock(dims[2], depth[2], 160, 3, (2, 2, 2), int(dims[2] * 4)))  # 27x4x4

        self.encoder = nn.ModuleList([*encoder])

        # self.prepare_expand = basic_conv_bn(160, 256)
        self.prepare_expand = basic_conv_bn(160, 128)

        decode_layer = []
        for i, layer_info in enumerate(model_info["decoder"]):
            in_channels = layer_info[0]
            out_channels = layer_info[1]
            decode_layer.append(up_op(in_channels, out_channels))
        self.decoder = nn.ModuleList([*decode_layer])

        self.extra_expand = nn.ConvTranspose3d(16, 4, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))

        self.last = nn.Conv3d(4, 1, 1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.encoder[0](x)
        x = self.encoder[1](x1)
        x2 = self.encoder[2](x)
        x = self.encoder[3](x2)
        x3 = self.encoder[4](x)
        x = self.encoder[5](x3)
        x4 = self.encoder[6](x)
        x = self.encoder[7](x4)
        x = self.encoder[8](x)
        x = self.encoder[9](x)

        x = self.prepare_expand(x)

        x = self.decoder[0](x, x4)
        x = self.decoder[1](x, x3)
        x = self.decoder[2](x, x2)
        x = self.decoder[3](x, x1)

        x = self.extra_expand(x)
        result = self.last(x)

        return torch.sigmoid(result)


def mobilevit_s():
    dims = [144, 192, 240]
    model_info = {
        "decoder": [[256, 96], [192, 64], [128, 32], [48, 16]]
    }
    return MobileViT(dims, model_info, init_weights=True)
