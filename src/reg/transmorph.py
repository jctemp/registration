from reg.spatial_transformer import SpatialTransformer
from reg.swin_transformer_tm import SwinTransformer

from torch.distributions.normal import Normal
import torch
import torch.nn as nn


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        nm =  nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class TransMorph(nn.Module):
    def __init__(self, config):
        """
        TransMorph Model
        """
        super(TransMorph, self).__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )

        self.up0 = DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4 if self.if_transskip else 0,
            use_batchnorm=False,
        )

        self.up1 = DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2 if self.if_transskip else 0,
            use_batchnorm=False,
        )  # 384, 20, 20, 64

        self.up2 = DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim if self.if_transskip else 0,
            use_batchnorm=False,
        )  # 384, 40, 40, 64

        self.up3 = DecoderBlock(
            embed_dim,
            embed_dim // 2,
            skip_channels=embed_dim // 2 if self.if_convskip else 0,
            use_batchnorm=False,
        )  # 384, 80, 80, 128

        self.up4 = DecoderBlock(
            embed_dim // 2,
            config.reg_head_chan,
            skip_channels=(
                config.reg_head_chan if self.if_convskip else 0 if self.if_convskip else 0
            ),
            use_batchnorm=False,
        )  # 384, 160, 160, 256

        self.c1 = Conv3dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)

        self.c2 = Conv3dReLU(config.in_chans, config.reg_head_chan, 3, 1, use_batchnorm=False)
        # in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True,

        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )

        self.stn = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        # (B, C, W, H, t) e.g. [1, 1, 256, 256, 192]
        # source = x[:, 0:1, :, :]
        source = x.clone()

        f1 = None
        f2 = None
        f3 = None
        f4 = None
        f5 = None

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)

        flow = self.reg_head(x)
        out = self.stn(source, flow)

        return out, flow
