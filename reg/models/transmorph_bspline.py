from .modules.swin_transformer import SwinTransformer
from .modules.conv_layers import Conv3dReLU, DecoderBlock
from .modules.transformation import CubicBSplineFFDTransform, warp

import math
import torch.nn as nn
import torch.nn.functional as nnf


def conv_nd(ndim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation
    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv = getattr(nn, f"Conv{ndim}d")(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding)
    nn.init.kaiming_uniform_(conv.weight, a=a)
    return conv


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = nnf.interpolate(x, scale_factor=scale_factor, size=size, mode=mode, )
    return y


class TransMorphBspline(nn.Module):
    def __init__(self, config):
        """
        TransMorph Model
        """
        super(TransMorphBspline, self).__init__()
        embed_dim = config.embed_dim

        ndim = len(config.img_size)
        img_size = config.img_size
        cps = config.cps
        resize_channels = config.resize_channels
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((im_size - 1) / c) + 1 + 2)
                                  for im_size, c in zip(img_size, cps)])

        self.transformer = SwinTransformer(
            img_size=config.img_size,
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

        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip

        self.up0 = DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4 if self.if_transskip else 0,
            use_batchnorm=False
        )

        self.up1 = DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2 if self.if_transskip else 0,
            use_batchnorm=False
        )  # 384, 20, 20, 64

        self.up2 = DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim if self.if_transskip else 0,
            use_batchnorm=False
        )  # 384, 40, 40, 64

        self.up3 = DecoderBlock(
            embed_dim,
            embed_dim // 2,
            skip_channels=embed_dim // 2 if self.if_convskip else 0,
            use_batchnorm=False
        )  # 384, 80, 80, 128

        self.c1 = Conv3dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                in_ch = embed_dim // 2
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(conv_nd(ndim, in_ch, out_ch, a=0.2), nn.LeakyReLU(0.2)))

        # final prediction layer
        self.out_layer = conv_nd(ndim, resize_channels[-1], ndim)
        self.transform = CubicBSplineFFDTransform(ndim=ndim, svf=True, cps=cps)

    def forward(self, x):
        # [batch, channel, W, H, L]
        source = x.clone()

        f1, f2, f3, f4, f5 = None, None, None, None, None

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)

        out_feats = self.transformer(x)

        x = out_feats[-1]
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]

        x = self.up0(x, f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)

        x = interpolate_(x, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        x = self.out_layer(x)
        flow, disp = self.transform(x)
        out = warp(source, disp)
        return out, flow, disp
