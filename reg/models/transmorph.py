from modules.swin_transformer import SwinTransformer
from modules.spatial_transformer import SpatialTransformer, SpatialTransformerSeries
from modules.conv_layers import Conv3dReLU, DecoderBlock, RegistrationHead

import torch.nn as nn


class TransMorph(nn.Module):
    def __init__(self, config):
        """
        TransMorph Model
        """
        super(TransMorph, self).__init__()
        embed_dim = config.embed_dim

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

        self.up4 = DecoderBlock(
            embed_dim // 2,
            config.reg_head_chan,
            skip_channels=config.reg_head_chan if self.if_convskip else 0,
            use_batchnorm=False
        )  # 384, 160, 160, 256

        self.c1 = Conv3dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(config.in_chans, config.reg_head_chan, 3, 1, use_batchnorm=False)

        self.reg_head3d = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )

        self.reg_head2d = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=2,
            kernel_size=3,
        )

        self.spatial_series_trans = SpatialTransformerSeries(config.img_size)
        self.spatial_trans = SpatialTransformer(config.img_size)

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        self.series_reg = config.series_reg

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
        x = self.up4(x, f5)

        if self.series_reg:
            flow = self.reg_head2d(x)
            out = self.spatial_series_trans(source, flow)
        else:
            flow = self.reg_head3d(x)
            out = self.spatial_trans(source, flow)

        return out, flow
