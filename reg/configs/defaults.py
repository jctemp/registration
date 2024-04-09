import ml_collections


def get_transmorph_only_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.reg_head_chan = 16
    config.img_size = (256, 256, 192)
    config.series_reg = False
    return config


def get_swin_default_config():
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.qkv_bias = False
    config.qk_scale = None
    config.drop_rate = 0
    config.attn_drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.out_indices = (0, 1, 2, 3)
    config.patch_norm = True
    config.use_checkpoint = False
    config.pat_merg_rf = 4
    config.mc_drop = 0.15
    config.use_mc = False
    return config


def get_swin_large_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 128
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4, 4, 8, 16)
    return config


def get_swin_small_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 48
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 4, 4)
    return config


def get_swin_tiny_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 6
    config.depths = (2, 2, 4, 2)
    config.num_heads = (2, 2, 4, 4)
    return config
