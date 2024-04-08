import ml_collections

def get_transmorph_only_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.reg_head_chan = 16
    config.img_size = (256, 256, 192)
    return config

def get_swin_default_config():
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 6, 2)
    config.num_heads = (4,4,8,8) 
    config.window_size = (5,6,7)
    config.mlp_ratio = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.pat_merg_rf = 4
    return config

def get_swin_large_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 128 
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4,4,8,16) 
    return config 

def get_swin_small_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 48 
    config.depths = (2,2,4,2)
    config.num_heads = (4,4,4,4) 
    return config 

def get_swin_tiny_config():
    config = ml_collections.ConfigDict()
    config.update(get_swin_default_config())
    config.embed_dim = 6 
    config.depths = (2,2,4,2)
    config.num_heads = (2,2,4,4) 
    return config 

'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
rpe (bool): Enable relative position embedding. Default: True
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
'''
def get_TransMorph_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_only_config())
    config.update(get_swin_default_config())
    return config

def get_TransMorph_NoRelPosEmbd_config():
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.rpe = False
    return config

def get_TransMorph_SinPosEmbd_config():
    '''
    TransMorph with Sinusoidal Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.spe = True
    return config

def get_TransMorph_LrnPosEmbd_config():
    '''
    TransMorph with Learnable Positional Embedding
    '''
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.ape = True
    return config

def get_TransMorph_NoConvSkip_config():
    '''
    No skip connections from convolution layers

    Computational complexity:       577.34 GMac
    Number of parameters:           63.56 M
    '''
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.if_convskip = False
    return config

def get_TransMorph_NoTransSkip_config():
    '''
    No skip connections from Transformer blocks

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.if_transskip = False
    return config

def get_TransMorph_NoSkip_config():
    '''
    No skip connections

    Computational complexity:       639.93 GMac
    Number of parameters:           58.4 M
    '''
    config = ml_collections.ConfigDict()
    config.update(get_TransMorph_config())
    config.if_transskip = False
    config.if_convskip = False
    return config

def get_TransMorph_Large_config():
    '''
    A Large TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_only_config())
    config.update(get_swin_large_config())
    return config

def get_TransMorph_Small_config():
    '''
    A Small TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_only_config())
    config.update(get_swin_small_config())
    return config

def get_TransMorph_Tiny_config():
    '''
    A Tiny TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_only_config())
    config.update(get_swin_tiny_config())
    return config

CONFIGS = {
    'TransMorph-No-Conv-Skip': get_TransMorph_NoConvSkip_config(),
    'TransMorph-No-Trans-Skip': get_TransMorph_NoTransSkip_config(),
    'TransMorph-No-Skip': get_TransMorph_NoSkip_config(),
    'TransMorph-Lrn': get_TransMorph_LrnPosEmbd_config(),
    'TransMorph-Sin': get_TransMorph_SinPosEmbd_config(),
    'TransMorph-No-RelPosEmbed': get_TransMorph_NoRelPosEmbd_config(),

    'TransMorph': get_TransMorph_config(),
    'TransMorph-Large': get_TransMorph_Large_config(),
    'TransMorph-Small': get_TransMorph_Small_config(),
    'TransMorph-Tiny': get_TransMorph_Tiny_config(),
}
