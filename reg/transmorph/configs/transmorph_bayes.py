from .defaults import *


def get_transmorph_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_bayes_config())
    config.update(get_swin_default_config())
    return config


# ======================================================================================================================
# MODEL CAPACITY


def get_transmorph_tiny_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_bayes_config())
    config.update(get_swin_tiny_config())
    return config


def get_transmorph_small_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_bayes_config())
    config.update(get_swin_small_config())
    return config


def get_transmorph_large_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_bayes_config())
    config.update(get_swin_large_config())
    return config


# ======================================================================================================================
# MODEL EMBEDDINGS


def get_transmorph_no_rel_pos_embedding_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.rpe = False
    return config


def get_transmorph_sin_pos_embedding_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.spe = True
    return config


def get_transmorph_lrn_pos_embedding_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.ape = True
    return config


# ======================================================================================================================
# MODEL SKIP-CONNECTIONS


def get_transmorph_no_conv_skip_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.if_convskip = False
    return config


def get_transmorph_no_trans_skip_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.if_transskip = False
    return config


def get_transmorph_no_skip_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_config())
    config.if_transskip = False
    config.if_convskip = False
    return config


# ======================================================================================================================
# CONFIGURATIONS


CONFIGS = {
    "transmorph-bayes": get_transmorph_config(),
    "transmorph-bayes-tiny": get_transmorph_tiny_config(),
    "transmorph-bayes-small": get_transmorph_small_config(),
    "transmorph-bayes-large": get_transmorph_large_config(),
    "transmorph-bayes-no-conv-skip": get_transmorph_no_conv_skip_config(),
    "transmorph-bayes-no-trans-skip": get_transmorph_no_trans_skip_config(),
    "transmorph-bayes-no-skip": get_transmorph_no_skip_config(),
    "transmorph-bayes-lrn_pos_embedding": get_transmorph_lrn_pos_embedding_config(),
    "transmorph-bayes-sin_pos_embedding": get_transmorph_sin_pos_embedding_config(),
    "transmorph-bayes-no_rel_pos_embedding": get_transmorph_no_rel_pos_embedding_config(),
}
