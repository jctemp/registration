from .defaults import *


def get_transmorph_config():
    config = ml_collections.ConfigDict()
    config.update(get_transmorph_only_config())
    config.update(get_swin_default_config())
    return config


# ======================================================================================================================
# CONFIGURATIONS


CONFIGS = {
    "transmorph-identity": get_transmorph_config(),
}
