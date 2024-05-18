from .transmorph import CONFIGS as CONFIG_DEFAULT
from .transmorph_bayes import CONFIGS as CONFIG_BAYES
from .transmorph_bspline import CONFIGS as CONFIG_BSPLINE
from .transmorph_identity import CONFIGS as CONFIG_IDENTITY

CONFIG_TM = {}
CONFIG_TM.update(CONFIG_DEFAULT)
CONFIG_TM.update(CONFIG_BAYES)
CONFIG_TM.update(CONFIG_IDENTITY)
