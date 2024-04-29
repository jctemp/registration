from .transmorph import CONFIGS as CONFIG_DEFAULT
from .transmorph_bayes import CONFIGS as CONFIG_BAYES
from .transmorph_bspline import CONFIGS as CONFIG_BSPLINE

CONFIG_TM = {}
CONFIG_TM.update(CONFIG_DEFAULT)
CONFIG_TM.update(CONFIG_BAYES)
