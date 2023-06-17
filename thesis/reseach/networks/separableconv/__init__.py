import torch
from .version import __version__


__version__ += "+torch{}".format(torch.__version__)
del torch
