from typing import Optional

import warnings
import random
import numpy
import torch
import torch.cuda
import torch.backends.cudnn as cudnn

from frostml.utils.tracker import *


def enable_reproducibility(seed: Optional[int] = None, distributed: bool = False) -> None:
    if seed:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if distributed:
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to enable reproducibility. '
                      'This will turn on the CUDNN deterministic setting, which can extremely slow down training! '
                      'Furthermore, you may see unexpected behavior when restarting from the checkpoint.')
    else:
        cudnn.benchmark = True
