
import random
import numpy as np
import torch
from torch.backends import cudnn
def set_seed(seed=42,deterministic=True):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True

    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
