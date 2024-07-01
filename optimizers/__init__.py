from .adamp import AdamP
from .adamw import AdamW
from .adafactor import Adafactor
from .adahessian import Adahessian
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

from .optim_factory import create_optimizer
from torch.optim import lr_scheduler


def create_scheduler(cfg,optimizer):
    if cfg == None:
        return None
    elif cfg.type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T, eta_min=1e-9)
        return scheduler
    elif cfg.type == 'CosineAnnealingLR':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
        return scheduler
    else:
        raise NotImplementedError