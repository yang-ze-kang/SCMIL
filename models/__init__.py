from .TransMIL import TransMIL
from .AMIL import AMIL
from .DSMIL import DSMILSurvival
from .CLAM import CLAM_SB, CLAM_SB_Survival
from .MDN import SurvivalMDN
from .SCMIL import SCMIL

backbone_cls = {
    'AMIL':AMIL,
    'DSMIL':DSMILSurvival,
    'TransMIL':TransMIL,
    'CLAM-SB':CLAM_SB_Survival,
    "SCMIL":SCMIL
}

def create_WSI_model(cfg):
    if cfg.model.model_name == 'SurvivalMDN':
        cfg.model.as_backbone = True
        cfg.model.backbone = backbone_cls[cfg.model.backbone_name](**dict(cfg.model))
        model = SurvivalMDN(**dict(cfg.model))
    else:
        raise NotImplementedError
    return model