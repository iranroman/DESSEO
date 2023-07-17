# created by Iran R. Roman <iran@ccrma.stanford.edu>
from desseo.models import build_model

import numpy as np
import torch

def train(cfg):
    """
    Train a DESSEO model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode):
    """
    # Set random seed from configs.
    if cfg.NUMPY_SEED:
        np.random.seed(cfg.NUMPY_SEED)
    if cfg.TORCH_SEED:
        torch.manual_seed(cfg.TORCH_SEED)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    print('built DESSEO model with parameters:')
    print('alpha\t',float(model.alpha.detach()))
    print('beta1\t',float(model.beta1.detach()))
    print('beta2\t',float(model.beta2.detach()))
    print('cs\t',float(model.cs.detach()))
    print('cr\t',float(model.cr.detach()))
    print('cw\t',float(model.cw.detach()))
    print('f0\t',float(model.f0.detach()))
