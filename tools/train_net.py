# created by Iran R. Roman <iran@ccrma.stanford.edu>
from desseo.models import build_model
from desseo.datasets import golumbic_loader

import numpy as np
import torch
from torchdiffeq import odeint

def train_epoch(
    train_loader,
    model,
    optimizer,
    cfg,
    writer=None,
):
    """
    Perform training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        cfg (CfgNode): configs. 
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    data_size = len(train_loader)

    for cur_iter, (inputs, masks, targets) in enumerate(
        train_loader
    ):
        inputs = inputs.cuda()
        targets = targets.cuda()
        masks = masks.cuda()

        dur = inputs.shape[1]/model.fs
        t = torch.linspace(0, dur, int(dur * model.fs)).cuda()

        model.x = inputs
        model.t = t
        init_conds = (torch.tensor(inputs.shape[0]*[[0.1+1j*0]]).cuda(), torch.tensor(inputs.shape[0]*[[model.f0]]).cuda())

        pred_y, pred_f = odeint(model, init_conds, t[:-1], method='rk4')
        pred_y = pred_y[...,0].t()
        pred_f = pred_f[...,0].t()


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

    # Build the model and print parameters.
    model = build_model(cfg)
    print('built DESSEO model with parameters:')
    print('alpha\t',float(model.alpha.detach()))
    print('beta1\t',float(model.beta1.detach()))
    print('beta2\t',float(model.beta2.detach()))
    print('cs\t',float(model.cs.detach()))
    print('cr\t',float(model.cr.detach()))
    print('cw\t',float(model.cw.detach()))
    print('f0\t',float(model.f0.detach()))

    # Construct the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    # Create the video train and val loaders.
    train_loader = golumbic_loader(cfg, "train")
    val_loader = golumbic_loader(cfg, "val")

    ### # set up writer for logging to Tensorboard format.
    ### writer = tb.TensorboardWriter(cfg)

    # Perform the training loop.
    patience = 0
    while patience < cfg.SOLVER.PATIENCE_LIM:

        # Train for one epoch.
        train_epoch(
            train_loader,
            model,
            optimizer,
            cfg,
        )
        eval_epoch(
            val_loader,
            model,
            cfg,
            train_loader,
        )
