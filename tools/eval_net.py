# created by Iran R. Roman <iran@ccrma.stanford.edu>
from desseo.models import build_model
from desseo.datasets import golumbic_loader

import numpy as np
import torch
import scipy
import os
import math
from torchdiffeq import odeint
import matplotlib.pyplot as plt

def polar_plot(Rs, out_path):
    i = 0
    for d in Rs:
      ax = plt.subplot(111, polar=True)
      plt.plot(np.angle(d),np.abs(d),'o',markerfacecolor="None",markeredgecolor='blue',markersize=10)
      #print(i,np.angle(d))
      i += 1
    (markers, stemlines, baseline) = plt.stem(np.angle(np.mean(Rs)),np.abs(np.mean(Rs)))
    plt.setp(stemlines, linestyle="-", color="red", linewidth=2)
    plt.setp(markers, color="red",markersize=10,alpha=0.75)
    ax.set_rticks([0.5, 1])
    ax.set_yticklabels([])
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])
    plt.ylim([0,1])
    plt.savefig(os.path.join(out_path,f'circular'))
    print('theta (rad):\t','{:.3f}'.format(np.mean(np.angle(Rs))))
    print('theta (deg):\t','{:.3f}'.format(math.degrees(np.mean(np.angle(Rs)))))
    print('R:\t\t','{:.3f}'.format(np.mean(np.abs(Rs))))


def plot_outputs(
    t,
    pred_y,
    pred_f,
    inputs,
    masks,
    out_path='',
    figsize=(20,10)
):
    Rs = []
    for i in range(len(pred_y)):

        x = inputs[i][:-1][np.where(masks[i])[0]]
        y = pred_y[i][np.where(masks[i])[0]]
        f = pred_f[i][np.where(masks[i])[0]]
        t_i = t[np.where(masks[i])[0]]
        p = scipy.signal.find_peaks(x)[0]
        R = np.mean(np.exp(1j*np.angle(y[p])))
        Rs.append(R)
        plt.figure(figsize=figsize)
        plt.subplot(2,1,1)
        plt.title('R: {:.3f}'.format(np.abs(R)))
        plt.plot(t_i,x,label='input')
        plt.plot(t_i,np.real(y)/np.std(np.real(y)),label='DESSEO')
        plt.legend()
        #plt.xlim([0,t_i[-1]])
        plt.xlim([0,7.5])
        plt.ylabel('A.U.')
        plt.subplot(2,1,2)
        plt.plot(t_i,np.real(f),label='f_dot')
        plt.xlabel('time (s)')
        plt.ylabel('f (Hz)')
        plt.legend()
        #plt.xlim([0,t_i[-1]])
        plt.xlim([0,7.5])
        plt.savefig(os.path.join(out_path,f'{i}'))
        plt.close()

    polar_plot(Rs,out_path)

def evaluate(
    eval_loader,
    model,
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
    model.eval()

    with torch.no_grad():
        for cur_iter, (inputs, masks, targets) in enumerate(
            eval_loader
        ):
            inputs = inputs.cuda()
            targets = targets.cuda()
            masks = masks.cuda()

            dur = inputs.shape[1]/model.fs
            t = torch.linspace(0, dur, int(dur * model.fs)).cuda()

            # prepare model for input data
            model.x = inputs
            model.t = t
            init_conds = (torch.tensor(inputs.shape[0]*[[0.1+1j*0]]).cuda(), torch.tensor(inputs.shape[0]*[[model.f0]]).cuda())
            
            # model inference
            pred_y, pred_f = odeint(model, init_conds, t[:-1], method='rk4')
            pred_y = pred_y[...,0].t()
            pred_f = pred_f[...,0].t()

            # calculate the loss and optimize
            inputs_peaks = [scipy.signal.find_peaks(x.cpu().numpy())[0] for x in inputs]
            min_peaks = np.min([len(p) for p in inputs_peaks])
            top_indices = [np.argsort(inputs[ix,x].cpu().numpy())[::-1][:min_peaks] for ix, x in enumerate(inputs_peaks)]
            top_indices = torch.tensor(np.array([np.array(inputs_peaks[it])[t] for it, t in enumerate(top_indices)])).cuda()
            z_peaks = torch.gather(masks*pred_y,1,top_indices)
            circ_mean = torch.mean(torch.exp(1j*torch.angle(z_peaks)),axis=1)
            R = torch.abs(circ_mean)
            loss = -torch.log(torch.mean(R))

            print(' (val):\t',"{:.3f}".format(float(loss.detach())),'( R:',"{:.3f}".format(float(torch.mean(R).detach())),')')
            return float(loss.cpu()), pred_y.cpu().numpy(), pred_f.cpu().numpy(), inputs.cpu().numpy(), masks.cpu().numpy(), t.cpu().numpy()



def test(cfg):
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
    print('alpha\t',"{:.3f}".format(float(model.alpha.detach())))
    print('beta1\t',"{:.3f}".format(float(model.beta1.detach())))
    print('beta2\t',"{:.3f}".format(float(model.beta2.detach())))
    print('cs\t',"{:.3f}".format(float(model.cs.detach())))
    print('cr\t',"{:.3f}".format(float(model.cr.detach())))
    print('cw\t',"{:.3f}".format(float(model.cw.detach())))
    print('f0\t',"{:.3f}".format(float(model.f0.detach())),'\n')

    # Create the data loader.
    train_loader = golumbic_loader(cfg, "train")
    val_loader = golumbic_loader(cfg, "val")

    ### # set up writer for logging to Tensorboard format.
    ### writer = tb.TensorboardWriter(cfg)

    # Perform inference.
    loss, pred_y, pred_f, inputs, masks, t = evaluate(
        val_loader,
        model,
        cfg,
    )
    if np.isnan(loss):
        print('*************************************')
        print('*************************************')
        print('*************************************')
        print(f'nan loss found with evaluation data')
        print('*************************************')
        print('*************************************')
        print('*************************************')
    plot_outputs(t, pred_y, pred_f, inputs, masks.astype(int), cfg.OUT_PLOTS)
