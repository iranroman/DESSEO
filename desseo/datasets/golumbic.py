# created by Iran R. Roman <iran@ccrma.stanford.edu>
import torch
import os
import numpy as np
import scipy
from scipy.signal import butter, hilbert, filtfilt

def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

class golumbic_data(torch.utils.data.Dataset):

    def __init__(self, cfg, split):
        assert cfg.DATASET.NAME == 'Golumbic'
        self.fs = cfg.DESSEO.FS
        self.path_to_files = cfg.DATASET.PATH
        self.split = split
        self.filenames = self.get_filenames(self.split,cfg.DATASET.PERCENT_VAL)
        self.npoints = len(self.filenames)
        self.diff = cfg.DATASET.DIFF
        self.standardize = cfg.DATASET.STANDARDIZE
        self.envelopes, self.masks = self.prepare_data(cfg.DATASET.LOWPASS_CUTOFF)

    def get_filenames(self, split, perc_val):
        all_files = os.listdir(self.path_to_files)
        all_env = [f for f in all_files if f[-1]=='t' and f[0] in ['1','2']]
        all_env.sort()
        nenv = len(all_env)
        rand_idx = np.random.choice(nenv,nenv,replace=False)
        if split == 'train':
            rand_idx = rand_idx[int(nenv*perc_val/100):]
        elif split == 'val':
            rand_idx = rand_idx[-int(nenv*perc_val/100):]
        return [all_env[i] for i in rand_idx]

    def load_mat_files(self):
        data = [scipy.io.loadmat(os.path.join(self.path_to_files,f)) for f in self.filenames]
        all_envelopes = [np.squeeze(f['env_ds']) for f in data]
        return all_envelopes

    def filter_envs(self,all_envelopes,cutoff):
        return [butter_lowpass_filter(np.squeeze(np.array(env)),cutoff,self.fs) for env in all_envelopes]

    def pad_and_get_masks(self, all_envelopes):
        max_len = len(max(all_envelopes,key=len))+1
        all_envelopes = np.array([np.pad(env,(0,max_len-len(env))) for env in all_envelopes])
        all_masks = np.ma.masked_where(all_envelopes[:,1:] != 0.0, all_envelopes[:,1:]).mask.astype(np.float32)
        return all_envelopes, all_masks

    def prepare_data(self, lowpass_cutoff):
        all_envelopes = self.load_mat_files()
        all_envelopes = self.filter_envs(all_envelopes,lowpass_cutoff)
        if self.diff:
            all_envelopes = [np.diff(env) for env in all_envelopes]
        if self.standardize:
            all_envelopes = [env-np.mean(env) for env in all_envelopes]
            all_envelopes = [env/np.std(env) for env in all_envelopes]
        envelopes, masks = self.pad_and_get_masks(all_envelopes)
        return envelopes, masks

    def __len__(self):
        return self.npoints

    def __getitem__(self, idx):
       return self.envelopes[idx], self.masks[idx], self.envelopes[idx][1:] 

def golumbic_loader(cfg, split):

    dataset = golumbic_data(cfg, split)

    return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.npoints,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        )
