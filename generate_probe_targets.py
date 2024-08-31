import torch
import time
import sys
import os
import pandas as pd
import numpy as np
from math import factorial
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error
from analyze_models import get_model_hs_df, get_df_models
from util import get_data
from tqdm import tqdm

def generate_lr_targets(datatype = 'linreg1', traintest = 'train'):
    # Makes the targets for the linreg model, which are w, wx
    # ONLY WORKS FOR 1D W
    criterion = torch.nn.MSELoss()
    w, sequences = get_data(datatype, traintest)
    X, y = sequences[:,:-1], sequences[:,1:]
    X, y = X.squeeze(-1), y.squeeze(-1)
    w = w.repeat(X.shape[1], 1).T
    targets = {}
    for i in range(1, 6):
        targets[f'lr_w{i}'] = w**i
        targets[f'lr_w{i}x'] = targets[f'lr_w{i}'] * X
        targets[f'lr_w{i}x{i}'] = targets[f'lr_w{i}'] * X**i
    
    x = X[:,0::2]
    y = y[:,0::2]
    w = w[:,0::2]
    # Make w of the same size as X
    ypred = w*x
    loss = criterion(ypred, y)
    print(f'prediction loss: {loss}')
    save_probetargets(targets, 'lr_targets.pth', datatype, traintest)
    return targets


def generate_lr_cca_targets(datatype = 'linreg1cca', traintest = 'train', maxdeg = 5, save = True):
    w, sequences = get_data(datatype, traintest)
    X, y = sequences[:,:-1], sequences[:,1:]
    X, y = X.squeeze(-1), y.squeeze(-1)
    w = w.repeat(X.shape[1], 1).T
    wpow = torch.zeros((w.shape[0], w.shape[1], maxdeg))
    wpowx = torch.zeros(wpow.shape)
    wxpow = torch.zeros(wpow.shape)
    for deg in range(1, maxdeg+1):
        wpow[:, :, deg - 1] = w**deg
        wpowx[:, :, deg - 1] = wpow[:, :, deg - 1] * X
        wxpow[:, :, deg - 1] = wpow[:, :, deg - 1] * X**deg
    targets = {}
    targets['lr_wpow'] = wpow
    targets['lr_wpowx'] = wpowx
    targets['lr_wxpow'] = wxpow
    if save:
        save_probetargets(targets, f'lr_cca_targets_deg{maxdeg}.pth', datatype, traintest)
    return targets


def generate_reverselr_targets(datatype = 'rlinreg1', traintest = 'train'):
    ccatargets = generate_lr_cca_targets(datatype, traintest, save = False)
    rlr_targets = {}
    rlr_targets['rlr_wi2'] = ccatargets['lr_wpow'][:, :, :2]
    # rlr_targets['rlr_wi1'] = ccatargets['lr_wpow'][:, :, :1]
    # rlr_targets['rlr_wix2'] = ccatargets['lr_wpowx'][:, :, :2]
    save_probetargets(rlr_targets, 'rlr_targets.pth', datatype, traintest)


def save_probetargets(targets, fname, datatype, traintest):
    bigdir = 'probe_targets'
    dir = f'{datatype}_{traintest}'
    if dir not in os.listdir(bigdir):
        os.mkdir(f'{bigdir}/{dir}')
    torch.save(targets, f'{bigdir}/{dir}/{fname}')

if __name__ == '__main__':
    for datatype in ['overdamped']:
        for traintest in ['train', 'test']:
            for reverse in [True, False]:
                print(f'Generating targets for {datatype} {traintest}')
                generate_exp_targets(datatype, traintest, maxdeg = 5, reverse = reverse)
                generate_rk_targets(datatype, traintest, maxdeg = 5, reverse = reverse)
                if reverse:
                    generate_rkexp_targets_REVERSE(datatype, traintest, maxdeg = 5)
