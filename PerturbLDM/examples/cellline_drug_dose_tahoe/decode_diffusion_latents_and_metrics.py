"""Decode diffusion-generated Tahoe latents and compute condition-level metrics."""

import os
import time
import argparse
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="CUDA device(s) to use, e.g., 0 or '0,1'")
parser.add_argument("--diff_dir", required=True, help="diff_dir")
parser.add_argument("--data_root", required=True,
                    help="Processed Tahoe dataset directory containing collection/ and processed/")
parser.add_argument("--save_cf_expr", default=False, action=argparse.BooleanOptionalAction,
                    help="Save per-cell counterfactual expression arrays. Disabled by default because all-condition Tahoe outputs are very large.")


args = parser.parse_args()

diff_dir = args.diff_dir
if not os.path.isdir(diff_dir):
    raise ValueError(f'Cannot find diff_dir: {diff_dir}')

save_dir = diff_dir
print(diff_dir)

save_cf_expr = args.save_cf_expr
print('save_cf_expr: ', save_cf_expr)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print('cuda ', os.environ['CUDA_VISIBLE_DEVICES'])




import pickle
import pandas as pd
import numpy as np
import torch
torch.cuda.is_available()

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler
import time
import json
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, List
import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


print(args.data_root)

datadir = args.data_root
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")

cond2id = json.load(open(os.path.join(processed_datadir, "cond2id.json"), 'r'))
id2cond = {v:k for k,v in cond2id.items()}



import json
import torch
import random
from collections import defaultdict

diff_inference_latent = torch.load(os.path.join(diff_dir, 'test_latent_outputs.pt'))
temppath = os.path.join(diff_dir, 'test_allcondnames.json')
diff_inference_condnames = json.load(open(temppath, 'r'))
print('diff inference latent: ', len(diff_inference_latent), len(diff_inference_condnames))

from util_diff import *

config_train_dict = json.load(open(os.path.join(diff_dir, 'config_train.json'), 'r'))
config_dict = json.load(open(os.path.join(diff_dir, 'config.json'), 'r'))
latent_dir = config_train_dict['latent_dir']
latent_dim = config_dict['latent_dim']
temppath = os.path.join(latent_dir, "condname2gtmean_rcmean.pkl")
condname2gtmean_rcmean = pickle.load(open(temppath, 'rb'))

print(config_train_dict)
print(config_dict)

clip_sample_range = config_train_dict['clip_sample_range']
print(clip_sample_range)
prediction_type = config_train_dict.get("prediction_type", "v_prediction")
PREDICTION_TYPE = prediction_type
print("prediction_type: ", PREDICTION_TYPE)

rescale_latent = config_train_dict.get("rescale_latent", False)
print("rescale_latent: ", rescale_latent)

train_std_item = config_train_dict.get("train_std_item")
print("train_std_item: ", train_std_item)


from LatentModelFinal import CellEncoderWithLogvar
latent_config = json.load(open(os.path.join(latent_dir, 'config.json'), 'r'))
print(latent_config)
latentmodel = CellEncoderWithLogvar(**{kk:vv for kk,vv in latent_config.items() if kk not in ['lr', 'weight_decay', 'batchsize', 'max_steps']})
state_dict = torch.load(os.path.join(latent_dir, 'model_weights.pth'))
latentmodel.load_state_dict(state_dict)
latentmodel.eval()
latentmodel.cuda()
print('latent: load finetuned weights.')





from util_inference_diff import *


from util_metrics import *
import time
start_time = time.time()


# diff_inference_latent, diff_inference_condnames

cond_result = {}
all_cf_expr_list = []
all_cf_cond_list = []
all_cf_expr_mean_list = []
for ii, condi in enumerate(diff_inference_condnames):
    latent_output = diff_inference_latent[ii]
    print(condi, latent_output.shape)
    if rescale_latent:
        latent_output = latent_output * train_std_item # rescale back. scale is 1/train_std_item
    with torch.no_grad():
        recon_mu_diff = latentmodel.decode(latent_output.cuda())['recon_mu'].cpu()
    recon_mu_diff_clamped = recon_mu_diff.clamp(0)
    if save_cf_expr:
        all_cf_expr_list.append(recon_mu_diff_clamped.numpy())
    all_cf_cond_list.append(condi)

    cf_mean = recon_mu_diff_clamped.mean(0)
    all_cf_expr_mean_list.append(cf_mean.numpy())

    gt_mean = condname2gtmean_rcmean[condi]['gt_mean']
    rc_mean = condname2gtmean_rcmean[condi]['rc_mean']
    mean_met_temp = compute_metrics_single(gt_mean, cf_mean)
    dict_output00 = {}
    dict_output00.update({kk+'_cf':vv for kk,vv in mean_met_temp.items()})    
    mean_met_temp = compute_metrics_single(gt_mean, rc_mean)
    dict_output00.update({kk+'_rc':vv for kk,vv in mean_met_temp.items()})    
    cond_result[condi] = dict_output00

if save_cf_expr:
    all_cf_exprs = np.stack(all_cf_expr_list, 0)
    print('all_cf_exprs shape: ', all_cf_exprs.shape) 
    temppath = os.path.join(save_dir, f'all_cf_exprs_diffU.npy')
    np.save(temppath, all_cf_exprs)
else:
    print('Not saving all_cf_exprs arrays. Just mean expressions will be saved.')

all_cf_exprs_mean = np.stack(all_cf_expr_mean_list, 0)
print('all_cf_exprs_mean shape: ', all_cf_exprs_mean.shape)
temppath = os.path.join(save_dir, f'all_cf_exprs_mean_diffU.npy')
np.save(temppath, all_cf_exprs_mean)



end_time = time.time()
print(f"Total time taken: {round((end_time - start_time)/60, 2)} minutes")


recon_df000 = pd.DataFrame({kk:vv for kk,vv in cond_result.items()}).T
dicttmp1 = recon_df000[['mae_cf', 'mse_cf', 'r2_cf', 'chatterjee_cf', 'mae_rc',  'mse_rc', 'r2_rc','chatterjee_rc']].mean(0).to_dict()


recon_df000.to_csv(os.path.join(save_dir, f'all_cf_rc_metrics_percondition_diffU.csv'))
save_path = os.path.join(save_dir, f'all_cf_mean_metrics_diffU.json')
with open(save_path, 'w') as f:
    json.dump(dicttmp1, f)
