import os
import time
import argparse
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="CUDA device(s) to use, e.g., 0 or '0,1'")
parser.add_argument("--data_root", required=True,
                    help="Processed Tahoe dataset directory containing collection/ and processed/")
parser.add_argument("--latent_dir", required=True, help="latent_dir")


args = parser.parse_args()
data_root = args.data_root
print(data_root)

latent_dir = args.latent_dir
if not os.path.isdir(latent_dir):
    raise ValueError(f'Cannot find latent_dir: {latent_dir}')

save_dir = latent_dir
print(latent_dir)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('cuda ', os.environ['CUDA_VISIBLE_DEVICES'])

import pandas as pd
import numpy as np
import torch
torch.cuda.is_available()
import scanpy as sc
import json

from LatentModelFinal import CellEncoderWithLogvar
latent_config = json.load(open(os.path.join(latent_dir, 'config.json'), 'r'))
print(latent_config)
latentmodel = CellEncoderWithLogvar(**{kk:vv for kk,vv in latent_config.items() if kk not in ['lr', 'weight_decay', 'batchsize', 'max_steps']})
state_dict = torch.load(os.path.join(latent_dir, 'model_weights.pth'))
latentmodel.load_state_dict(state_dict)
latentmodel.eval()
latentmodel.cuda()
print('load finetuned weights.')


def get_rc_mu(array):    
    with torch.no_grad():
        latent_temp = latentmodel.encode(torch.tensor(array).cuda())['latents']
        recon_mu_diff = latentmodel.decode(latent_temp)['recon_mu'].cpu()
    recon_mu_clamped = recon_mu_diff.clamp(0)
    return(recon_mu_clamped.numpy())




from util_metrics import *

datadir = data_root
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)



test_adata = sc.read_h5ad(os.path.join(collection_datadir, 'test_adata.h5ad'))
test_df = pd.read_csv(os.path.join(collection_datadir, 'test_metadf.csv'))
condname_test_list = list(test_df['CondID'])

import random
from collections import defaultdict

groups_test = defaultdict(list)
for idx, name in enumerate(condname_test_list):
    groups_test[name].append(idx)


cond_result = {}
iii = 0

condname2gtmean_rcmean = {}

for condi in groups_test:
    iii += 1
    print(condi)        
    if condi in cond_result:
        raise ValueError(f'{condi} in cond_result')
    
    temp_expr = test_adata.X[groups_test[condi]]
    expr_pt = torch.tensor(to_dense_array(temp_expr), dtype=torch.float32)
    
    with torch.no_grad():
        encode_temp_dict = latentmodel.encode(expr_pt.cuda())
        latent_condi = encode_temp_dict['latents'].cpu()
    
    with torch.no_grad():
        recon_condi = latentmodel.decode(latent_condi.cuda())['recon_mu'].cpu()
        recon_condi_clamped = recon_condi.clamp(0)
        rc_mean_temp = recon_condi_clamped.mean(0)
        expr_pt_mean = expr_pt.mean(0)
    
    dict_output00 = {'mae_pw_rc':torch.mean(torch.abs(expr_pt-recon_condi_clamped)).item(), 
     'mse_pw_rc':torch.mean((expr_pt-recon_condi_clamped)**2).item()}
    mean_met_temp = compute_metrics_single(expr_pt_mean, rc_mean_temp)
    dict_output00.update({kk+'_rc':vv for kk,vv in mean_met_temp.items()})    
    cond_result[condi] = dict_output00
    condname2gtmean_rcmean[condi] = {'gt_mean': expr_pt_mean, 'rc_mean': rc_mean_temp}


recon_df000 = pd.DataFrame({kk:vv for kk,vv in cond_result.items()}).T
dicttmp1 = recon_df000[['mae_pw_rc', 'mse_pw_rc', 'mse_rc', 'r2_rc', 'mae_rc', 'chatterjee_rc']].mean(0).to_dict()


recon_df000.to_csv(os.path.join(save_dir, f'recon_metrics_percondition.csv'))
save_path = os.path.join(save_dir, f'rc_mean_metrics.json')
with open(save_path, 'w') as f:
    json.dump(dicttmp1, f)

import pickle
save_path = os.path.join(save_dir, f'condname2gtmean_rcmean.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(condname2gtmean_rcmean, f)
print(f'Saved rc mean metrics to {save_path}')
