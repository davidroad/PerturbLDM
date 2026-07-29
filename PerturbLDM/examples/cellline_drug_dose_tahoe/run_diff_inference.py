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
parser.add_argument("--condition_col", default="CondID",
                    help="Condition identifier column in test_metadf.csv.")
parser.add_argument("--num_samples", type=int, default=50,
                    help="Number of generated cells sampled per held-out condition.")
parser.add_argument("--save_latent_steps", default=False, action=argparse.BooleanOptionalAction,
                    help="Also save reverse-diffusion latent snapshots. This can be very large for all Tahoe conditions.")



args = parser.parse_args()
if args.num_samples < 1:
    raise ValueError("--num_samples must be positive")

diff_dir = args.diff_dir
if not os.path.isdir(diff_dir):
    raise ValueError(f'Cannot find diff_dir: {diff_dir}')

save_dir = diff_dir
print(diff_dir)


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


datadir = args.data_root
print(datadir)
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

cond2id = json.load(open(os.path.join(processed_datadir, "cond2id.json"), 'r'))
id2cond = {v:k for k,v in cond2id.items()}


from util_diff import *
condconvertor = Condition_IDConverter(cond2id.keys())
cond_labelnum = condconvertor.sepid_labelnum

config_train_dict = json.load(open(os.path.join(diff_dir, 'config_train.json'), 'r'))
config_dict = json.load(open(os.path.join(diff_dir, 'config.json'), 'r'))
latent_dim = config_dict['latent_dim']

test_df = pd.read_csv(os.path.join(collection_datadir, "test_metadf.csv"))
if args.condition_col not in test_df.columns:
    raise KeyError(f"{args.condition_col!r} is missing from test metadata")
allcondnames = list(pd.unique(test_df[args.condition_col].astype(str)))
if not allcondnames:
    raise ValueError("No held-out test conditions were found")




print(config_train_dict)
print(config_dict)

clip_sample_range = config_train_dict['clip_sample_range']
print(clip_sample_range)
prediction_type = config_train_dict.get("prediction_type", "v_prediction")
PREDICTION_TYPE = prediction_type
print("prediction_type: ", PREDICTION_TYPE)

drug_emb_pretrained_path = config_train_dict.get("drug_emb_pretrained_path", None)
print("drug_emb_pretrained_path: ", drug_emb_pretrained_path)

drug_emb_pretrained = None
try:
    if drug_emb_pretrained_path:
        if os.path.isfile(drug_emb_pretrained_path):
            print('Using pretrained drug embedding model from ', drug_emb_pretrained_path)
            drug_emb_pretrained = torch.load(drug_emb_pretrained_path)
            print('drug_emb_pretrained', drug_emb_pretrained.shape)
except Exception as e:
    print('Error in loading pretrained drug embedding model:', e)
    drug_emb_pretrained = None

if drug_emb_pretrained is not None:
    print('Using pretrained drug embedding model with shape:', drug_emb_pretrained.shape)
else:
    print('Not using pretrained drug embedding model')
    


rescale_latent = config_train_dict.get("rescale_latent", False)
print("rescale_latent: ", rescale_latent)

train_std_item = config_train_dict.get("train_std_item")
print("train_std_item: ", train_std_item)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample_range=clip_sample_range,prediction_type=PREDICTION_TYPE,beta_start = 0.00085,beta_end = 0.015)


print('Load Lineardose Diffusion Model')
from DenoisingMLPFinal import DenoisingModelDrugDose
denoising_model = DenoisingModelDrugDose(**config_dict, drug_emb_pretrained = drug_emb_pretrained)


state_dict_diff = torch.load(os.path.join(diff_dir, 'model_weights.pth'))
denoising_model.load_state_dict(state_dict_diff)
denoising_model.eval()
denoising_model.cuda()

print('denoising model: load finetuned weights.')
print(denoising_model)

control_adata = sc.read_h5ad(os.path.join(collection_datadir, "control_adata.h5ad"))
control_df = pd.read_csv(os.path.join(collection_datadir, "control_metadf.csv"))
control_adata_pt = torch.tensor(to_dense_array(control_adata.X), dtype=torch.float32)
control_cell_name_list = control_df['cell_name']
cellname_list = pd.unique(control_cell_name_list)
cell2exprmean = {}
for cell in cellname_list:
    cell2exprmean[cell] = torch.mean(control_adata_pt[control_cell_name_list == cell], 0)

from util_inference_diff import *

def get_input_by_condition(condname):
    ctrl_latent = cell2exprmean[condname.split("___")[2]]
    cond = condconvertor._process_cond(condname)
    temp = {
        "ctrl_expr": ctrl_latent,
    }
    temp.update({kk:cond[kk] for kk in ["drug_id"]})
    temp["dose_uM_level"] = np.float32(np.log10(np.float32(cond["dose_uM"])/0.05))
    return(temp)



import time
start_time = time.time()

latent_output_list = []
latent_steps_output_list = [] if args.save_latent_steps else None

for condi in allcondnames:
    print(condi)
    input_00 = get_input_by_condition(condi)
    batchtemp = make_batch(input_00, args.num_samples)
    if args.save_latent_steps:
        latent_output, latent_output_steps = inference_process_batch_logstep(
            batchtemp, noise_scheduler, denoising_model, latent_dim=latent_dim
        )
        latent_steps_output_list.append(latent_output_steps.cpu())
    else:
        latent_output = inference_process_batch(
            batchtemp, noise_scheduler, denoising_model, latent_dim=latent_dim
        )
    latent_output_list.append(latent_output.cpu())
    

latent_output_final = torch.stack(latent_output_list, 0)

print("latent_output_final shape: ", latent_output_final.shape)
if args.save_latent_steps:
    latent_steps_output_final = torch.stack(latent_steps_output_list, 1)
    print("latent_steps_output_final shape: ", latent_steps_output_final.shape)


end_time = time.time()
print(f"Total time taken: {round((end_time - start_time)/60, 2)} minutes")


# save allcondnames
save_path = os.path.join(save_dir, f'test_allcondnames.json')
json.dump(allcondnames, open(save_path, 'w'))

# save latent_output_final
save_path =  os.path.join(save_dir, f'test_latent_outputs.pt')
torch.save(latent_output_final, save_path)

# save latent_steps_output_final
if args.save_latent_steps:
    save_path =  os.path.join(save_dir, f'test_latent_steps_outputs.pt')
    torch.save(latent_steps_output_final, save_path)
