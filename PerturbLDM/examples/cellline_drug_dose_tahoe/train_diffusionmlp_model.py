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
parser.add_argument("--save_dir", default="train_results", help="save dir")
parser.add_argument("--save_model_name", default="diffusionmlpfinal", help="save model name")
parser.add_argument("--latent_dir", required=True, help="latent_dir")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batchsize', type=int, default=1024)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--drug_vocab_size', type=int, default=None,
                    help='Drug vocabulary size; defaults to number of drugs in --data_root')
parser.add_argument('--ctrl_dim', type=int, default=None,
                    help='Control expression dimension; defaults to number of genes in control_adata.h5ad')
parser.add_argument('--hidden_dim', type=int, default=1536)
parser.add_argument('--context_dim', type=int, default=768)
parser.add_argument('--time_emb_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--feature_num', type=int, default=3, help='Number of features')
parser.add_argument('--use_dose_level_mapping', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--use_post_norm', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--use_residual', default=True, action=argparse.BooleanOptionalAction)
# parser.add_argument('--use_dose_level_mapping', action='store_true', help='Whether to use dose level mapping')
# parser.add_argument('--use_post_norm', action='store_true')
# parser.add_argument('--use_residual', action='store_true')

parser.add_argument('--dose_level_mapping_hidden_dim', type=int, default=256, help='Hidden dimension for dose level mapping')
parser.add_argument('--dose_level_mapping_dim', type=int, default=2, help='Dimension for dose level mapping')
parser.add_argument('--prediction_type', type=str, default='v_prediction', help='Type of prediction for the diffusion model: v_prediction, epsilon, or sample')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
parser.add_argument('--rescale_latent', default=False, action=argparse.BooleanOptionalAction, help='Whether to rescale latent representations')
parser.add_argument('--drug_emb_pretrained_path', type=str, default='', help='Path to pretrained drug embedding model')
parser.add_argument('--freeze_drug_emb', default=True, action=argparse.BooleanOptionalAction, help='Whether to freeze the pretrained drug embedding layer')



args = parser.parse_args()
data_root = args.data_root
dataset_name = os.path.basename(os.path.normpath(data_root)) or "Tahoe"
print(data_root)

PREDICTION_TYPE = args.prediction_type  # epsilon (predicts the noise of the diffusion process), sample (directly predicts the noisy sample) or v_prediction
print('PREDICTION_TYPE ', PREDICTION_TYPE)

latent_dir = args.latent_dir
if not os.path.isdir(latent_dir):
    raise ValueError(f'Cannot find latent_dir: {latent_dir}')

modeloutputdir = './{}/{}/{}-{}'.format(args.save_dir, dataset_name, args.save_model_name, time.strftime("%Y%m%d-%H%M%S"))


os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('cuda ', os.environ['CUDA_VISIBLE_DEVICES'])

import pickle
import pandas as pd
import numpy as np
import torch
torch.cuda.is_available()



drug_emb_pretrained_path = args.drug_emb_pretrained_path
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
    



import json
import scanpy as sc
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from util_diff import *
from util_metrics import *

datadir = data_root
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


cond2id = json.load(open(os.path.join(processed_datadir, "cond2id.json"), 'r'))
id2cond = {v:k for k,v in cond2id.items()}
train_df = pd.read_csv(os.path.join(collection_datadir, 'train_metadf.csv'))

control_adata = sc.read_h5ad(os.path.join(collection_datadir, "control_adata.h5ad"))
control_df = pd.read_csv(os.path.join(collection_datadir, "control_metadf.csv"))
control_adata_pt = torch.tensor(to_dense_array(control_adata.X), dtype=torch.float32)
control_cell_name_list = control_df['cell_name']
cellname_list = pd.unique(control_cell_name_list)
cell2exprmean = {}
for cell in cellname_list:
    cell2exprmean[cell] = torch.mean(control_adata_pt[control_cell_name_list == cell], 0)


condconvertor = Condition_IDConverter(cond2id.keys())
cond_labelnum = condconvertor.sepid_labelnum
condname_train_list = list(train_df['CondID'])
drug_vocab_size = args.drug_vocab_size or len(condconvertor.drug2id)
ctrl_dim = args.ctrl_dim or control_adata.shape[1]
if ctrl_dim != control_adata.shape[1]:
    raise ValueError(
        f"ctrl_dim ({ctrl_dim}) does not match control_adata gene dimension ({control_adata.shape[1]})."
    )
if drug_emb_pretrained is not None and drug_emb_pretrained.shape[0] != drug_vocab_size:
    raise ValueError(
        f"pretrained drug embedding rows ({drug_emb_pretrained.shape[0]}) do not match drug_vocab_size ({drug_vocab_size})."
    )

def get_cor_cell_expr_mean(ii):
    return(cell2exprmean[ii.split("___")[2]])


latent_train_data_path = os.path.join(latent_dir, 'train_latents.pt')
latent_train_data = torch.load(latent_train_data_path)
train_std = torch.std(latent_train_data)
print('train_std', train_std)
train_std_item = train_std.item()


rescale_latent = args.rescale_latent
if rescale_latent:
    print('Rescaling latent representations by train std')
    scale_factor = 1.0 / train_std
    latent_train_data = latent_train_data * scale_factor

condname_train_list_used = condname_train_list
traindataset = LatentDiffusionCachedDataset(condname_train_list_used, condconvertor, cell2exprmean, latents = latent_train_data)

temp = os.path.join(latent_dir, 'config.json')
latent_config = json.load(open(temp, 'r'))
print('latent_config ', latent_config)


print(PREDICTION_TYPE)

if PREDICTION_TYPE not in ['v_prediction', 'epsilon', 'sample']:
    raise ValueError('Unknown PREDICTION_TYPE')

max_scaled_latent = torch.max(torch.abs(latent_train_data))  
print('max_scaled_latent', max_scaled_latent)
clip_sample_range = max_scaled_latent.item()
print('clip_sample_range', clip_sample_range) # clip_sample_range is already the scaled value if rescale_latent is True

noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample_range=clip_sample_range,prediction_type=PREDICTION_TYPE,beta_start = 0.00085,beta_end = 0.015)



num_epochs = args.num_epochs
lr = args.lr
batchsize  = args.batchsize
weight_decay = args.weight_decay
config_dict = {'batchsize': batchsize, 'num_epochs': num_epochs, 'lr': lr,
              'latent_dir': latent_dir, "clip_sample_range": clip_sample_range,
              "prediction_type": PREDICTION_TYPE, "weight_decay": weight_decay,
              "train_std_item": train_std_item, "rescale_latent": rescale_latent,
              "drug_emb_pretrained_path": drug_emb_pretrained_path}
print(config_dict)

traindataloader = DataLoader(traindataset, batch_size=config_dict['batchsize'], shuffle=True)

from DenoisingMLPFinal import DenoisingModelDrugDose
latent_dim = latent_train_data.shape[-1]

print('latent_dim ', latent_dim)
config_diff = {
    "latent_dim": latent_dim,
    "drug_vocab_size": drug_vocab_size,
    "ctrl_dim": ctrl_dim,
    "hidden_dim": args.hidden_dim,
    "num_layers": args.num_layers,
    "context_dim": args.context_dim,
    "time_emb_dim": args.time_emb_dim,
    "dropout": args.dropout,
    "use_post_norm": args.use_post_norm,
    "use_residual": args.use_residual,
    "feature_num": args.feature_num,
    "use_dose_level_mapping": args.use_dose_level_mapping,
    "dose_level_mapping_hidden_dim": args.dose_level_mapping_hidden_dim,
    "dose_level_mapping_dim": args.dose_level_mapping_dim,
    "freeze_drug_emb": args.freeze_drug_emb,
}
print(config_diff)

denoising_model = DenoisingModelDrugDose(**config_diff, drug_emb_pretrained = drug_emb_pretrained)
num_params = sum(p.numel() for p in denoising_model.parameters() if p.requires_grad)
num_params_M = num_params / 1e6
print(f"Trainable parameters: {num_params_M:.2f} M")
print(denoising_model)
denoising_model.cuda()
denoising_model.train()

log_history = train_loop_diffusion(denoising_model, traindataloader, noise_scheduler, num_epochs = config_dict['num_epochs'], lr = config_dict['lr'], eval_steps = 1000, prediction_type = PREDICTION_TYPE, 
                                   weight_decay = config_dict['weight_decay'])


save_all_trained_result(modeloutputdir, config_diff, config_dict, log_history,  denoising_model)
