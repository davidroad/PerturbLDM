import os
import time
import argparse
import sys
import ast
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="gpu device(s) to use, e.g., 0")
parser.add_argument("--data_root", required=True,
                    help="Processed Tahoe dataset directory containing collection/ and processed/")
parser.add_argument("--save_dir", default="train_results", help="save dir")
parser.add_argument("--save_model_name", default="latent", help="save model name")
parser.add_argument('--lr', type=float, default=6e-4)
parser.add_argument('--max_steps', type=int, default=20000)
parser.add_argument('--batchsize', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=8e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--use_variational', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--latent_dim', type=int, default=768)
parser.add_argument('--input_dim', type=int, default=None,
                    help='Expression dimension; defaults to number of genes in train_adata.h5ad')
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--dec_hidden', type=int, default=2048)
parser.add_argument('--hidden_dim_en', type=str, default="[2048, 1024]")
parser.add_argument('--hidden_dim_de', type=str, default="[1024, 1024]")


args = parser.parse_args()
data_root = args.data_root
dataset_name = os.path.basename(os.path.normpath(data_root)) or "Tahoe"
modeloutputdir = './{}/{}/{}-{}'.format(args.save_dir, dataset_name, args.save_model_name, time.strftime("%Y%m%d-%H%M%S"))


os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('cuda ', os.environ['CUDA_VISIBLE_DEVICES'])
print(data_root)


import pickle
import pandas as pd
import numpy as np
import torch
print('cuda: ', torch.cuda.is_available())
import json
import scanpy as sc
from util_latent import *


datadir = data_root
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")
train_adata = sc.read_h5ad(os.path.join(collection_datadir, 'train_adata.h5ad'))
train_df = pd.read_csv(os.path.join(collection_datadir, 'train_metadf.csv'))
traindataset = TensorDataset(train_adata.X)

input_dim = args.input_dim or train_adata.shape[1]
if input_dim != train_adata.shape[1]:
    raise ValueError(
        f"input_dim ({input_dim}) does not match train_adata gene dimension ({train_adata.shape[1]})."
    )
hidden_dim_en = ast.literal_eval(args.hidden_dim_en)
hidden_dim_de = ast.literal_eval(args.hidden_dim_de)
if not isinstance(hidden_dim_en, list) or not isinstance(hidden_dim_de, list):
    raise ValueError("hidden_dim_en and hidden_dim_de must be Python list literals.")


config = {
    "latent_dim": args.latent_dim,
    "input_dim": input_dim,
    "hidden_dim": args.hidden_dim,
    "dec_hidden": args.dec_hidden,
    "hidden_dim_en": hidden_dim_en,
    "hidden_dim_de": hidden_dim_de,
    "lr": args.lr,
    "max_steps": args.max_steps,
    "batchsize": args.batchsize,
    "kl_weight": args.kl_weight,
    "dropout": args.dropout,
    "use_variational": args.use_variational,
    "weight_decay": args.weight_decay,
}

print(config)

train_loader = DataLoader(traindataset, config['batchsize'], shuffle=True,)
fix_train_loader = DataLoader(traindataset, config['batchsize'], shuffle=False, drop_last=False)



from LatentModelFinal import CellEncoderWithLogvar
latentmodel = CellEncoderWithLogvar(**{kk:config[kk] for kk in config if kk not in ['lr', 'max_steps', "batchsize", "weight_decay"]})
print(latentmodel)
latentmodel.cuda()
print('Start training latent model')
log_history = train_loop_latent(latentmodel, train_loader= train_loader, lr = config['lr'], max_steps = config['max_steps'], device = 'cuda', log_step = 500, save_dir = modeloutputdir, weight_decay = config['weight_decay'])
print(modeloutputdir)

os.makedirs(modeloutputdir, exist_ok=True)
torch.save(latentmodel.state_dict(), os.path.join(modeloutputdir, "model_weights.pth"))

with open(os.path.join(modeloutputdir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)
with open(os.path.join(modeloutputdir, "log_history.json"), "w") as f:
    json.dump(log_history, f, indent=4)


latentmodel.eval()
latentmodel.cuda()
with torch.no_grad():
    trainlatent_list = []
    for x in fix_train_loader:
        output = latentmodel.encode(**{k:v.cuda() for k,v in x.items()})
        latents = output["latents"]
        trainlatent_list.append(latents.cpu())       
    latenttmp = torch.concat(trainlatent_list)
    
torch.save(latenttmp, os.path.join(modeloutputdir, "train_latents.pt"))
print('finish generating train_latents')
