"""Run the colon-development transfer experiment with PerturbLDM."""

import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument("--subtype", type=str, default="mse",
                    help="Subtype (e.g., mse)")
parser.add_argument("--topgenenum", type=int, default=800,
                    help="Number of top genes")
parser.add_argument("--cuda_id", type=int, default=0,
                    help="CUDA device ID")
parser.add_argument("--basedir", default=".",
                    help="Directory containing Development/adata_enterocyte_colon.h5ad")
parser.add_argument("--output_dir", default=None,
                    help="Directory for outputs; defaults to basedir")


args = parser.parse_args()

subtype = args.subtype
TOPGENENUM = args.topgenenum
cuda_id = args.cuda_id
basedir = args.basedir
output_base = args.output_dir or basedir

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)


import os


import time
timestr = time.strftime("%m%d-%H%M%S")


colon_datasetname = 'adata_enterocyte_colon.h5ad'
use_dataset_path = os.path.join(basedir, 'Development', colon_datasetname)

print(use_dataset_path)

str00 = use_dataset_path[:-5].split('_')[-1]+f'_{subtype}'
modeloutputdir = os.path.join(output_base, f'development_{str00}_{TOPGENENUM}_{timestr}')
os.makedirs(modeloutputdir, exist_ok=True)
print(str00)
resultoutputdir = modeloutputdir



import scanpy as sc
import anndata
import numpy as np
import pandas as pd


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


adata = sc.read_h5ad(use_dataset_path)
print(adata.shape)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


min_cells = int(adata.shape[0]*0.01)
print('min_cells: ', min_cells)
sc.pp.filter_genes(adata, min_cells=min_cells)
print(adata.shape)

## prepare conditions
def convert_pcw_to_broad_stage(pcw):
    if pcw <= 14:
        return 1.0
    if pcw <= 18:
        return 2.0
    else:
        return 3.0
adata.obs['broad_stage'] = adata.obs['pcw'].apply(convert_pcw_to_broad_stage)
condition_key = 'broad_stage'



print('split train and test data based on pcw')
train_id = ~adata.obs['pcw'].isin([15,16,17,18])
train_adata = adata[train_id].copy()
test_adata = adata[~train_id].copy()


train_adatatemp = adata.copy()
# train_adatatemp = train_adata.copy()

sc.pp.highly_variable_genes(train_adatatemp, n_top_genes=TOPGENENUM, flavor='seurat')
# 3️⃣ Subset the data to include only HVGs
gene_train_adatatemp = train_adatatemp.var[train_adatatemp.var['highly_variable']].index
print(len(gene_train_adatatemp))


train_adata_final = train_adata[:, gene_train_adatatemp]
test_adata_final = test_adata[:, gene_train_adatatemp]


from anndata import AnnData
import pandas as pd

### train VAE
os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
import torch


from LatentModelFinal import CellEncoderWithLogvar
from util_latent import train_loop_latent, TensorDataset
from util_metrics import compute_metrics_single


traindataset = TensorDataset(train_adata_final.X)


from torch.utils.data import DataLoader


use_dec_logvar = False
if subtype == 'logvar':
    use_dec_logvar = True
    recon_loss_type = 'mse'
elif subtype == 'smoothl1':
    recon_loss_type = "smoothl1"
elif subtype == 'mse':
    recon_loss_type = 'mse'
elif subtype == 'l1':
    recon_loss_type = 'l1'
else:
    raise ValueError('invalid value')

print(use_dec_logvar, recon_loss_type)


latent_config = {
    "lr": 1e-4,
    "max_steps": 15000,
    "batchsize": 256,
    'latent_dim': 64,
    'input_dim': TOPGENENUM,
    "kl_weight": 0.0002,
    "dropout": 0.2,
    "use_variational": True,
    'hidden_dim': 128,
    'dec_hidden': 384,
    'hidden_dim_en': [384],
    'hidden_dim_de': [128],
    "weight_decay": 3e-4,
    "use_dec_logvar": use_dec_logvar,
    "recon_loss_type": recon_loss_type,#'mse', smoothl1, l1
    "distribution_type": "gauss" # "gauss", "nb"
}

print(latent_config)

train_loader = DataLoader(traindataset, latent_config['batchsize'], shuffle=True,)
fix_train_loader = DataLoader(traindataset, latent_config['batchsize'], shuffle=False, drop_last=False)

latentmodel = CellEncoderWithLogvar(**{kk:latent_config[kk] for kk in latent_config if kk not in ['lr', 'max_steps', "batchsize", 'weight_decay']})
print(latentmodel)
latentmodel.cuda()




print('Start training latent model')
log_history = train_loop_latent(latentmodel, train_loader= train_loader, lr = latent_config['lr'], max_steps = latent_config['max_steps'], device = 'cuda', log_step = 100, save_dir = modeloutputdir, weight_decay = latent_config['weight_decay'])
print(modeloutputdir)




import json
os.makedirs(modeloutputdir, exist_ok=True)
torch.save(latentmodel.state_dict(), os.path.join(modeloutputdir, "latent_model_weights.pth"))

with open(os.path.join(modeloutputdir, "latent_config.json"), "w") as f:
    json.dump(latent_config, f, indent=4)
with open(os.path.join(modeloutputdir, "latent_log_history.json"), "w") as f:
    json.dump(log_history, f, indent=4)
    

print('finish training latent model and generate latent representations for train data of diffusion model')
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


with torch.no_grad():
    temp_dict = latentmodel.encode(torch.FloatTensor(to_dense_array(train_adata_final.X)).cuda())
    rc_dict = latentmodel.decode(temp_dict['latents'].cuda())
    train_latents0 = temp_dict['latents'].cpu().numpy()
rc_output_train = rc_dict['recon_mu'].cpu().clamp(min=0).numpy()
gt_train = to_dense_array(train_adata_final.X)

with torch.no_grad():
    temp_dict = latentmodel.encode(torch.FloatTensor(to_dense_array(test_adata_final.X)).cuda())
    rc_dict = latentmodel.decode(temp_dict['latents'].cuda())
    test_latents0 = temp_dict['latents'].cpu().numpy()
rc_output_test = rc_dict['recon_mu'].cpu().clamp(min=0).numpy()
gt_test = to_dense_array(test_adata_final.X)


### customize conditions
### Need specify latents and conditions in samples, and condition_setting_dict

from torch.utils.data import Dataset
class TensorDatasetCondition(Dataset):
    def __init__(self, data_array, labels):
        self.data_array = data_array
        self.labels = labels
    def __len__(self):
        return len(self.data_array)
    def __getitem__(self, idx):
        return {"latents": self.data_array[idx], condition_key: self.labels[idx]}


latent_train_data = latenttmp
label_train = train_adata_final.obs[condition_key].astype(np.float32)
label_train = np.array(label_train).reshape(-1,1)
condition_setting_dict = {condition_key: ('continuous', 1)}


from DenoisingMLPFinal import DenoisingModelConditions

from util_diff import train_loop_diffusion

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput



SCALE_FACTOR = 1.0
# SCALE_FACTOR = 1.0/torch.std(latent_train_data).item()
print("SCALE_FACTOR:", SCALE_FACTOR)

train_dataset_diff = TensorDatasetCondition(latent_train_data*SCALE_FACTOR, np.array(label_train))

denoising_setting_config = {
    'latent_dim': latent_config['latent_dim'],
    'condition_setting_dict': condition_setting_dict,
    'context_dim': 256,
    'time_emb_dim': 32,
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.2,
    'use_post_norm': True,
    'use_residual': True,
}


max_scaled_latent = torch.max(torch.abs(latent_train_data*SCALE_FACTOR))  
print(max_scaled_latent)
clip_sample_range = max_scaled_latent.item()
print(clip_sample_range)

num_epochs = 1200
lr = 1.5e-4
batchsize  = 256
diff_train_config_dict = {'batchsize': batchsize, 'num_epochs': num_epochs, 'lr': lr, "clip_sample_range": clip_sample_range, 'weight_decay': 3e-4, 'prediction_type': 'v_prediction'}
print(diff_train_config_dict)

PREDICTION_TYPE = diff_train_config_dict['prediction_type'] # epsilon (predicts the noise of the diffusion process), sample (directly predicts the noisy sample) or v_prediction
print(PREDICTION_TYPE)

if PREDICTION_TYPE not in ['v_prediction', 'epsilon', 'sample']:
    raise ValueError('Unknown PREDICTION_TYPE')
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample_range=clip_sample_range,prediction_type=PREDICTION_TYPE,beta_start = 0.00085,beta_end = 0.015)



denoising_model = DenoisingModelConditions(**denoising_setting_config)

num_params = sum(p.numel() for p in denoising_model.parameters() if p.requires_grad)
num_params_M = num_params / 1e6
print(f"Trainable parameters: {num_params_M:.2f} M")
print(denoising_model)
denoising_model.cuda()
denoising_model.train()

traindataloader = DataLoader(train_dataset_diff, batch_size=diff_train_config_dict['batchsize'], shuffle=True, drop_last=True)

diff_log_history = train_loop_diffusion(denoising_model, traindataloader, noise_scheduler, num_epochs = diff_train_config_dict['num_epochs'], lr = diff_train_config_dict['lr'], eval_steps = 1000, prediction_type = PREDICTION_TYPE, weight_decay = diff_train_config_dict['weight_decay'])


torch.save(denoising_model.state_dict(), os.path.join(modeloutputdir, "denoising_model_weights.pth"))

with open(os.path.join(modeloutputdir, "diff_train_config_dict.json"), "w") as f:
    json.dump(diff_train_config_dict, f, indent=4)
with open(os.path.join(modeloutputdir, "diff_log_history.json"), "w") as f:
    json.dump(diff_log_history, f, indent=4)
with open(os.path.join(modeloutputdir, "denoising_setting_config.json"), "w") as f:
    json.dump(denoising_setting_config, f, indent=4)


from util_plot import *
from util_inference_diff import inference_process_batch_conditions

print('construct test batch for inference, only condition inputs')
label_test = test_adata_final.obs[condition_key].astype(np.float32)
label_test = np.array(label_test).reshape(-1,1)
test_batch = {condition_key: torch.FloatTensor(label_test)}


diff_result, latent_diff_steps = inference_process_batch_conditions(test_batch, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True)
with torch.no_grad():
    cf_pred = latentmodel.decode(diff_result.cuda()/SCALE_FACTOR)
cf_expr_test = cf_pred['reconstruction_expr'].cpu()

gt_test = to_dense_array(test_adata_final.X)
cf_expr_test_np = cf_expr_test.numpy()
cf_expr_test_np = np.clip(cf_expr_test_np, a_min=0, a_max=None)




train_batch = {condition_key: torch.FloatTensor(label_train)}
diff_result, latent_diff_steps_train = inference_process_batch_conditions(train_batch, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True)
with torch.no_grad():
    cf_pred = latentmodel.decode(diff_result.cuda()/SCALE_FACTOR)
cf_expr_train = cf_pred['reconstruction_expr'].cpu()

gt_train = to_dense_array(train_adata_final.X)
cf_expr_train_np = cf_expr_train.numpy()
cf_expr_train_np = np.clip(cf_expr_train_np, a_min=0, a_max=None)


rc_metric = compute_metrics_single(gt_test.mean(0), rc_output_test.mean(0))
test_metrics = compute_metrics_single(gt_test.mean(0), cf_expr_test_np.mean(0))
print('test metrics: ', test_metrics)
with open(os.path.join(resultoutputdir, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=4)
with open(os.path.join(resultoutputdir, "rc_test_metrics.json"), "w") as f:
    json.dump(rc_metric, f, indent=4)


one_obs = pd.concat([test_adata_final.obs, train_adata_final.obs])
one_var = pd.DataFrame(index=train_adata_final.var.index)
def plot_umap_sc(inputX, scale = False, random_state1 = 7434, random_state2 = 1454, n_pcs = 20, n_neighbors = 12):
    adata00 = AnnData(X=inputX.copy(), obs=one_obs, var=None)
    if scale:
        sc.pp.scale(adata00, max_value=10)  # clip extreme values to avoid heavy tails
    sc.tl.pca(adata00, svd_solver='arpack')
    sc.pp.neighbors(adata00, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state = random_state1)
    sc.tl.umap(adata00, random_state = random_state2)
    sc.pl.umap(adata00, color='pcw')


exprdata_cftest_gttrain = np.concatenate([cf_expr_test_np, to_dense_array(train_adata_final.X)], axis=0)
exprdata_allgt = np.concatenate([to_dense_array(test_adata_final.X), to_dense_array(train_adata_final.X)], axis=0)
exprdata_allcf = np.concatenate([cf_expr_test_np, cf_expr_train_np], axis=0)

latentdata_allcf = []
for ii in range(latent_diff_steps.shape[0]):
    latentdata_allcf.append( np.concatenate([latent_diff_steps[ii].numpy(), latent_diff_steps_train[ii].numpy()], axis=0) )

group_points = np.concatenate([test_adata_final.obs['pcw'], train_adata_final.obs['pcw']], axis=0).astype(np.float32)


test_adata_final.obsm['cf_expr'] = cf_expr_test_np
test_adata_final.obsm['cf_latent'] = latent_diff_steps[-1].numpy()
for ii,step_idx in enumerate(range(1000, -1, -50)):
    test_adata_final.obsm[f'cf_latent_step{step_idx}'] = latent_diff_steps[ii].numpy()


train_adata_final.obsm['cf_expr'] = cf_expr_train_np
train_adata_final.obsm['cf_latent'] = latent_diff_steps_train[-1].numpy()
for ii,step_idx in enumerate(range(1000, -1, -50)):
    train_adata_final.obsm[f'cf_latent_step{step_idx}'] = latent_diff_steps_train[ii].numpy()

# save test_adata_final and train_adata_final
temp = os.path.join(modeloutputdir, 'test_adata_final_diffU.h5ad')
test_adata_final.write_h5ad(temp)
print(f'saved test adata to {temp}')
temp = os.path.join(modeloutputdir, 'train_adata_final_diffU.h5ad')
train_adata_final.write_h5ad(temp)
print(f'saved train adata to {temp}')



import matplotlib.pyplot as plt
t1 = gt_test.mean(0)
t2 = cf_expr_test_np.mean(0)
t3 = rc_output_test.mean(0)
plt.figure(figsize = (4,4))
plt.scatter(t1, t2, alpha=0.5)
plt.scatter(t1, t3, alpha=0.5)
plt.plot([t1.min(), t1.max()], [t1.min(), t1.max()], 'k--')
plt.savefig(os.path.join(modeloutputdir, f"scatter_test_cf_rc.png"))
plt.show()
