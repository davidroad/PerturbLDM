"""Run the PBMC IFN transfer experiment with PerturbLDM."""

import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument("--subtype", type=str, default="mse",
                    help="Subtype (e.g., mse)")
parser.add_argument("--topgenenum", type=int, default=2000,
                    help="Number of top genes")
parser.add_argument("--cuda_id", type=int, default=0,
                    help="CUDA device ID")
parser.add_argument("--basedir", default=".",
                    help="Directory containing pbmc_IFN_filtered.h5ad")
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

print(f"subtype: {subtype}")
print(f"TOPGENENUM: {TOPGENENUM}")
print(f"Using CUDA device: {cuda_id}")
datasetname = 'pbmc_IFN_filtered.h5ad'
use_dataset_path = os.path.join(basedir, datasetname)
print(use_dataset_path)

import time
timestr = time.strftime("%m%d-%H%M%S")

latent_dim = 128


modeloutputdir = os.path.join(output_base, f'pbmc_{subtype}_{TOPGENENUM}_ld{latent_dim}_{timestr}')
os.makedirs(modeloutputdir, exist_ok=True)
print(modeloutputdir)


import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


adata = sc.read_h5ad(use_dataset_path)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


print(adata.shape)
mincell = int(len(adata)*0.01)
sc.pp.filter_genes(adata, min_cells=mincell)
print(adata.shape)


adata.obs['split'] = 'train'
cellnames_small = ['Megakaryocytes', 'Dendritic cells', 'CD8 T cells', 'NK cells']

train_cellnames = ['CD4 T cells', 'CD14+ Monocytes',  'Dendritic cells', 'NK cells', ]
test_cellnames = ['FCGR3A+ Monocytes',  'CD8 T cells','B cells',  ]

print('train_cellnames: ',train_cellnames)
print('test_cellnames: ', test_cellnames)

rmcells = cellnames_small[:1]
print(rmcells)

adata_used = adata[~adata.obs['cell.type'].isin(rmcells)]
print(adata_used.shape)
adata_used.X = adata_used.X.astype(np.float32)
testid = (adata_used.obs['cell.type'].isin(test_cellnames)) & (adata_used.obs['stim'] =='stim')
print(sum(testid))

test_adata = adata_used[testid].copy()
train_adata = adata_used[~testid].copy()

train_adatatemp = adata_used.copy()
# train_adatatemp = train_adata.copy()
sc.pp.highly_variable_genes(train_adatatemp, n_top_genes=TOPGENENUM, flavor='seurat')
# 3️⃣ Subset the data to include only HVGs
gene_train_adatatemp = train_adatatemp.var[train_adatatemp.var['highly_variable']].index

train_adata_final = train_adata[:, gene_train_adatatemp]
test_adata_final = test_adata[:, gene_train_adatatemp]
train_adata_final.X = train_adata_final.X.astype(np.float32)
test_adata_final.X = test_adata_final.X.astype(np.float32)

all_adata_final = anndata.AnnData.concatenate(train_adata_final, test_adata_final,
                                              batch_key='set', batch_categories=['train', 'test'])


conditions_key = ['cell.type', 'stim']

### train VAE
os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)


import torch


from LatentModelFinal import CellEncoderWithLogvar
from util_latent import train_loop_latent, TensorDataset
from util_metrics import compute_metrics_single


traindataset = TensorDataset(train_adata_final.X.astype(np.float32))
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
    "max_steps": 20000,
    "batchsize": 256,
    'latent_dim': latent_dim,
    'input_dim': TOPGENENUM,
    "kl_weight": 0.0002,
    "dropout": 0.2,
    "use_variational": True,
    'hidden_dim': latent_dim*2,
    'dec_hidden': latent_dim*6,
    'hidden_dim_en': [latent_dim*6],
    'hidden_dim_de': [latent_dim*2],
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

train_X_array = to_dense_array(train_adata_final.X).astype(np.float32)
test_X_array = to_dense_array(test_adata_final.X).astype(np.float32)

with torch.no_grad():
    temp_dict = latentmodel.encode(torch.FloatTensor(train_X_array).cuda())
    rc_dict = latentmodel.decode(temp_dict['latents'].cuda())
    train_latents0 = temp_dict['latents'].cpu().numpy()
rc_output_train = rc_dict['recon_mu'].cpu().clamp(min=0).numpy()
gt_train = train_X_array

with torch.no_grad():
    temp_dict = latentmodel.encode(torch.FloatTensor(test_X_array).cuda())
    rc_dict = latentmodel.decode(temp_dict['latents'].cuda())
    test_latents0 = temp_dict['latents'].cpu().numpy()
rc_output_test = rc_dict['recon_mu'].cpu().clamp(min=0).numpy()
gt_test = test_X_array


from util_plot import *
from util_inference_diff import inference_process_batch_conditions


temp0 = test_adata_final.obs['cell.type']
unique00 = temp0.unique()
rctemp_result0 = {}
for xx in unique00:
    id00 = temp0==xx
    rctemp_result0[xx] = compute_metrics_single(gt_test[id00].mean(0), rc_output_test[id00].mean(0))

rctemp_result0df = pd.DataFrame(rctemp_result0).T
rctemp_result0df.to_csv(os.path.join(modeloutputdir, f"rc_metrics_test.csv"))


latent_train_data = latenttmp
conditionskey_name2id = {}
conditionskey_id2name = {}
for xx in conditions_key:
    conditionskey_name2id[xx] = {name: i for i, name in enumerate(train_adata_final.obs[xx].unique())}
    conditionskey_id2name[xx] = {i: name for name, i in conditionskey_name2id[xx].items()}

label_train = {xx:np.array(train_adata_final.obs[xx].map(conditionskey_name2id[xx])).astype(np.int32) for xx in conditions_key}


condition_setting_dict = {xx: ('categorical', len(conditionskey_name2id[xx])) for xx in conditions_key}

label_test_dict = {xx:np.array(test_adata_final.obs[xx].map(conditionskey_name2id[xx])).astype(np.int32) for xx in conditions_key}


### customize conditions
### Need specify latents and conditions in samples, and condition_setting_dict

from torch.utils.data import Dataset
class TensorDatasetConditionDict(Dataset):
    def __init__(self, data_array, label_dict):
        self.data_array = data_array
        self.label_dict = label_dict
    def __len__(self):
        return len(self.data_array)
    def __getitem__(self, idx):
        temp = {xx: self.label_dict[xx][idx] for xx in self.label_dict}
        temp['latents'] = self.data_array[idx]
        return temp



from DenoisingMLPFinal import DenoisingModelConditions

from util_diff import train_loop_diffusion

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput


SCALE_FACTOR = 1.0
# SCALE_FACTOR = 1.0/torch.std(latent_train_data).item()
print("SCALE_FACTOR:", SCALE_FACTOR)

train_dataset_diff = TensorDatasetConditionDict(latent_train_data*SCALE_FACTOR, label_train)

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

num_epochs = 2000
lr = 2e-4
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

control_adata_final = train_adata_final[train_adata_final.obs['stim'] == 'ctrl'].copy()
control_ori_temp = np.concatenate([
    to_dense_array(control_adata_final[control_adata_final.obs['cell.type'] == tc].X)
    for tc in test_cellnames
])

with torch.no_grad():
    control_latents_temp = latentmodel.encode(torch.FloatTensor(control_ori_temp).cuda())['latents'].cpu()
celltypelist_temp = []
for tc in test_cellnames:
    celltypelist_temp += [tc]*len(control_adata_final[control_adata_final.obs['cell.type'] == tc])
batch_temp = {'cell.type': torch.LongTensor([conditionskey_name2id['cell.type'][xx] for xx in celltypelist_temp]), 'stim': torch.LongTensor([conditionskey_name2id['stim']['stim']]*len(celltypelist_temp))}

celltypelist_temp_np = np.array(celltypelist_temp)


len(celltypelist_temp_np)
from util_plot import *
from util_inference_diff import inference_process_batch_conditions, inference_process_strength

strenth2expr = {}
strenth2result = {}
for strenth0 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    diff_result_st, latent_diff_steps_st = inference_process_strength(batch_temp, control_latents_temp, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True, strength = strenth0)

    with torch.no_grad():
        cf_pred_st = latentmodel.decode(diff_result_st.cuda()/SCALE_FACTOR)
    cf_expr_test_st = cf_pred_st['reconstruction_expr'].cpu()
    cf_expr_test_np_st = cf_expr_test_st.numpy()
    cf_expr_test_np_st = np.clip(cf_expr_test_np_st, a_min=0, a_max=None)
    temp_result = {}
    for tc in test_cellnames:
        temp_result[tc] = compute_metrics_single(gt_test[test_adata_final.obs['cell.type']==tc].mean(0), cf_expr_test_np_st[celltypelist_temp_np==tc].mean(0))
    strenth2expr[strenth0] = cf_expr_test_np_st
    strenth2result[strenth0] = pd.DataFrame(temp_result).T

inf_tempdf = pd.concat(strenth2result, axis=0).reset_index()
inf_tempdf.to_csv(os.path.join(modeloutputdir, "inference_strength_results.csv"), index=False)

control_obs_df = pd.concat([control_adata_final[control_adata_final.obs['cell.type'] == tc].obs.copy() for tc in test_cellnames])
control_obs_df['stim'] = 'stim'


for ii in [0.3, 0.4, 0.5]:
    cf_expr_test_np_st = strenth2expr[ii]
    AnnData_cf = anndata.AnnData(X=cf_expr_test_np_st, obs=control_obs_df, var=test_adata_final.var.copy())
    AnnData_cf.write_h5ad(os.path.join(modeloutputdir, f"inference_strength_{ii}_cf_based_on_control.h5ad"))
    strenth2result[ii].to_csv(os.path.join(modeloutputdir, f"inference_strength_{ii}_cf_metrics.csv"))


diff_result_0, latent_diff_steps_0 = inference_process_batch_conditions(batch_temp, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True,)


with torch.no_grad():
    cf_pred_0 = latentmodel.decode(diff_result_0.cuda()/SCALE_FACTOR)
cf_expr_test_0 = cf_pred_0['reconstruction_expr'].cpu()

cf_expr_test_np_0 = cf_expr_test_0.numpy()
cf_expr_test_np_0 = np.clip(cf_expr_test_np_0, a_min=0, a_max=None)
compute_metrics_single(gt_test[test_adata_final.obs['cell.type']==tc].mean(0), cf_expr_test_np_0.mean(0))



from util_plot import *
from util_inference_diff import inference_process_batch_conditions

print('construct test batch for inference, only condition inputs')

test_batch = {xx: torch.LongTensor(yy) for xx,yy in label_test_dict.items()}


diff_result, latent_diff_steps = inference_process_batch_conditions(test_batch, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True)
with torch.no_grad():
    cf_pred = latentmodel.decode(diff_result.cuda()/SCALE_FACTOR)
cf_expr_test = cf_pred['reconstruction_expr'].cpu()

cf_expr_test_np = cf_expr_test.numpy()
cf_expr_test_np = np.clip(cf_expr_test_np, a_min=0, a_max=None)

temp0 = test_adata_final.obs['cell.type']
unique00 = temp0.unique()
temp_result0 = {}
for xx in unique00:
    id00 = temp0==xx
    temp_result0[xx] = compute_metrics_single(gt_test[id00].mean(0), cf_expr_test_np[id00].mean(0))

temp_result0df = pd.DataFrame(temp_result0).T


train_batch = {xx: torch.LongTensor(yy) for xx,yy in label_train.items()}

diff_result, latent_diff_steps_train = inference_process_batch_conditions(train_batch, noise_scheduler, denoising_model, generator = None, latent_dim = denoising_setting_config['latent_dim'], output_latent_steps = True)
with torch.no_grad():
    cf_pred = latentmodel.decode(diff_result.cuda()/SCALE_FACTOR)
cf_expr_train = cf_pred['reconstruction_expr'].cpu()
cf_expr_train_np = cf_expr_train.numpy()
cf_expr_train_np = np.clip(cf_expr_train_np, a_min=0, a_max=None)

temp0 = train_adata_final.obs['cell.type']
unique00 = temp0.unique()
train_temp_result0 = {}
for xx in unique00:
    id00 = temp0==xx
    train_temp_result0[xx] = compute_metrics_single(gt_train[id00].mean(0), cf_expr_train_np[id00].mean(0))

traintemp_result0df = pd.DataFrame(train_temp_result0).T
temp_result0df.to_csv(os.path.join(modeloutputdir, "test_cf_metrics.csv"))
rctemp_result0df.to_csv(os.path.join(modeloutputdir, "test_rc_metrics_latentmodel.csv"))

test_cell_types = test_adata_final.obs['cell.type'].unique()
for ct in test_cell_types:
    id00 = test_adata_final.obs['cell.type'] == ct
    t1 = gt_test[id00].mean(0)
    t2 = cf_expr_test_np[id00].mean(0)
    t3 = rc_output_test[id00].mean(0)
    plt.figure(figsize = (4,4))
    plt.scatter(t1, t2, alpha=0.5)
    plt.scatter(t1, t3, alpha=0.5)
    plt.plot([t1.min(), t1.max()], [t1.min(), t1.max()], 'k--')
    plt.title(f'cell type: {ct} {sum(id00)}')
    plt.savefig(os.path.join(modeloutputdir, f"scatter_{ct}.png"))
    plt.show()




temp0 =test_adata_final
adatatmpp = anndata.AnnData(X=temp0.X.copy(), obs=temp0.obs.copy(), var=temp0.var.copy())

adatatmpp.obsm['cf_expr'] = cf_expr_test_np
adatatmpp.obsm['cf_latent'] = latent_diff_steps[-1].numpy()
for ii,step_idx in enumerate(range(1000, -1, -50)):
    adatatmpp.obsm[f'cf_latent_step{step_idx}'] = latent_diff_steps[ii].numpy()
temp = os.path.join(modeloutputdir, 'test_adata_final_diffU.h5ad')
adatatmpp.write_h5ad(temp)
print(f'saved test adata to {temp}')


temp0 =train_adata_final
adatatmpp = anndata.AnnData(X=temp0.X.copy(), obs=temp0.obs.copy(), var=temp0.var.copy())

adatatmpp.obsm['cf_expr'] = cf_expr_train_np
adatatmpp.obsm['cf_latent'] = latent_diff_steps_train[-1].numpy()
for ii,step_idx in enumerate(range(1000, -1, -50)):
    adatatmpp.obsm[f'cf_latent_step{step_idx}'] = latent_diff_steps_train[ii].numpy()

temp = os.path.join(modeloutputdir, 'train_adata_final_diffU.h5ad')
adatatmpp.write_h5ad(temp)
print(f'saved train adata to {temp}')


temp_result0df.to_csv(os.path.join(modeloutputdir, 'test_celltype_metrics_diffU.csv'))
traintemp_result0df.to_csv(os.path.join(modeloutputdir, 'train_celltype_metrics_diffU.csv'))

mergedX = np.concatenate([cf_expr_train_np, cf_expr_test_np, gt_train, gt_test], 0)
mergedobs = pd.concat([all_adata_final.obs.copy(),all_adata_final.obs.copy()])
mergedobs['cfflag'] = 'cf'
mergedobs['cfflag'][len(all_adata_final):] = 'gt'

merged_adata = anndata.AnnData(X=mergedX.copy(), obs=mergedobs,
var=all_adata_final.var.copy())

merged_adata.write_h5ad(os.path.join(modeloutputdir, 'merged_adata_cf_gt_diffU.h5ad'))
