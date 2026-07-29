import random
from collections import defaultdict
import torch
import numpy as np
import os
import json
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import Dataset


class Condition_IDConverter:
    def __init__(self, condname_list):
        drug2id = {}
        cell2id = {}
        dose2id = {}
        # condid2sepids = {}
        self.condname2outcome = {}
        
        # for cond, condid in cond2id.items():
        for cond in condname_list:
            drugname, dosename, cellname = cond.split('___')
            if drugname not in drug2id:
                drug2id[drugname] = len(drug2id)
            if cellname not in cell2id:
                cell2id[cellname] = len(cell2id)
            if dosename not in dose2id:
                dose2id[dosename] = len(dose2id)
        
        self.id2dose = {v:k for k,v in dose2id.items()}
        self.id2cell = {v:k for k,v in cell2id.items()}
        self.id2drug = {v:k for k,v in drug2id.items()}
        self.drug2id = drug2id
        self.cell2id = cell2id
        self.dose2id = dose2id
        
        self.sepid_labelnum = [len(drug2id), len(dose2id), len(cell2id)]
        print(self.sepid_labelnum)
        
    def _process_cond(self, cond):
        if cond in self.condname2outcome:
            return(self.condname2outcome[cond])
        drugname, dosename, cellname = cond.split('___')
        try:
            temp = {"drug_id": self.drug2id[drugname], "dose_id": self.dose2id.get(dosename, None), "cell_id": self.cell2id[cellname], "dose_uM": np.float32(dosename),}
        except:
            raise ValueError(f"Can not find ids of {cond}")
            
        self.condname2outcome[cond] = temp
        return(temp)
        
    def process_conds(self, cond):
        if isinstance(cond, list):
            result = [self._process_cond(item) for item in cond]
            return(result)
        else:
            return([self._process_cond(cond)])


class LatentDiffusionCachedDataset(Dataset):
    def __init__(self, condname_list, condconvertor, cell2exprmean, controllatents= None, latents=None):
        self.condname_list = condname_list
        self.controllatents = controllatents
        self.latents = latents
        self.condconvertor = condconvertor
        self.cell2exprmean = cell2exprmean

    def __len__(self):
        return len(self.condname_list)

    def __getitem__(self, index):
        condname = self.condname_list[index]
        ctrl_latent = self.cell2exprmean[condname.split("___")[2]]
        cond = self.condconvertor._process_cond(condname)
        temp = {
            "ctrl_expr": ctrl_latent,
        }
        if self.latents is not None:
            temp["latents"] = self.latents[index]
        temp.update({kk:cond[kk] for kk in ["drug_id", ]})
        # temp["dose_uM"] = np.float32(cond["dose_uM"])
        temp["dose_uM_level"] = np.float32(np.log10(np.float32(cond["dose_uM"])/0.05))
        return(temp)



def save_all_trained_result(modeloutputdir, config_dict, config_dict_train, log_history, denoising_model):
    os.makedirs(modeloutputdir, exist_ok=True)
    denoising_model.cpu()
    modelpath00 = os.path.join(modeloutputdir, "model_weights.pth")
    if os.path.isfile(modelpath00):
        raise ValueError('file exists!')
        
    torch.save(denoising_model.state_dict(), modelpath00)
    with open(os.path.join(modeloutputdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    with open(os.path.join(modeloutputdir, "config_train.json"), "w") as f:
        json.dump(config_dict_train, f, indent=4)
    
    with open(os.path.join(modeloutputdir, "log_history.json"), "w") as f:
        json.dump(log_history, f, indent=4)
        
    print('save all files.')



import torch
import torch.nn.functional as F
import time
import numpy as np
import math



def train_loop_diffusion(denoising_model, traindataloader, noise_scheduler,num_epochs = 10, lr = 1e-4, eval_steps = 1000, prediction_type = 'v_prediction', weight_decay = 0.0):
    st_time = time.time()
    optimizer      = torch.optim.AdamW(denoising_model.parameters(), lr=lr, weight_decay=weight_decay)
    all_loss_log = []
    eval_steps = eval_steps
    loss_temp = []
    device = next(denoising_model.parameters()).device

    num_training_steps = len(traindataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    denoising_model.train()
    for epoch in range(num_epochs):
        for ii, batch in enumerate(traindataloader):
            latent_origin = batch["latents"]                      
            bsz, *_ = latent_origin.shape
            timesteps = torch.randint(0, 1000, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latent_origin)
            noisy_latent = noise_scheduler.add_noise(latent_origin, noise, timesteps)
            if prediction_type == 'v_prediction':
                v_source = noise_scheduler.get_velocity(latent_origin, noise, timesteps)
            elif prediction_type == 'epsilon':
                v_source = noise
            elif prediction_type == 'sample':
                v_source = latent_origin
            else:
                raise ValueError('Unknown prediction_type')
            v_pred = denoising_model(latents=noisy_latent.to(device),timesteps=timesteps,
                **{kk: vv.to(device) for kk, vv in batch.items() if kk != "latents"},
            )['predict_output']
            loss = F.mse_loss(v_pred, v_source.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_temp.append(loss.item())  # .item() to reduce memory usage
            if ii % eval_steps == 0:
                loss_mean = torch.mean(torch.tensor(loss_temp))
                current_lr = scheduler.get_last_lr()[0]
                print(f"{ii}: loss {loss_mean.item():.4f}")
                all_loss_log.append(loss_mean.item())
                loss_temp = []
        print(f"Epoch {epoch:02d}  Final batch loss: {loss.item():.4f}")
    ed_time = time.time()
    print('time: ', ed_time - st_time)
    return(all_loss_log)


