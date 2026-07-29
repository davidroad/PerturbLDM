import torch
from torch.utils.data import Dataset
import time
# class TensorDataset(Dataset):
#     def __init__(self, data):
#         try:
#             self.data = data.toarray()
#         except:
#             self.data = data
#     def __len__(self):
#         return self.data.shape[0]
#     def __getitem__(self, idx):
#         temp = self.data[idx]#.toarray()
#         return {"expr": temp[0]}

from torch.utils.data import DataLoader, Dataset

class TensorDataset(Dataset):
    def __init__(self, data):
        try:
            self.data = data.toarray()
        except:
            self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        temp = self.data[idx]
        return {"expr": temp}

import json
import os, io, random, torch, numpy as np
import math
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.utils as nn_utils
    

def train_loop_latent(model, train_loader,
    epochs: int | None = None,
    max_steps: int | None = None,         
    lr: float = 1e-4,
    weight_decay: float = 1e-8,
    device: str | None = None,
    log_step: int = 2000,
    save_step: int = 1000,
    save_dir: str = './',
    resume_dir: str | None = None,
):
    steps_per_epoch = len(train_loader)
    if max_steps is None:
        assert epochs is not None, "Set either epochs or max_steps"
        max_steps = epochs * steps_per_epoch
    if epochs is None:
        epochs = math.ceil(max_steps/steps_per_epoch)
    total_update_steps = max_steps
    opt       = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_update_steps)
    history, update_step, running_loss = [], 0, 0.0
    kl_running_loss = 0.0
    rc_running_loss = 0.0
    mse_running_loss = 0.0
    t0 = time.time(); end_flag = False
    n_loss = 0
    model.train()
    epoch_id = 0
    while update_step < total_update_steps:
        for batch_idx, batch in enumerate(train_loader, 1):
            model_output   =  model(**{k: v.to(device) for k, v in batch.items()})
            loss = model_output['loss']
            loss.backward()
            # check_grads(model)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            running_loss += loss.item()
            kl_running_loss += model_output['kl_loss'].item()
            rc_running_loss += model_output['recon_loss'].item()
            mse_running_loss += model_output['mse_loss'].item()
            opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
            update_step += 1
            if update_step % log_step == 0:
                train_loss = running_loss/log_step
                row = {"step": update_step, "train_loss": round(train_loss, 6),
                      "kl_loss": round(kl_running_loss/log_step, 6),
                      "recon_loss": round(rc_running_loss/log_step, 6),
                      'mse_loss': round(mse_running_loss/log_step, 6)}
                running_loss = 0.0 
                kl_running_loss = 0.0
                rc_running_loss = 0.0
                mse_running_loss = 0.0
                print(str(row))
                history.append(row)
            if update_step >= total_update_steps:
                break
        epoch_id += 1
            
            
    elapsed = time.time() - t0
    time_log = f"Finished {update_step} update steps ({epochs} epochs) in {elapsed/60:.1f} min."
    print(time_log)
    
    return({"metrics": history,  "log": time_log})

