import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoidal_embeddings(timesteps: torch.LongTensor, embedding_dim: int) -> torch.FloatTensor:
    device = timesteps.device
    half_dim = embedding_dim // 2
    exponents = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
    freqs = torch.exp(-math.log(10000.0) * exponents)
    emb = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  
    emb = torch.cat([emb.sin(), emb.cos()], dim=1) 
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, dropout: float = 0.,
                  use_post_norm: bool = True,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.use_post_norm = use_post_norm
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        z = self.mlp(x + context )  
        z = self.dropout(z)
        if self.use_residual:
            out = self.post_norm(x + z) if self.use_post_norm else x + z
        else:
            out = self.post_norm(z) if self.use_post_norm else z
        return out

import numpy as np

class DenoisingModelDrugDose(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        drug_vocab_size: int,
        ctrl_dim: int,
        hidden_dim: int = 3072,
        num_layers: int = 20,
        context_dim: int = 2560,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
        use_post_norm: bool = True,
        use_residual: bool = True,
        feature_num: int = 3,
        use_dose_level_mapping: bool = False,
        dose_level_mapping_hidden_dim: int = 128, 
        dose_level_mapping_dim: int = 2,
        drug_emb_pretrained: None = None,
        freeze_drug_emb: bool = True,
    ):
        super().__init__()
        # 1. Input context projections
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim
        if drug_emb_pretrained is None:
            self.drug_emb_dim = context_dim
            self.drug_emb = nn.Embedding(drug_vocab_size, self.drug_emb_dim)
        else:
            if isinstance(drug_emb_pretrained, np.ndarray):
                drug_emb_pretrained = torch.tensor(drug_emb_pretrained, dtype=torch.float32)
            elif not isinstance(drug_emb_pretrained, torch.Tensor):
                raise ValueError("drug_emb_pretrained must be a numpy array or torch Tensor.")
            self.drug_emb = nn.Embedding.from_pretrained(drug_emb_pretrained, freeze=freeze_drug_emb)
            self.drug_emb_dim = drug_emb_pretrained.shape[1]
            self.freeze_drug_emb = freeze_drug_emb

        self.ctrl_proj = nn.Linear(ctrl_dim, context_dim)
        self.time_emb_proj = nn.Linear(time_emb_dim, context_dim)
        if use_dose_level_mapping:
            if dose_level_mapping_dim > 0:
                self.dose_level_mapping = nn.Sequential(
                    nn.Linear(1, dose_level_mapping_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(dose_level_mapping_hidden_dim, dose_level_mapping_dim),
                )
                self.dose_level_proj = nn.Linear(dose_level_mapping_dim, context_dim)
            else:
                self.dose_level_mapping = nn.Sequential(
                    nn.Linear(1, dose_level_mapping_hidden_dim),
                    nn.ReLU(),
                )
                self.dose_level_proj = nn.Linear(dose_level_mapping_hidden_dim, context_dim)
        else:
            self.dose_level_mapping = nn.Identity()
            self.dose_level_proj = nn.Linear(1, context_dim)

        # 3. Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        self.feature_num = feature_num
        
        self.context_encoder= nn.Sequential(
            nn.Linear(context_dim*(feature_num) + self.drug_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.res_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout=dropout,
                                 use_post_norm = use_post_norm,
                                 use_residual = use_residual)
            for _ in range(num_layers)
        ])

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

    def get_context_feats(
        self,
        timesteps: torch.LongTensor,   
        ctrl_expr: torch.FloatTensor|None,   
        drug_id: torch.LongTensor|None,    
        dose_uM_level: torch.FloatTensor |None,
        drug_emb: torch.FloatTensor|None = None,
    ) -> torch.Tensor:
        
        drug_e = self.drug_emb(drug_id) if drug_id is not None else drug_emb
        ctrl_e = self.ctrl_proj(ctrl_expr) if ctrl_expr is not None else None
        dose_uM_e = self.dose_level_proj(self.dose_level_mapping(dose_uM_level.unsqueeze(-1))) if dose_uM_level is not None else None
        t_emb = self.time_emb_proj(get_sinusoidal_embeddings(timesteps, self.time_emb_dim))
        temp = [drug_e, ctrl_e, dose_uM_e, t_emb] 
        return([xx for xx in temp if xx is not None])


    def forward(
        self,
        latents: torch.FloatTensor,  
        timesteps: torch.LongTensor,   
        ctrl_expr: Optional[torch.FloatTensor] = None,  
        drug_id: Optional[torch.LongTensor] = None,    
        dose_uM_level: Optional[torch.FloatTensor] = None,
        drug_emb: torch.FloatTensor|None = None,
    ):
        if drug_emb is None and drug_id is None:
            raise ValueError("Either drug_id or drug_emb must be provided.")
        x = self.latent_proj(latents)  
        context_feats = self.get_context_feats(timesteps, ctrl_expr, drug_id, dose_uM_level, drug_emb) 
        context_feats = torch.concat(context_feats, dim=1)  
        context_emb = self.context_encoder(context_feats)

        for block in self.res_blocks:
            x = block(x, context_emb)     
        out = self.out_head(x)        
        return {
            "predict_output": out    
        }



import torch
import torch.nn as nn


class DenoisingModelConditions(nn.Module):
    ### must specify condition_setting_dict as {'key': (type, dim)} , type is 'categorical' or 'continuous', dim is dimension of the condition
    '''
    A flexible denoising model that can handle various types of conditions specified in condition_setting_dict.
    For example, condition_setting_dict = {'dose': ('continuous', 1), 'drug': ('categorical', 500)}
    dataloader should provide inputs accordingly.
    Each sample in training dataloader should provide:
    {keys in condition_setting_dict: their values}
    In inference, provide the same keys as in condition_setting_dict.
    '''
    def __init__(
        self,
        latent_dim: int,
        condition_setting_dict: dict, # {'key': (type, dim)}
        hidden_dim: int = 256,
        num_layers: int = 5,
        context_dim: int = 128,
        time_emb_dim: int = 32,
        dropout: float = 0.0,
        use_post_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        # 1. Input context projections
        self.context_dim = context_dim
        self.time_emb_dim = time_emb_dim
        self.condition_setting_dict = condition_setting_dict
        self.time_emb_proj = nn.Linear(time_emb_dim, context_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        feature_num = 0

        for key, (cond_type, cond_dim) in condition_setting_dict.items():
            feature_num +=1
            if cond_type == 'categorical':
                emb = nn.Embedding(cond_dim, context_dim)
                setattr(self, f'{key}_emb', emb)
            elif cond_type == 'continuous':
                proj = nn.Linear(cond_dim, context_dim)
                if cond_dim == 1:
                    proj = nn.Sequential(
                        nn.Linear(cond_dim, context_dim//4),
                        nn.SiLU(),
                        nn.Linear(context_dim//4, context_dim),
                    )
                setattr(self, f'{key}_proj', proj)
            else:
                raise ValueError(f'Unknown condition type: {cond_type}, use categorical or continuous')
        
        self.context_encoder= nn.Sequential(
            nn.Linear(context_dim*(feature_num+1), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.res_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout=dropout,
                                 use_post_norm = use_post_norm,
                                 use_residual = use_residual)
            for _ in range(num_layers)
        ])

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )
    

    def get_context_feats(
        self,
        timesteps: torch.LongTensor,   
        **condition_inputs,
    ) -> torch.Tensor:
        feats = []
        for key in self.condition_setting_dict.keys():
            if key not in condition_inputs:
                raise ValueError(f'Missing condition input for key: {key}')
            layer = getattr(self, f'{key}_emb' if self.condition_setting_dict[key][0]=='categorical' else f'{key}_proj')
            temp = layer(condition_inputs[key])
            feats.append(temp)
        t_emb = self.time_emb_proj(get_sinusoidal_embeddings(timesteps, self.time_emb_dim))
        temp = feats + [t_emb] 
        return([xx for xx in temp if xx is not None])


    def forward(
        self,
        latents: torch.FloatTensor,  
        timesteps: torch.LongTensor,   
        **condition_inputs,
    ):
        x = self.latent_proj(latents)  
        context_feats = self.get_context_feats(timesteps, **condition_inputs) 
        context_feats = torch.concat(context_feats, dim=1)  
        context_emb = self.context_encoder(context_feats)
        for block in self.res_blocks:
            x = block(x, context_emb)     
        out = self.out_head(x)        
        return {
            "predict_output": out    
        }
    