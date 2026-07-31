import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MLPModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2048, hidden_dim: list = [2048, 2048],
                 dropout: float = 0.0):
        super().__init__()
        if len(hidden_dim) == 2:
            self.fc_layers = nn.Sequential(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim[0]),
                    nn.BatchNorm1d(hidden_dim[0], eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim[0], hidden_dim[1]),
                    nn.BatchNorm1d(hidden_dim[1], eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim[1], output_dim),
                    nn.BatchNorm1d(output_dim, eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        elif len(hidden_dim) ==1:
            self.fc_layers = nn.Sequential(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim[0]),
                    nn.BatchNorm1d(hidden_dim[0], eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ),
                nn.Sequential(
                    nn.Linear(hidden_dim[0], output_dim),
                    nn.BatchNorm1d(output_dim, eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        else:
            raise ValueError("hidden_dim should be a list of length 1 or 2.")

    def forward(self, x):
        return self.fc_layers(x)
    

from torch.distributions import NegativeBinomial
class CellEncoderWithLogvar(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        hidden_dim: int = 2048,
        dec_hidden: int = 4096,
        dropout: float = 0.0,
        use_variational: bool = True,
        kl_weight: float = 5e-4,   
        hidden_dim_en: list = [4096, 2048],
        hidden_dim_de: list = [1024, 2048],  
        eps: float = 0.001, 
        momentum: float = 0.01,
        use_dec_logvar: bool = False,
        recon_loss_type: str = "mse",
        distribution_type: str = "gauss",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_variational = use_variational 
        self.kl_weight = kl_weight
        self.use_dec_logvar = use_dec_logvar
        self.recon_loss_type = recon_loss_type
        self.distribution_type = distribution_type
        # use_variational = False, kl_weight >0: AE with KL loss
        # use_variational = True, kl_weight >0: VAE

        # Replace transformer with MLP encoder
        self.encoder = MLPModule(input_dim=input_dim, output_dim=hidden_dim, hidden_dim = hidden_dim_en, dropout = dropout,)
        
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----- Decoder core -----
        self.decoder_core = MLPModule(input_dim = latent_dim, output_dim=dec_hidden, hidden_dim = hidden_dim_de, dropout = dropout, )
        self.dec_mu = nn.Linear(dec_hidden, input_dim)
        if self.distribution_type == "nb":
            self.px_r = nn.Parameter(torch.randn(input_dim))  # gene-specific dispersion
            # self.px_r = nn.Parameter(torch.ones(input_dim) * 1.0)  # softplus(1)=~1.31 # or 2.0 => softplus(2)=~2.13
            self.use_dec_logvar = False  # logvar is determined by mu and r in NB case
            self.alpha_layer = nn.Linear(dec_hidden, 1)  # for modeling expression ratio (not normalized, cause use subset of genes for training)
        elif self.distribution_type == "gauss":
            if self.use_dec_logvar:
                self.dec_logvar = nn.Linear(dec_hidden, input_dim)
        else:
            raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")


    def _compute_var(self, raw_logvar: torch.Tensor):
        # var = torch.exp(raw_logvar) + 1e-4
        # logvar = torch.log(var)
        # return var, logvar

        raw_logvar = torch.clamp(raw_logvar, -6, 2)   # 强烈建议
        logvar_z = raw_logvar
        var_z = torch.exp(logvar_z) + 1e-4
        return var_z, logvar_z


    @staticmethod
    def _reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = var.sqrt()
        eps = torch.randn_like(std) if torch.is_grad_enabled() else torch.zeros_like(std)
        return mu + std * eps

    def encode(self, expr: torch.Tensor):
        h = self.encoder(expr)                  
        mu_z = self.enc_mu(h)                      
        raw_logvar = self.enc_logvar(h)

        var_z, logvar_z = self._compute_var(raw_logvar)
        
        if self.use_variational:
            z = self._reparameterize(mu_z, var_z)
        else:
            z = mu_z                               
        return {"latents": z, "z": z, "mu_z": mu_z, "logvar_z": logvar_z, "var_z": var_z}

    def decode(self, latents: torch.Tensor):
        h = self.decoder_core(latents)                   
        recon_mu = self.dec_mu(h)   
        recon_logvar = None
        px_rate = None
        px_scale = None
        alpha = None
        reconstruction_expr = recon_mu
        if self.distribution_type == "nb":
            alpha = self.alpha_layer(h).sigmoid().clamp(1e-4, 1-1e-4)  # modeling expression ratio (not normalized, cause use subset of genes for training)
            px_scale = F.softmax(recon_mu, dim=-1) * alpha
            # px_scale = F.softplus(recon_mu) + 1e-8 # modeling expression ratio (not normalized, cause use subset of genes for training)
            reconstruction_expr = px_scale
            log_pred = torch.log1p(px_scale * 1e4)
        else:
            log_pred = recon_mu
        if self.use_dec_logvar:
            recon_logvar = self.dec_logvar(h)               
        return {"recon_mu": recon_mu, 'recon_logvar': recon_logvar, 'reconstruction_expr': reconstruction_expr, "px_scale": px_scale, 'log_pred': log_pred, "alpha": alpha}


    def forward(self, expr: torch.Tensor, library: torch.Tensor = None):
        enc = self.encode(expr)
        z, mu_z, logvar_z, var_z = enc["z"], enc["mu_z"], enc["logvar_z"], enc['var_z']

        dec = self.decode(z)
        recon_mu = dec["recon_mu"]
        recon_logvar = dec['recon_logvar']
        px_scale = dec["px_scale"]
        px_rate = None
        alpha = dec["alpha"]

        loss_alpha = 0.0
        
        if self.distribution_type == "nb":  
            if library is None:
                library = expr.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            theta = (F.softplus(self.px_r) + 1e-8).unsqueeze(0) 
            px_rate = (library * px_scale).clamp(min=1e-8)
            alpha_gt = expr.sum(dim=-1).clamp(min=1e-8).view(-1)/ library.view(-1)
            loss_alpha = F.mse_loss(alpha_gt, alpha.view(-1))


            # probs = theta / (theta + px_rate)
            # px = NegativeBinomial(total_count=theta, probs=probs)      
            # print("px_scale min/max:", px_scale.min().item(), px_scale.max().item())
            # print("library min:", library.min().item())
            # print("px_rate min:", px_rate.min().item())
            # print("theta min:", theta.min().item())
            # print("probs min/max:", probs.min().item(), probs.max().item())
            # print('----')
            logits = torch.log(px_rate) - torch.log(theta)
            # logits = torch.log(theta) - torch.log(px_rate) # wrong
            logits = torch.clamp(logits, min=-20, max=20)
            
            px = NegativeBinomial(total_count=theta, logits=logits)      
            recon_loss = -px.log_prob(expr).sum(dim=-1).mean()
            
        elif self.distribution_type == "gauss":
            if self.use_dec_logvar:
                recon_logvar = torch.clamp(recon_logvar, -6, 2)
                recon_var = torch.exp(recon_logvar)
                recon_loss = ((expr - recon_mu)**2 / recon_var + recon_logvar).mean()
            else:
                if self.recon_loss_type == "smoothl1":
                    recon_loss = F.smooth_l1_loss(recon_mu, expr)
                elif self.recon_loss_type == 'l1':
                    recon_loss = F.l1_loss(recon_mu, expr)
                elif self.recon_loss_type == "mse":
                    recon_loss = F.mse_loss(recon_mu, expr, reduction="mean")
                else:
                    raise ValueError(f"Unsupported recon_loss_type: {self.recon_loss_type}")
            

        if self.kl_weight > 0.0:
            # kl_loss = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - var_z)
            kl_per_cell = -0.5 * (1 + logvar_z - mu_z.pow(2) - var_z).sum(dim=1)
            kl_loss = kl_per_cell.mean()
        else:
            kl_loss = torch.zeros((), device=expr.device, dtype=expr.dtype)

        if self.kl_weight > 0.0:
            loss = recon_loss + self.kl_weight * kl_loss + 0.05 * loss_alpha
        else:
            loss = recon_loss + 0.05 * loss_alpha
        
        if self.distribution_type == "gauss":
            mse_loss = F.mse_loss(recon_mu, expr, reduction="mean")
        elif self.distribution_type == "nb":
            mse_loss = F.mse_loss(px_rate, expr, reduction="mean")
        else:
            mse_loss = 0.0

        if np.isnan(loss.item()):
            print(kl_loss)
            print(recon_loss)
            raise ValueError('loss is nan')
            
        out = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z,           
            "recon_mu": recon_mu,
            "recon_logvar": recon_logvar,
            "mse_loss": mse_loss,
            "px_rate": px_rate,
            "px_scale": px_scale,
        }
        return out
