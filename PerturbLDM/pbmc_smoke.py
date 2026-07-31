"""CPU-safe end-to-end diffusion example on the Kang et al. PBMC dataset.

This standalone example validates installation, preprocessing, a biologically
structured response-transfer split, latent-model optimization, conditional
diffusion optimization and generation. Its deliberately reduced feature and
cell counts are not intended to reproduce manuscript results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader, TensorDataset

from .DenoisingMLPFinal import DenoisingModelConditions
from .LatentModelFinal import CellEncoderWithLogvar


HELD_OUT_CELL_TYPES = (
    "B cells",
    "CD8 T cells",
    "FCGR3A+ Monocytes",
)
REQUIRED_OBS_COLUMNS = ("cell.type", "stim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a five-minute end-to-end PerturbLDM example on a PBMC H5AD."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--genes", type=int, default=1000)
    parser.add_argument(
        "--cells-per-condition",
        type=int,
        default=0,
        help="Maximum cells per condition; 0 uses all available cells",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--diffusion-epochs", type=int, default=160)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, threads))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def stratified_sample_indices(
    obs: Any,
    *,
    group_columns: tuple[str, ...],
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    grouped = obs.reset_index(drop=True).groupby(
        list(group_columns), observed=True, sort=True
    )
    for _, frame in grouped:
        indices = frame.index.to_numpy()
        take = min(maximum, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def write_diagnostic_plots(
    output_dir: Path,
    *,
    vae_history: list[float],
    diffusion_history: list[float],
    condition_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, str]:
    """Write example-only training and prediction diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_path = output_dir / "training_losses.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(
        np.arange(1, len(vae_history) + 1),
        vae_history,
        color="#2878B5",
        linewidth=1.8,
    )
    axes[0].set(title="Latent model", xlabel="Epoch", ylabel="Training objective")
    axes[1].plot(
        np.arange(1, len(diffusion_history) + 1),
        diffusion_history,
        color="#C82423",
        linewidth=1.8,
    )
    axes[1].set(
        title="Conditional diffusion",
        xlabel="Epoch",
        ylabel="Noise-prediction MSE",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Standalone PBMC example: training diagnostics", fontsize=11)
    fig.savefig(loss_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    prediction_path = output_dir / "prediction_diagnostics.png"
    cell_types = list(condition_arrays)
    fig, axes = plt.subplots(
        2,
        len(cell_types),
        figsize=(3.25 * len(cell_types), 6.0),
        constrained_layout=True,
        squeeze=False,
    )
    for column, cell_type in enumerate(cell_types):
        values = condition_arrays[cell_type]
        pairs = (
            (values["observed_mean"], values["predicted_mean"], "Whole state"),
            (
                values["observed_effect"],
                values["predicted_effect"],
                "Matched-control effect",
            ),
        )
        for row, (observed, predicted, label) in enumerate(pairs):
            axis = axes[row, column]
            axis.scatter(observed, predicted, s=6, alpha=0.35, color="#2878B5")
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            if lower == upper:
                upper = lower + 1.0
            axis.plot(
                [lower, upper],
                [lower, upper],
                "--",
                color="0.35",
                linewidth=1,
            )
            correlation = safe_pearson(observed, predicted)
            correlation_text = "NA" if correlation is None else f"{correlation:.3f}"
            axis.set_title(f"{cell_type}\n{label}; r={correlation_text}", fontsize=9)
            axis.set_xlabel("Observed")
            axis.set_ylabel("Predicted")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.14, linewidth=0.5)
    fig.suptitle("Standalone PBMC example: condition-mean diagnostics", fontsize=11)
    fig.savefig(prediction_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "training_losses": loss_path.name,
        "prediction_diagnostics": prediction_path.name,
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def write_run_artifacts(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    genes: list[str],
    cell_to_id: dict[str, int],
    stim_to_id: dict[str, int],
    latent_model: CellEncoderWithLogvar,
    denoiser: DenoisingModelConditions,
    scheduler: DDPMScheduler,
    vae_history: list[float],
    diffusion_history: list[float],
    metrics: dict[str, dict[str, float | None]],
    condition_arrays: dict[str, dict[str, np.ndarray]],
    predicted: np.ndarray,
    observed: np.ndarray,
    test_cell_type: np.ndarray,
    test_stim: np.ndarray,
) -> dict[str, str]:
    """Persist every object needed to inspect or reuse one example run."""
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "epoch", "loss", "value"))
        writer.writeheader()
        for epoch, value in enumerate(vae_history, start=1):
            writer.writerow(
                {
                    "stage": "latent_model",
                    "epoch": epoch,
                    "loss": "training_objective",
                    "value": value,
                }
            )
        for epoch, value in enumerate(diffusion_history, start=1):
            writer.writerow(
                {
                    "stage": "conditional_diffusion",
                    "epoch": epoch,
                    "loss": "noise_prediction_mse",
                    "value": value,
                }
            )

    metrics_path = output_dir / "condition_metrics.csv"
    metric_names = (
        "absolute_profile_pearson",
        "absolute_profile_mae",
        "matched_control_effect_pearson",
        "matched_control_effect_mae",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cell_type",
                "evaluation_unit",
                "effect_reference",
                *metric_names,
            ),
        )
        writer.writeheader()
        for cell_type, values in metrics.items():
            writer.writerow(
                {
                    "cell_type": cell_type,
                    "evaluation_unit": "condition_mean_over_selected_genes",
                    "effect_reference": "mean_ctrl_same_cell_type",
                    **{name: values[name] for name in metric_names},
                }
            )

    genes_path = output_dir / "selected_genes.txt"
    genes_path.write_text("\n".join(genes) + "\n", encoding="utf-8")

    condition_path = output_dir / "condition_mean_profiles.npz"
    cell_type_order = list(condition_arrays)
    np.savez_compressed(
        condition_path,
        genes=np.asarray(genes, dtype=str),
        cell_types=np.asarray(cell_type_order, dtype=str),
        observed_mean=np.stack(
            [condition_arrays[name]["observed_mean"] for name in cell_type_order]
        ),
        predicted_mean=np.stack(
            [condition_arrays[name]["predicted_mean"] for name in cell_type_order]
        ),
        observed_effect=np.stack(
            [condition_arrays[name]["observed_effect"] for name in cell_type_order]
        ),
        predicted_effect=np.stack(
            [condition_arrays[name]["predicted_effect"] for name in cell_type_order]
        ),
    )

    predictions_path = output_dir / "heldout_predictions.npz"
    np.savez_compressed(
        predictions_path,
        genes=np.asarray(genes, dtype=str),
        cell_type=np.asarray(test_cell_type, dtype=str),
        stim=np.asarray(test_stim, dtype=str),
        observed_expression=observed.astype(np.float32, copy=False),
        predicted_expression=predicted.astype(np.float32, copy=False),
    )

    latent_checkpoint = output_dir / "latent_model_state.pt"
    denoiser_checkpoint = output_dir / "denoising_model_state.pt"
    torch.save(cpu_state_dict(latent_model), latent_checkpoint)
    torch.save(cpu_state_dict(denoiser), denoiser_checkpoint)

    config_path = output_dir / "run_configuration.json"
    configuration = {
        "seed": args.seed,
        "device": str(next(latent_model.parameters()).device),
        "features": {
            "selection": "training_only_hvg",
            "count": len(genes),
            "gene_file": genes_path.name,
        },
        "sampling": {
            "cells_per_condition": args.cells_per_condition,
            "zero_means_all_available": True,
        },
        "latent_model": {
            "latent_dim": args.latent_dim,
            "epochs": args.vae_epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "kl_weight": 1e-4,
            "checkpoint": latent_checkpoint.name,
        },
        "conditional_diffusion": {
            "epochs": args.diffusion_epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "prediction_type": "epsilon",
            "training_timesteps": int(scheduler.config.num_train_timesteps),
            "inference_steps": args.inference_steps,
            "beta_start": float(scheduler.config.beta_start),
            "beta_end": float(scheduler.config.beta_end),
            "checkpoint": denoiser_checkpoint.name,
        },
        "condition_mappings": {
            "cell_type": cell_to_id,
            "stim": stim_to_id,
        },
    }
    config_path.write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "training_history": history_path.name,
        "condition_metrics": metrics_path.name,
        "selected_genes": genes_path.name,
        "condition_mean_profiles": condition_path.name,
        "heldout_predictions": predictions_path.name,
        "latent_model_checkpoint": latent_checkpoint.name,
        "denoising_model_checkpoint": denoiser_checkpoint.name,
        "run_configuration": config_path.name,
    }


def train_latent_model(
    train_x: np.ndarray,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[CellEncoderWithLogvar, list[float], torch.Tensor]:
    model = CellEncoderWithLogvar(
        latent_dim=latent_dim,
        input_dim=train_x.shape[1],
        hidden_dim=64,
        dec_hidden=64,
        dropout=0.1,
        use_variational=True,
        kl_weight=1e-4,
        hidden_dim_en=[128],
        hidden_dim_de=[64],
        use_dec_logvar=False,
        recon_loss_type="mse",
        distribution_type="gauss",
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        for (batch,) in loader:
            output = model(batch.to(device))
            loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        train_latents = model.encode(torch.from_numpy(train_x).to(device))["mu_z"]
    return model, history, train_latents.detach().cpu()


def train_denoiser(
    train_latents: torch.Tensor,
    cell_ids: torch.Tensor,
    stim_ids: torch.Tensor,
    *,
    cell_type_count: int,
    stim_count: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DenoisingModelConditions, DDPMScheduler, list[float]]:
    model = DenoisingModelConditions(
        latent_dim=train_latents.shape[1],
        condition_setting_dict={
            "cell_type": ("categorical", cell_type_count),
            "stim": ("categorical", stim_count),
        },
        hidden_dim=64,
        num_layers=2,
        context_dim=32,
        time_emb_dim=16,
        dropout=0.1,
        use_post_norm=True,
        use_residual=True,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=50,
        prediction_type="epsilon",
        beta_start=0.00085,
        beta_end=0.015,
    )
    loader = DataLoader(
        TensorDataset(train_latents, cell_ids, stim_ids),
        batch_size=min(batch_size, len(train_latents)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        for latents, batch_cell_ids, batch_stim_ids in loader:
            latents = latents.to(device)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (len(latents),),
                device=device,
                dtype=torch.long,
            )
            noisy = scheduler.add_noise(latents, noise, timesteps)
            prediction = model(
                latents=noisy,
                timesteps=timesteps,
                cell_type=batch_cell_ids.to(device),
                stim=batch_stim_ids.to(device),
            )["predict_output"]
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return model, scheduler, history


def generate(
    denoiser: DenoisingModelConditions,
    scheduler: DDPMScheduler,
    decoder: CellEncoderWithLogvar,
    cell_ids: torch.Tensor,
    stim_ids: torch.Tensor,
    *,
    latent_dim: int,
    inference_steps: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(inference_steps, device=device)
    latents = torch.randn(
        len(cell_ids), latent_dim, generator=generator, device=device
    )
    denoiser.eval()
    decoder.eval()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            timesteps = timestep.repeat(len(latents)).to(device)
            prediction = denoiser(
                latents=latents,
                timesteps=timesteps,
                cell_type=cell_ids.to(device),
                stim=stim_ids.to(device),
            )["predict_output"]
            latents = scheduler.step(
                prediction, timestep, latents, generator=generator
            ).prev_sample
        expression = decoder.decode(latents)["reconstruction_expr"].clamp(min=0)
    return expression.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    args = parse_args()
    started = time.time()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"PBMC input does not exist: {input_path}")
    if min(
        args.genes,
        args.latent_dim,
        args.vae_epochs,
        args.diffusion_epochs,
        args.inference_steps,
        args.batch_size,
        args.threads,
    ) < 1:
        raise ValueError("Model, optimization and resource settings must be positive")
    if args.cells_per_condition < 0:
        raise ValueError("--cells-per-condition must be zero or positive")

    # 1. Validate the input contract and reproduce the biological hold-out.
    # Only the stimulated states are hidden; controls from the same cell types
    # remain available, so this is response transfer rather than prediction of
    # a completely unseen cell type.
    seed_everything(args.seed, args.threads)
    device = resolve_device(args.device)
    adata = sc.read_h5ad(input_path)
    missing = [column for column in REQUIRED_OBS_COLUMNS if column not in adata.obs]
    if missing:
        raise ValueError(f"PBMC input is missing required obs columns: {missing}")
    initial_shape = [int(adata.n_obs), int(adata.n_vars)]

    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata, min_cells=max(1, int(0.01 * adata.n_obs)))
    adata = adata[adata.obs["cell.type"] != "Megakaryocytes"].copy()

    held_out = adata.obs["cell.type"].isin(HELD_OUT_CELL_TYPES) & (
        adata.obs["stim"] == "stim"
    )
    train_full = adata[~held_out].copy()
    test_full = adata[held_out].copy()
    if test_full.n_obs == 0:
        raise ValueError("The documented held-out PBMC states were not found")
    for cell_type in HELD_OUT_CELL_TYPES:
        control_count = int(
            (
                (train_full.obs["cell.type"] == cell_type)
                & (train_full.obs["stim"] == "ctrl")
            ).sum()
        )
        test_count = int((test_full.obs["cell.type"] == cell_type).sum())
        if control_count == 0 or test_count == 0:
            raise ValueError(
                f"PBMC split lacks control or held-out cells for {cell_type}"
            )

    # 2. Keep a condition-stratified subset so the complete example remains
    # CPU-safe and comfortably below its five-minute runtime target.
    if args.cells_per_condition == 0:
        train = train_full
        test = test_full
        sampling_rule = "all_available_cells"
    else:
        train_indices = stratified_sample_indices(
            train_full.obs,
            group_columns=("cell.type", "stim"),
            maximum=args.cells_per_condition,
            seed=args.seed,
        )
        test_indices = stratified_sample_indices(
            test_full.obs,
            group_columns=("cell.type",),
            maximum=args.cells_per_condition,
            seed=args.seed + 1,
        )
        train = train_full[train_indices].copy()
        test = test_full[test_indices].copy()
        sampling_rule = f"up_to_{args.cells_per_condition}_cells_per_condition"

    # 3. Select features from the sampled training cells only, then preserve
    # exactly the same gene order in the held-out cells.
    gene_count = min(args.genes, train.n_vars)
    sc.pp.highly_variable_genes(train, n_top_genes=gene_count, flavor="seurat")
    genes = train.var_names[train.var["highly_variable"]].tolist()
    if len(genes) != gene_count:
        raise RuntimeError(
            f"Expected {gene_count} training-selected HVGs, observed {len(genes)}"
        )
    train = train[:, genes].copy()
    test = test[:, genes].copy()
    train_x = to_dense_float32(train.X)
    test_x = to_dense_float32(test.X)

    cell_types = sorted(train.obs["cell.type"].astype(str).unique().tolist())
    stim_values = sorted(train.obs["stim"].astype(str).unique().tolist())
    cell_to_id = {name: index for index, name in enumerate(cell_types)}
    stim_to_id = {name: index for index, name in enumerate(stim_values)}
    if not set(test.obs["cell.type"].astype(str)).issubset(cell_to_id):
        raise ValueError("A held-out cell type is absent from the training controls")
    if "stim" not in stim_to_id:
        raise ValueError("The stimulated condition is absent from training")

    train_cell_ids = torch.tensor(
        train.obs["cell.type"].astype(str).map(cell_to_id).to_numpy(),
        dtype=torch.long,
    )
    train_stim_ids = torch.tensor(
        train.obs["stim"].astype(str).map(stim_to_id).to_numpy(),
        dtype=torch.long,
    )
    test_cell_ids = torch.tensor(
        test.obs["cell.type"].astype(str).map(cell_to_id).to_numpy(),
        dtype=torch.long,
    )
    test_stim_ids = torch.tensor(
        test.obs["stim"].astype(str).map(stim_to_id).to_numpy(),
        dtype=torch.long,
    )

    # 4. Compress expression first, then learn p(z | cell type, stimulation)
    # with the conditional denoiser. The defaults are deliberately compact,
    # but long enough to produce interpretable training-loss trajectories.
    latent_model, vae_history, train_latents = train_latent_model(
        train_x,
        latent_dim=args.latent_dim,
        epochs=args.vae_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    denoiser, scheduler, diffusion_history = train_denoiser(
        train_latents,
        train_cell_ids,
        train_stim_ids,
        cell_type_count=len(cell_types),
        stim_count=len(stim_values),
        epochs=args.diffusion_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    # 5. Generate each missing stimulated state from diffusion noise. The
    # decoder maps generated latent samples back to the selected genes.
    predicted = generate(
        denoiser,
        scheduler,
        latent_model,
        test_cell_ids,
        test_stim_ids,
        latent_dim=args.latent_dim,
        inference_steps=args.inference_steps,
        device=device,
        seed=args.seed + 2,
    )

    # 6. Report both whole-state agreement and response relative to the matched
    # control mean; the latter prevents background expression from dominating.
    metrics: dict[str, dict[str, float | None]] = {}
    condition_arrays: dict[str, dict[str, np.ndarray]] = {}
    train_cell_type = train.obs["cell.type"].astype(str).to_numpy()
    train_stim = train.obs["stim"].astype(str).to_numpy()
    test_cell_type = test.obs["cell.type"].astype(str).to_numpy()
    for cell_type in HELD_OUT_CELL_TYPES:
        test_mask = test_cell_type == cell_type
        control_mask = (train_cell_type == cell_type) & (train_stim == "ctrl")
        observed_mean = test_x[test_mask].mean(axis=0)
        predicted_mean = predicted[test_mask].mean(axis=0)
        control_mean = train_x[control_mask].mean(axis=0)
        condition_arrays[cell_type] = {
            "observed_mean": observed_mean,
            "predicted_mean": predicted_mean,
            "observed_effect": observed_mean - control_mean,
            "predicted_effect": predicted_mean - control_mean,
        }
        metrics[cell_type] = {
            "absolute_profile_pearson": safe_pearson(
                observed_mean, predicted_mean
            ),
            "absolute_profile_mae": float(
                np.mean(np.abs(observed_mean - predicted_mean))
            ),
            "matched_control_effect_pearson": safe_pearson(
                observed_mean - control_mean,
                predicted_mean - control_mean,
            ),
            "matched_control_effect_mae": float(
                np.mean(
                    np.abs(
                        (observed_mean - control_mean)
                        - (predicted_mean - control_mean)
                    )
                )
            ),
        }

    finite_fraction = float(np.isfinite(predicted).mean())
    passed = (
        predicted.shape == test_x.shape
        and finite_fraction == 1.0
        and all(math.isfinite(value) for value in vae_history)
        and all(math.isfinite(value) for value in diffusion_history)
    )
    plot_files = write_diagnostic_plots(
        output_dir,
        vae_history=vae_history,
        diffusion_history=diffusion_history,
        condition_arrays=condition_arrays,
    )
    artifact_files = write_run_artifacts(
        output_dir,
        args=args,
        genes=genes,
        cell_to_id=cell_to_id,
        stim_to_id=stim_to_id,
        latent_model=latent_model,
        denoiser=denoiser,
        scheduler=scheduler,
        vae_history=vae_history,
        diffusion_history=diffusion_history,
        metrics=metrics,
        condition_arrays=condition_arrays,
        predicted=predicted,
        observed=test_x,
        test_cell_type=test_cell_type,
        test_stim=test.obs["stim"].astype(str).to_numpy(),
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "purpose": "standalone_five_minute_pbmc_diffusion_example",
        "interpretation_boundary": (
            "diagnostic example only; not manuscript reproduction or benchmark evidence"
        ),
        "input": {
            "name": input_path.name,
            "sha256": file_sha256(input_path),
            "initial_shape": initial_shape,
            "post_filter_shape": [int(adata.n_obs), int(adata.n_vars)],
        },
        "split": {
            "held_out_rule": {
                "cell_types": list(HELD_OUT_CELL_TYPES),
                "stim": "stim",
                "megakaryocytes_excluded": True,
            },
            "full_train_cells": int(train_full.n_obs),
            "full_test_cells": int(test_full.n_obs),
            "example_sampling": sampling_rule,
            "example_train_cells": int(train.n_obs),
            "example_test_cells": int(test.n_obs),
            "example_test_by_cell_type": {
                name: int((test.obs["cell.type"] == name).sum())
                for name in HELD_OUT_CELL_TYPES
            },
        },
        "features": {
            "selection": "training_sample_only_hvg",
            "count": len(genes),
            "order_identical_between_train_and_test": bool(
                np.array_equal(train.var_names, test.var_names)
            ),
        },
        "model": {
            "device": str(device),
            "latent_dim": args.latent_dim,
            "vae_epochs": args.vae_epochs,
            "vae_loss": vae_history,
            "diffusion_epochs": args.diffusion_epochs,
            "diffusion_loss": diffusion_history,
            "inference_steps": args.inference_steps,
        },
        "generation": {
            "predicted_shape": list(predicted.shape),
            "observed_shape": list(test_x.shape),
            "finite_fraction": finite_fraction,
        },
        "diagnostic_metrics": metrics,
        "diagnostic_plots": plot_files,
        "artifacts": {
            **artifact_files,
            **plot_files,
            "summary": "pbmc_example_summary.json",
        },
        "resources": {
            "threads": args.threads,
            "runtime_target_seconds": 300,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3
            ),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path = output_dir / "pbmc_example_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
