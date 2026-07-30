#!/usr/bin/env python3
"""Run the corrected-split CPA benchmark with the frozen 40-CPU contract.

This wrapper reuses the previously audited data preparation and model code,
while enforcing at most 20 epochs, validation after every epoch, patience-four
early stopping, one recoverable Lightning checkpoint per completed epoch,
durable epoch metrics, and explicit selection of the global validation-best
checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["NUMEXPR_NUM_THREADS"] = "40"

import pandas as pd
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint


RUN = Path(__file__).resolve().parents[1]
BASE_SCRIPT = RUN / "scripts/CPA_global_random_corrected_split_pretrain_gauss_base40cpu20epoch.py"
CHECKPOINT_ROOT = RUN / "random_gauss_result/checkpoints_40cpu_20epochs"
EVERY_EPOCH_DIR = CHECKPOINT_ROOT / "every_epoch"
BEST_DIR = CHECKPOINT_ROOT / "best"
EPOCH_METRICS_DIR = CHECKPOINT_ROOT / "epoch_metrics"
CONTRACT_PATH = RUN / "provenance/runtime_contract_40cpu20epoch.json"


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class DurableEpochMetrics(Callback):
    """Write a complete train/validation record after every validation epoch."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(trainer.current_epoch)
        history = pd.DataFrame.from_dict(pl_module.epoch_history)
        rows = history.loc[history["epoch"].astype(int).eq(epoch)].to_dict("records")
        modes = {str(row["mode"]) for row in rows}
        if modes != {"train", "valid"}:
            raise RuntimeError(
                f"epoch {epoch} does not contain exactly the expected train/valid modes: {modes}"
            )
        record = {
            "epoch_zero_based": epoch,
            "epoch_one_based": epoch + 1,
            "cpa_metric": json_value(trainer.callback_metrics["cpa_metric"]),
            "rows": [
                {key: json_value(value) for key, value in row.items()}
                for row in rows
            ],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_dir / f"epoch_{epoch:02d}.json"
        temporary_path = self.output_dir / f".epoch_{epoch:02d}.json.tmp"
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(final_path)


class ContractEarlyStopping(EarlyStopping):
    """Standard early stopping with an isolated plateau-test injection."""

    def _evaluate_stopping_criteria(self, current):
        if os.environ.get("CPA_FORCE_PLATEAU_FOR_DRY_TEST") == "1":
            current = torch.zeros_like(current)
        return super()._evaluate_stopping_criteria(current)


def load_base_module():
    spec = importlib.util.spec_from_file_location("cpa_corrected_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_epoch_checkpoints() -> list[Path]:
    return sorted(EVERY_EPOCH_DIR.glob("epoch*.ckpt"))


def rebuild_durable_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    rows = []
    for path in sorted(EPOCH_METRICS_DIR.glob("epoch_*.json")):
        record = json.loads(path.read_text())
        records.append(
            {
                "epoch_zero_based": int(record["epoch_zero_based"]),
                "epoch_one_based": int(record["epoch_one_based"]),
                "cpa_metric": float(record["cpa_metric"]),
                "record_path": str(path),
            }
        )
        rows.extend(record["rows"])
    if not records:
        raise RuntimeError("no durable epoch metrics were recorded")
    metric_table = pd.DataFrame(records).sort_values("epoch_zero_based")
    history = pd.DataFrame(rows).sort_values(["epoch", "mode"])
    completed_epochs = len(metric_table)
    if completed_epochs > 20:
        raise RuntimeError(f"completed epoch count exceeds max_epochs: {completed_epochs}")
    expected = list(range(completed_epochs))
    if metric_table["epoch_zero_based"].tolist() != expected:
        raise RuntimeError("durable validation metrics are not contiguous from epoch 0")
    train_epochs = (
        history.loc[history["mode"].eq("train"), "epoch"].astype(int).tolist()
    )
    valid_epochs = (
        history.loc[history["mode"].eq("valid"), "epoch"].astype(int).tolist()
    )
    if train_epochs != expected or valid_epochs != expected:
        raise RuntimeError("durable history lacks paired train/valid rows for completed epochs")
    return history, metric_table


def checkpoint_for_epoch(epoch: int) -> Path:
    candidates = sorted(EVERY_EPOCH_DIR.glob(f"epoch{epoch:02d}*.ckpt"))
    if not candidates:
        raise RuntimeError(f"missing checkpoint for zero-based epoch {epoch}")
    return candidates[-1]


def main() -> None:
    torch.set_num_threads(40)
    EVERY_EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    EPOCH_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_module()
    torch.set_num_threads(40)
    base.logger.info(
        "Frozen runtime contract: intraop_threads=%s interop_threads=%s max_epochs=20 "
        "validation_every_epoch=true early_stopping=true patience=4",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    original_train = base.CPA.train

    def train_with_frozen_contract(cpa_model, *args, **kwargs):
        epoch_checkpoint = ModelCheckpoint(
            dirpath=str(EVERY_EPOCH_DIR),
            filename="epoch{epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=False,
        )
        durable_metrics = DurableEpochMetrics(EPOCH_METRICS_DIR)
        early_stopping_callback = ContractEarlyStopping(
            monitor="cpa_metric",
            mode="max",
            patience=4,
            min_delta=0.0001,
            check_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks = [durable_metrics, epoch_checkpoint, early_stopping_callback]
        existing = completed_epoch_checkpoints()
        resume_checkpoint = str(existing[-1]) if existing else None
        if resume_checkpoint:
            base.logger.info("Resuming from completed checkpoint: %s", resume_checkpoint)

        kwargs.update(
            {
                "max_epochs": 20,
                "early_stopping": False,
                "early_stopping_patience": 4,
                "enable_checkpointing": True,
                "check_val_every_n_epoch": 1,
                "callbacks": callbacks,
                "resume_from_checkpoint": resume_checkpoint,
                "save_path": False,
            }
        )
        result = original_train(cpa_model, *args, **kwargs)

        durable_history, metric_table = rebuild_durable_history()
        best_row = metric_table.loc[metric_table["cpa_metric"].idxmax()]
        best_epoch = int(best_row["epoch_zero_based"])
        best_source = checkpoint_for_epoch(best_epoch)
        selected_best = BEST_DIR / (
            f"selected_best_epoch{best_epoch:02d}_"
            f"cpa{float(best_row['cpa_metric']):.6f}.ckpt"
        )
        shutil.copy2(best_source, selected_best)

        checkpoint = torch.load(selected_best, map_location="cpu")
        cpa_model.training_plan.load_state_dict(checkpoint["state_dict"], strict=True)
        cpa_model.epoch_history = durable_history.reset_index(drop=True)

        metric_table.to_csv(CHECKPOINT_ROOT / "validation_cpa_metric_by_epoch.csv", index=False)
        completed_epochs = len(metric_table)
        selection = {
            "selection_metric": "cpa_metric",
            "selection_mode": "max",
            "best_epoch_zero_based": best_epoch,
            "best_epoch_one_based": best_epoch + 1,
            "best_metric": float(best_row["cpa_metric"]),
            "source_checkpoint": str(best_source),
            "selected_best_checkpoint": str(selected_best),
            "completed_epoch_checkpoint_count": len(completed_epoch_checkpoints()),
            "completed_epochs": completed_epochs,
            "max_epochs": 20,
            "early_stopping_enabled": True,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 0.0001,
            "early_stopping_monitor": "cpa_metric",
            "early_stopping_mode": "max",
            "early_stopping_stopped_epoch_zero_based": int(
                early_stopping_callback.stopped_epoch
            ),
            "early_stopping_wait_count": int(early_stopping_callback.wait_count),
            "early_stopping_best_score": json_value(
                early_stopping_callback.best_score
            ),
            "stopped_early": completed_epochs < 20,
        }
        (BEST_DIR / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
        base.logger.info(
            "Selected global validation-best checkpoint: epoch=%s cpa_metric=%.6f path=%s",
            best_epoch + 1,
            float(best_row["cpa_metric"]),
            selected_best,
        )
        return result

    base.CPA.train = train_with_frozen_contract
    contract = {
        "run_root": str(RUN),
        "intraop_threads": 40,
        "interop_threads": 16,
        "cpu_affinity_expected": "40 unique physical cores; see provenance/cpa_cpu_affinity_40physicalcores.txt",
        "max_epochs": 20,
        "validation_every_n_epoch": 1,
        "early_stopping": True,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.0001,
        "early_stopping_monitor": "cpa_metric",
        "early_stopping_mode": "max",
        "checkpoint_every_epoch": True,
        "best_checkpoint_metric": "cpa_metric",
        "best_checkpoint_mode": "max",
        "external_test_role": "OOD only",
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    base.main()


if __name__ == "__main__":
    main()
