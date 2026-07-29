#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import scanpy as sc
import numpy as np
import pandas as pd
import json
from cpa import ComPertAPI, CPA
import cpa._api

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.stats import rankdata
import time
import logging
import traceback
import sys
from datetime import datetime
import gc
import os

# Force CPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import torch
torch.cuda.is_available = lambda: False
torch.set_num_threads(16)
torch.set_num_interop_threads(8)

def json_default(obj):
    """Handle numpy types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def setup_logging():
    """Setup logging system"""
    log_dir = "./random_inference_full_gauss/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/cpa_single_cell_inference_gauss_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def chatterjee_corr(x, y):
    """Calculate Chatterjee correlation coefficient"""
    n = len(x)
    if n < 2:
        return 0.0
    order = np.argsort(x)
    y_ordered = y[order]
    ranks = rankdata(y_ordered, method='ordinal')
    diff = np.abs(np.diff(ranks))
    num = diff.sum()
    return 1 - (3 * num) / (n**2 - 1)

def filter_unseen_cell_lines(adata_all):
    """Filter out cell lines that weren't seen during training"""
    logger.info("Filtering out unseen cell lines...")
    
    # Cell lines seen during training (from model CPA_info.json)
    training_cell_lines = {
        "CVCL-0023", "CVCL-0028", "CVCL-0069", "CVCL-0099", "CVCL-0131", "CVCL-0152", 
        "CVCL-0179", "CVCL-0218", "CVCL-0292", "CVCL-0293", "CVCL-0320", "CVCL-0332", 
        "CVCL-0334", "CVCL-0359", "CVCL-0366", "CVCL-0371", "CVCL-0397", "CVCL-0399", 
        "CVCL-0428", "CVCL-0459", "CVCL-0480", "CVCL-0504", "CVCL-0546", "CVCL-1055", 
        "CVCL-1056", "CVCL-1094", "CVCL-1097", "CVCL-1098", "CVCL-1119", "CVCL-1125", 
        "CVCL-1239", "CVCL-1285", "CVCL-1381", "CVCL-1478", "CVCL-1495", "CVCL-1517", 
        "CVCL-1547", "CVCL-1550", "CVCL-1635", "CVCL-1666", "CVCL-1693", "CVCL-1715", 
        "CVCL-1716", "CVCL-1717", "CVCL-1724", "CVCL-1731", "CVCL-C466"
    }
    
    original_samples = adata_all.n_obs
    current_cell_lines = set(adata_all.obs['cell_line'].unique())
    
    logger.info(f"Cell lines in data: {len(current_cell_lines)}")
    logger.info(f"Cell lines in training: {len(training_cell_lines)}")
    
    # Find unseen cell lines
    unseen_cell_lines = current_cell_lines - training_cell_lines
    if unseen_cell_lines:
        logger.warning(f"Found {len(unseen_cell_lines)} unseen cell lines: {sorted(list(unseen_cell_lines))[:5]}...")
        
        # Filter out unseen cell lines
        mask = adata_all.obs['cell_line'].isin(training_cell_lines)
        adata_all = adata_all[mask].copy()
        
        filtered_samples = adata_all.n_obs
        removed_samples = original_samples - filtered_samples
        
        logger.info(f"Removed {removed_samples} samples with unseen cell lines")
        logger.info(f"Remaining samples: {filtered_samples}")
        
        # Updated split distribution
        new_split_counts = adata_all.obs["split"].value_counts().to_dict()
        logger.info(f"Updated split distribution: {new_split_counts}")
        
        if filtered_samples == 0:
            raise ValueError("No samples remain after filtering unseen cell lines")
    else:
        logger.info("All cell lines were seen during training")
    
    # Show remaining cell lines
    remaining_cell_lines = set(adata_all.obs['cell_line'].unique())
    logger.info(f"Remaining cell lines: {len(remaining_cell_lines)}")
    
    return adata_all

# Pretrain uses preprocessed data directly without string cleaning

def load_inference_data():
    """Load only test and control data for inference - EXACT same as working version"""
    logger.info("Loading inference data (test and control only)...")
    
    try:
        # Load only test and control data
        adata_test = sc.read_h5ad("../random_data/test_adata_processed.h5ad")
        adata_ctrl = sc.read_h5ad("../random_data/control_adata_processed.h5ad")
#        adata_test = sc.read_h5ad("../test_downsample.h5ad")
#        adata_ctrl = sc.read_h5ad("../control_downsample.h5ad")
        logger.info(f"Data loaded successfully: test={adata_test.n_obs}, ctrl={adata_ctrl.n_obs}")
        
        # Data cleaning to match exact training format from model registry
        datasets = (adata_ctrl, adata_test)
        for ad in datasets:
            # Convert cell line names from CVCL_xxxx to CVCL-xxxx format
            ad.obs["cell_line"] = ad.obs["cell_line"].str.replace("CVCL_", "CVCL-", regex=False)

            # Fix DMSO_TF to DMSO-TF to match training
            ad.obs["drug"] = ad.obs["drug"].str.replace("DMSO_TF", "DMSO-TF", regex=False)

            # Remove trailing spaces from drug names to match model registry
            ad.obs["drug"] = ad.obs["drug"].str.strip()

            # Create dose_str
            ad.obs["dose_str"] = (
                ad.obs["dose"]
                .astype(str)
                .str.replace(".", "-", regex=False)
            )
        
        # Concatenate control and test data
        adata_all = sc.concat(
            [adata_ctrl, adata_test],
            join="inner",
            label="split", 
            keys=["ctrl", "test"],
            index_unique=None
        )

        # Release original data to save memory
        del adata_ctrl, adata_test
        gc.collect()
        
        logger.info(f"Concatenated data: {adata_all.n_obs} cells, {adata_all.n_vars} genes")
        split_counts = adata_all.obs["split"].value_counts().to_dict()
        logger.info(f"Split distribution: {split_counts}")
        
        # Filter to only keep cell lines seen during training
        adata_all = filter_unseen_cell_lines(adata_all)
        
        return adata_all
        
    except Exception as e:
        logger.error(f"Failed to load inference data: {e}")
        raise

def find_control_group(adata_all):
    """Find the appropriate control group from the data - EXACT same as working version"""
    logger.info("Finding control group...")
    
    # Use simplified control group setup (matching training script)
    global_control = "DMSO-TF"
    
    # Validate control group exists
    control_mask = adata_all.obs["drug"] == global_control
    control_count = np.sum(control_mask)
    
    if control_count == 0:
        logger.warning(f"Control group {global_control} not found")
        logger.info("Checking available drug types:")
        all_drugs = set(adata_all.obs["drug"].unique())
        logger.info(f"Available drugs: {sorted(all_drugs)}")
        
        # Look for DMSO alternatives
        dmso_alternatives = [drug for drug in all_drugs if "DMSO" in drug.upper()]
        if dmso_alternatives:
            global_control = dmso_alternatives[0]
            control_count = np.sum(adata_all.obs["drug"] == global_control)
            logger.info(f"Using alternative control: {global_control} ({control_count} samples)")
        else:
            logger.error("No suitable control group found")
            raise ValueError("No suitable control group found")
    else:
        logger.info(f"Using control group: {global_control} ({control_count} samples)")
    
    return global_control

def load_trained_model(model_path, adata_all, global_control):
    """Load trained CPA model - EXACT same as working version"""
    logger.info(f"Loading trained model: {model_path}")
    
    try:
        # Setup CPA to match training script setup
        import warnings
        import sys
        from io import StringIO
        
        warnings.filterwarnings('ignore')
        sc.settings.verbosity = 0
        
        # Suppress output during setup
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            CPA.split_key = "split"
            CPA.setup_anndata(
                adata=adata_all,
                perturbation_key="drug",
                control_group=global_control,
                dosage_key="dose",
                categorical_covariate_keys=["cell_line"]
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        warnings.filterwarnings('default')
        sc.settings.verbosity = 1
        
        # Load model
        cpa_model = CPA.load(model_path, adata=adata_all)
        logger.info("Model loaded successfully")
        
        return cpa_model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

def perform_cell_line_batch_counterfactual_analysis(cpa_model, adata_all, global_control):
    """Perform counterfactual analysis by cell line batches with memory optimization"""
    logger.info("Starting cell line batch counterfactual analysis...")

    # Separate control and test data
    ctrl_data = adata_all[adata_all.obs["split"] == "ctrl"].copy()
    test_data = adata_all[adata_all.obs["split"] == "test"].copy()

    logger.info(f"Control samples: {ctrl_data.n_obs}")
    logger.info(f"Test samples: {test_data.n_obs}")

    # Get unique test conditions grouped by cell line
    test_conditions_df = test_data.obs[["cell_line", "drug", "dose_str"]].drop_duplicates()

    # Group conditions by cell line for batch processing
    conditions_by_cell_line = {}
    for _, row in test_conditions_df.iterrows():
        cell_line = row['cell_line']
        if cell_line not in conditions_by_cell_line:
            conditions_by_cell_line[cell_line] = []
        conditions_by_cell_line[cell_line].append((row['cell_line'], row['drug'], row['dose_str']))

    del test_conditions_df
    gc.collect()

    logger.info(f"Found {len(conditions_by_cell_line)} unique cell lines with conditions")

    # Create output directories
    os.makedirs("./random_inference_full_gauss", exist_ok=True)
    os.makedirs("./random_inference_full_gauss/by_cell_line", exist_ok=True)

    # Process each cell line batch
    overall_metrics = {}
    cell_line_files = {}
    total_cell_lines = len(conditions_by_cell_line)

    for cell_line_idx, (cell_line, conditions) in enumerate(conditions_by_cell_line.items(), 1):
        logger.info(f"[{cell_line_idx}/{total_cell_lines}] Processing cell line: {cell_line} ({len(conditions)} conditions)")

        try:
            # Process this cell line batch
            cell_line_results = process_single_cell_line_batch(
                cell_line, conditions, ctrl_data, test_data, cpa_model, global_control, adata_all.var_names
            )

            if cell_line_results is None:
                logger.warning(f"Failed to process cell line: {cell_line}")
                continue

            inference_adata, metrics = cell_line_results

            # Save cell line specific data
            cell_line_filename = f"cpa_inference_{cell_line.replace('-', '_')}.h5ad"
            cell_line_path = f"./random_inference_full_gauss/by_cell_line/{cell_line_filename}"
            sc.write(cell_line_path, inference_adata)

            # Save cell line specific metrics
            metrics_filename = f"metrics_{cell_line.replace('-', '_')}.json"
            metrics_path = f"./random_inference_full_gauss/by_cell_line/{metrics_filename}"
            with open(metrics_path, "w") as f:
                json.dump({
                    "cell_line": cell_line,
                    "metrics_by_condition": metrics,
                    "summary": {
                        "n_conditions": len(metrics),
                        "n_inference_cells": int(inference_adata.n_obs),
                        "mean_pearson_r": float(np.mean([m["pearson_r"] for m in metrics.values()])) if metrics else 0.0,
                        "mean_spearman_r": float(np.mean([m["spearman_r"] for m in metrics.values()])) if metrics else 0.0
                    }
                }, f, indent=2, default=json_default)

            overall_metrics[cell_line] = metrics
            cell_line_files[cell_line] = {
                "inference_file": cell_line_path,
                "metrics_file": metrics_path
            }

            logger.info(f"Saved {cell_line}: {inference_adata.n_obs} cells, {len(metrics)} conditions")

            # Clean up this cell line's data immediately
            del inference_adata, metrics
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing cell line {cell_line}: {e}")
            continue

    # Clean up control and test data
    del ctrl_data, test_data
    gc.collect()

    # Create merged metrics by condition (consistent with previous logic)
    logger.info("Creating merged metrics by condition...")
    merged_metrics_by_condition = {}

    for cell_line, cell_line_metrics in overall_metrics.items():
        for condition, metrics in cell_line_metrics.items():
            merged_metrics_by_condition[condition] = metrics

    # Save merged metrics by condition
    merged_metrics_path = "./random_inference_full_gauss/merged_metrics_by_condition.json"
    with open(merged_metrics_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "description": "Merged metrics for all conditions across all cell lines",
            "total_conditions": len(merged_metrics_by_condition),
            "metrics_by_condition": merged_metrics_by_condition,
            "summary_statistics": {
                "mean_pearson_r": float(np.mean([m["pearson_r"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0,
                "mean_spearman_r": float(np.mean([m["spearman_r"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0,
                "mean_mse": float(np.mean([m["mse"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0,
                "mean_mae": float(np.mean([m["mae"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0,
                "mean_r2_score": float(np.mean([m["r2_score"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0,
                "mean_chatterjee_r": float(np.mean([m["chatterjee_r"] for m in merged_metrics_by_condition.values()])) if merged_metrics_by_condition else 0.0
            }
        }, f, indent=2, default=json_default)

    # Save overall summary
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "cell_line_batch_cpa_counterfactual_inference_gauss",
        "description": "CPA counterfactual inference processed by cell line batches with memory optimization",
        "data_format": "float32_memory_optimized",
        "total_cell_lines": len(cell_line_files),
        "total_conditions": len(merged_metrics_by_condition),
        "metrics_calculated": ["mse", "mae", "r2_score", "pearson_r", "spearman_r", "chatterjee_r"],
        "cell_line_files": cell_line_files,
        "merged_metrics_file": merged_metrics_path,
        "overall_metrics": overall_metrics
    }

    summary_path = "./random_inference_full_gauss/overall_summary_with_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2, default=json_default)

    logger.info("Cell line batch processing complete:")
    logger.info(f"  Overall summary: {summary_path}")
    logger.info(f"  Merged metrics by condition: {merged_metrics_path}")
    logger.info(f"  Cell line specific files in: ./random_inference_full_gauss/by_cell_line/")
    logger.info(f"  Total cell lines processed: {len(cell_line_files)}")
    logger.info(f"  Total conditions with metrics: {len(merged_metrics_by_condition)}")

    return overall_summary


def process_single_cell_line_batch(cell_line, conditions, ctrl_data, test_data, cpa_model, global_control, var_names):
    """Process counterfactual inference for a single cell line batch"""
    logger.info(f"  Processing {len(conditions)} conditions for {cell_line}")

    # Find ALL control cells (DMSO-TF) for this cell line
    ctrl_mask = (ctrl_data.obs['cell_line'] == cell_line) & (ctrl_data.obs['drug'] == global_control)
    cell_line_controls = ctrl_data[ctrl_mask]

    if cell_line_controls.n_obs == 0:
        logger.warning(f"  No DMSO-TF control cells found for {cell_line}")
        return None

    logger.info(f"  Found {cell_line_controls.n_obs} DMSO-TF control cells")

    # Prepare inference data for all conditions of this cell line
    all_ctrl_cells = []
    all_obs_data = []
    cell_line_metrics = {}

    # Get control cell expressions once
    ctrl_expressions = cell_line_controls.X.toarray() if hasattr(cell_line_controls.X, 'toarray') else cell_line_controls.X
    ctrl_expressions = ctrl_expressions.astype(np.float32)

    for condition_idx, (cell_line_name, drug, dose_str) in enumerate(conditions):
        logger.info(f"    Condition {condition_idx + 1}/{len(conditions)}: {drug} @ {dose_str}")

        # Check if there are real perturbed cells for this condition
        test_mask = ((test_data.obs['cell_line'] == cell_line) &
                    (test_data.obs['drug'] == drug) &
                    (test_data.obs['dose_str'] == dose_str))
        actual_perturbed_cells = test_data[test_mask]

        if actual_perturbed_cells.n_obs == 0:
            logger.warning(f"    No real perturbed cells found for {cell_line}_{drug}_{dose_str}")
            continue

        # Add ALL control cells for this condition
        for j in range(cell_line_controls.n_obs):
            all_ctrl_cells.append(ctrl_expressions[j])

            all_obs_data.append({
                "cell_line": cell_line,
                "drug": drug,
                "dose_str": dose_str,
                "dose": np.float32(float(dose_str.replace("-", "."))),
                "condition": f"{cell_line}_{drug}_{dose_str}",
                "cell_idx": j,
                "original_ctrl_cell_idx": cell_line_controls.obs.index[j],
                "n_real_perturbed_cells": actual_perturbed_cells.n_obs,
                "n_ctrl_cells_used": cell_line_controls.n_obs
            })

    if not all_ctrl_cells:
        logger.warning(f"  No valid conditions found for {cell_line}")
        return None

    logger.info(f"  Total inference cells for {cell_line}: {len(all_ctrl_cells)}")

    # Create batch AnnData for this cell line
    ctrl_expressions_matrix = np.vstack(all_ctrl_cells).astype(np.float32)
    obs_df = pd.DataFrame(all_obs_data)

    batch_adata = sc.AnnData(X=ctrl_expressions_matrix, obs=obs_df)
    batch_adata.var_names = var_names

    # Clean up intermediate data
    del all_ctrl_cells, ctrl_expressions_matrix, all_obs_data
    gc.collect()

    # Setup CPA annotations
    try:
        import warnings
        import sys
        from io import StringIO

        warnings.filterwarnings('ignore')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            CPA.setup_anndata(
                adata=batch_adata,
                perturbation_key="drug",
                control_group=global_control,
                dosage_key="dose",
                categorical_covariate_keys=["cell_line"]
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            warnings.filterwarnings('default')

    except Exception as e:
        logger.error(f"  Failed to setup batch_adata for {cell_line}: {e}")
        return None

    # Perform inference
    logger.info(f"  Performing inference for {cell_line}...")
    cpa_model.predict(batch_adata)

    # Extract and ensure float32 format
    predictions = batch_adata.obsm["CPA_pred"]
    if not isinstance(predictions, np.ndarray):
        predictions = predictions.toarray()
    batch_adata.obsm["CPA_pred"] = predictions.astype(np.float32)

    if hasattr(batch_adata.X, 'toarray'):
        batch_adata.X = batch_adata.X.toarray().astype(np.float32)
    else:
        batch_adata.X = batch_adata.X.astype(np.float32)

    if 'dose' in batch_adata.obs.columns:
        batch_adata.obs['dose'] = batch_adata.obs['dose'].astype(np.float32)

    # Calculate metrics for each condition
    logger.info(f"  Calculating metrics for {cell_line}...")
    condition_names = batch_adata.obs['condition'].unique()

    for condition in condition_names:
        # Get inference results for this condition
        condition_mask = batch_adata.obs['condition'] == condition
        condition_inference = batch_adata[condition_mask]

        # Parse condition to get drug and dose info
        parts = condition.split('_')
        if len(parts) >= 3:
            drug = '_'.join(parts[1:-1])
            dose_str = parts[-1]
        else:
            continue

        # Get real perturbed cells for this condition
        real_mask = ((test_data.obs['cell_line'] == cell_line) &
                    (test_data.obs['drug'] == drug) &
                    (test_data.obs['dose_str'] == dose_str))
        real_perturbed = test_data[real_mask]

        if real_perturbed.n_obs == 0:
            continue

        # Calculate metrics
        try:
            inference_pred = condition_inference.obsm["CPA_pred"]
            real_expr = real_perturbed.X.toarray() if hasattr(real_perturbed.X, 'toarray') else real_perturbed.X

            mean_pred = np.mean(inference_pred, axis=0)
            mean_real = np.mean(real_expr, axis=0)

            mse = mean_squared_error(mean_real, mean_pred)
            mae = mean_absolute_error(mean_real, mean_pred)
            r2 = r2_score(mean_real, mean_pred)
            pearson_r, pearson_p = pearsonr(mean_real, mean_pred)
            spearman_r, spearman_p = spearmanr(mean_real, mean_pred)
            chatterjee_r = chatterjee_corr(mean_real, mean_pred)

            cell_line_metrics[condition] = {
                "mse": float(mse),
                "mae": float(mae),
                "r2_score": float(r2),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "chatterjee_r": float(chatterjee_r),
                "n_inference_cells": int(condition_inference.n_obs),
                "n_real_cells": int(real_perturbed.n_obs)
            }

        except Exception as e:
            logger.warning(f"    Failed to calculate metrics for {condition}: {e}")
            continue

    logger.info(f"  Completed {cell_line}: {len(cell_line_metrics)} conditions with metrics")

    return batch_adata, cell_line_metrics


def main():
    logger.info("="*60)
    logger.info("CPA Single-Cell Counterfactual Inference (Based on Working Version)")
    logger.info("="*60)
    
    try:
        # 1. Load inference data - SAME as working version
        adata_all = load_inference_data()
        
        # 2. Find control group - SAME as working version
        global_control = find_control_group(adata_all)
        
        # 3. Load trained model - SAME as working version
        model_path = "./random_gauss_result/models/cpa_global_model_gauss.pth"  # Directory containing model.pt
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            logger.error("Please confirm the model path is correct, or run training script first")
            return
        
        cpa_model = load_trained_model(model_path, adata_all, global_control)
        
        # 4. Perform cell line batch counterfactual analysis with memory optimization
        overall_summary = perform_cell_line_batch_counterfactual_analysis(cpa_model, adata_all, global_control)

        if overall_summary is None:
            logger.error("Cell line batch counterfactual analysis failed")
            return

        # Release model and data after processing
        del cpa_model, adata_all
        gc.collect()
        
        # 6. Final report
        logger.info("="*60)
        logger.info("CPA Cell Line Batch Counterfactual Inference Complete (Memory Optimized)")
        logger.info("="*60)
        logger.info("Key results:")
        logger.info("  Main output directory: ./random_inference_full_gauss/by_cell_line/")
        logger.info("  Overall summary: ./random_inference_full_gauss/overall_summary_with_metrics.json")
        logger.info("  Merged metrics by condition: ./random_inference_full_gauss/merged_metrics_by_condition.json")
        logger.info("  Data format: float32 optimized for memory efficiency")
        logger.info("  Analysis type: Cell line batch processing with immediate cleanup")
        logger.info("")
        logger.info("Memory optimization features:")
        logger.info("  ✓ Process one cell line at a time (batch processing)")
        logger.info("  ✓ Immediate cleanup after each cell line is processed")
        logger.info("  ✓ Save results immediately to disk")
        logger.info("  ✓ Delete processed cell line data from memory")
        logger.info("  ✓ Convert cell line names from CVCL_xxxx to CVCL-xxxx format")
        logger.info("")
        logger.info("Optimized methodology:")
        logger.info("  1. Group conditions by cell line for batch processing")
        logger.info("  2. For each cell line batch:")
        logger.info("     - Load ALL DMSO-TF control cells for that cell line")
        logger.info("     - Apply drug perturbations for all conditions")
        logger.info("     - Calculate metrics vs real perturbed cells")
        logger.info("     - Save results immediately (h5ad + JSON)")
        logger.info("     - Delete processed data from memory")
        logger.info("  3. Generate overall summary after all cell lines processed")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        
        # Clean up on error
        gc.collect()


if __name__ == "__main__":
    main()
