#!/usr/bin/env python

"""
ChemCPA Distribution Similarity Analysis - Plugin System
Performs single-cell counterfactual inference and computes distribution similarity metrics only.
Uses chemcpa_distribution_metrics.py for all metrics computation.
Focus: MMD, E-distance, Sliced Wasserstein, and OT Wasserstein metrics between predicted and real distributions.
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import rankdata
from chemcpa_distribution_metrics import (
    distributional_similarity_metrics,
    compute_condition_distribution_metrics
)
import matplotlib.patches as mpatches
import gc
import sys
import warnings
from io import StringIO
import glob
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ["OMP_NUM_THREADS"] = "90"
os.environ["OPENBLAS_NUM_THREADS"] = "90"
os.environ["MKL_NUM_THREADS"] = "90"
os.environ["NUMEXPR_NUM_THREADS"] = "90"

torch.set_num_threads(90)

RESULT_ROOT = "./random_distribution_similarity_gauss"
PRECOMPUTED_DIR = "./random_inference_full_gauss/by_cell_line"

def json_default(obj):
    """Handle numpy types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
def setup_logging():
    """Setup logging system"""
    log_dir = f"{RESULT_ROOT}/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/cpa_random_counterfactual_distribution_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========= 串联数据流：数据源检测和预计算数据加载 =========

def resolve_benchmark_data_file(filename):
    """Resolve one required input from the portable benchmark data root."""
    data_dir = os.environ.get("PERTURBLDM_BENCHMARK_DATA_DIR")
    if not data_dir:
        raise RuntimeError(
            "Missing required environment variable PERTURBLDM_BENCHMARK_DATA_DIR. "
            "See method_packages/PORTABILITY.md."
        )
    path = (Path(data_dir).expanduser().resolve() / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark input does not exist: {path}. "
            "Check PERTURBLDM_BENCHMARK_DATA_DIR."
        )
    return str(path)


def resolve_drug_metadata_file():
    """Resolve the required chemCPA drug-metadata CSV."""
    value = os.environ.get("PERTURBLDM_CHEMCPA_DRUG_METADATA")
    if not value:
        raise RuntimeError(
            "Missing required environment variable PERTURBLDM_CHEMCPA_DRUG_METADATA. "
            "See method_packages/PORTABILITY.md."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required chemCPA drug metadata does not exist: {path}. "
            "Check PERTURBLDM_CHEMCPA_DRUG_METADATA."
        )
    return str(path)

_cached_test_data = None

def load_test_data_once():
    """只加载一次test_data，后续使用缓存"""
    global _cached_test_data
    if _cached_test_data is not None:
        logger.info("✅ 使用缓存的test_data，避免重复加载")
        return _cached_test_data

    logger.info("🔄 首次加载test_data...")
    test_data_path = resolve_benchmark_data_file("test_adata_processed.h5ad")
    logger.info(f"加载test_data: {test_data_path}")
    adata_test = sc.read_h5ad(test_data_path)

    for col in ["cell_line", "drug"]:
        adata_test.obs[col] = (
            adata_test.obs[col]
            .astype(str)
            .str.strip()
            .str.replace("_", "-", regex=False)
        )

    # 强制重新创建dose_str以确保一致性
    logger.info("为测试数据创建/更新dose_str列...")
    adata_test.obs['dose_str'] = adata_test.obs['dose'].astype(str).str.replace(".", "-", regex=False)

    logger.info(f"✅ test_data加载完成: {adata_test.n_obs} 细胞, 数据类型: {adata_test.X.dtype}, 列: {list(adata_test.obs.columns)}")

    # 输出dose_str样例以便调试
    sample_doses = adata_test.obs[['dose', 'dose_str']].drop_duplicates().head(10)
    logger.info(f"dose_str样例 (test_data):\n{sample_doses.to_string()}")

    _cached_test_data = adata_test
    return _cached_test_data

def detect_data_source():
    """检测数据源：只使用分cell line预计算结果"""
    precomputed_dir = PRECOMPUTED_DIR
    if os.path.exists(precomputed_dir) and len(glob.glob(f"{precomputed_dir}/*_inference_*.h5ad")) > 0:
        logger.info("🔗 检测到预计算的分cell line推理结果，将使用内存高效模式")
        return "precomputed_cellline"
    else:
        logger.info("📊 未检测到预计算结果")
        return "not_found"

def load_precomputed_cellline_data_efficiently():
    """内存高效的分cell line数据处理 - 逐个处理而不是全量合并"""
    logger.info("🔗 开始内存高效的分cell line推理数据处理...")
    try:
        cellline_results_dir = PRECOMPUTED_DIR

        h5ad_files = glob.glob(f"{cellline_results_dir}/*_inference_*.h5ad")

        if not h5ad_files:
            logger.warning("未找到cell line推理结果文件")
            return None

        logger.info(f"发现 {len(h5ad_files)} 个cell line推理结果文件")

        adata_test = load_test_data_once()

        logger.info(f"测试数据准备完成: {adata_test.n_obs} 细胞, 列: {list(adata_test.obs.columns)}")

        return {
            "h5ad_files": h5ad_files,
            "adata_test": adata_test,
            "cellline_results_dir": cellline_results_dir
        }

    except Exception as e:
        logger.error(f"❌ 预计算数据初始化失败: {e}")
        return None

def load_precomputed_cellline_data():
    """保持原接口兼容性的wrapper函数"""
    result = load_precomputed_cellline_data_efficiently()
    if result is None:
        return None, None, None
    return "EFFICIENT_MODE", result["adata_test"], result

def process_cellline_distribution_metrics_efficiently(efficient_data):
    """内存高效的分cell line分布度量计算"""
    logger.info("🚀 开始内存高效的分cell line分布度量分析...")
    h5ad_files = efficient_data["h5ad_files"]
    adata_test = efficient_data["adata_test"]
    cellline_results_dir = efficient_data["cellline_results_dir"]

    all_condition_metrics = []
    all_cellline_metrics = []
    total_inference_cells = 0

    for file_idx, h5ad_file in enumerate(h5ad_files, 1):
        # 提取 cell line 名称，处理不同的文件命名格式
        basename = os.path.basename(h5ad_file)
        if "_inference_results.h5ad" in basename:
            cell_line = basename.replace("_inference_results.h5ad", "")
        else:
            # 处理 cpa_inference_XXX.h5ad 格式
            cell_line = basename.replace("cpa_inference_", "").replace(".h5ad", "").replace("_", "-")

        logger.info(f"\n{'='*60}")
        logger.info(f"处理Cell Line [{file_idx}/{len(h5ad_files)}]: {cell_line}")
        logger.info(f"{'='*60}")

        # 确保每次循环开始时清理前一次的变量
        if file_idx > 1:
            gc.collect()
            logger.info(f"  🧹 循环前内存清理完成")

        try:
            logger.info(f"加载 {cell_line} 推理数据...")
            cellline_adata = sc.read_h5ad(h5ad_file)
            logger.info(f"  推理细胞数: {cellline_adata.n_obs}, 数据类型: {cellline_adata.X.dtype}")

            for col in ["cell_line", "drug"]:
                cellline_adata.obs[col] = (
                    cellline_adata.obs[col]
                    .astype(str)
                    .str.strip()
                    .str.replace("_", "-", regex=False)
                )

            if "CPA_pred" not in cellline_adata.obsm:
                logger.warning(f"  {cell_line} 缺少预测数据，跳过")
                continue

            predictions = cellline_adata.obsm["CPA_pred"]
            if not isinstance(predictions, np.ndarray):
                predictions = predictions.toarray()

            logger.info(f"  预测数据形状: {predictions.shape}, 类型: {predictions.dtype}")

            logger.info(f"  可用列: {list(cellline_adata.obs.columns)}")

            # 强制重新创建dose_str以确保与test_data一致
            logger.info("  创建/更新dose_str列...")
            cellline_adata.obs['dose_str'] = cellline_adata.obs['dose'].astype(str).str.replace(".", "-", regex=False)

            # 输出dose_str样例以便调试
            sample_doses_inf = cellline_adata.obs[['dose', 'dose_str']].drop_duplicates().head(5)
            logger.info(f"  dose_str样例 (inference):\n{sample_doses_inf.to_string()}")

            unique_conditions = cellline_adata.obs[["cell_line", "drug", "dose_str", "dose"]].drop_duplicates()
            logger.info(f"  发现 {len(unique_conditions)} 个独特条件")

            # 4. 预计算所有条件的sigma（仅基于真实数据）
            logger.info(f"  预计算所有条件的sigma（基于真实数据，确保不同算法间可比性）...")
            from chemcpa_distribution_metrics import _median_heuristic_sigma_from_real_only

            condition_sigmas = {}  # 存储每个条件的sigma

            for _, condition_row in unique_conditions.iterrows():
                cl, drug, dose_str, dose = condition_row["cell_line"], condition_row["drug"], condition_row["dose_str"], condition_row["dose"]
                condition_name = f"{cl}_{drug}_{dose_str}"

                try:
                    # 获取真实数据
                    test_mask = (
                        (adata_test.obs['cell_line'] == cl) &
                        (adata_test.obs['drug'] == drug) &
                        (adata_test.obs['dose_str'] == dose_str)
                    )
                    actual_cells = adata_test[test_mask]

                    if actual_cells.n_obs > 0:
                        actual_expr = actual_cells.X.toarray() if hasattr(actual_cells.X, 'toarray') else actual_cells.X

                        # 仅基于真实数据计算sigma
                        sigma = _median_heuristic_sigma_from_real_only(actual_expr, max_samples=2000, rng=42)
                        condition_sigmas[condition_name] = sigma
                        logger.debug(f"    条件 {condition_name}: sigma={sigma:.4f}")
                except Exception as e:
                    logger.warning(f"    预计算sigma失败 {condition_name}: {e}")
                    condition_sigmas[condition_name] = None

            logger.info(f"  预计算完成，共 {len(condition_sigmas)} 个条件的sigma")

            # 5. 计算每个条件的分布度量（使用预计算的sigma）
            cellline_condition_metrics = []

            for _, condition_row in unique_conditions.iterrows():
                cl, drug, dose_str, dose = condition_row["cell_line"], condition_row["drug"], condition_row["dose_str"], condition_row["dose"]
                condition_name = f"{cl}_{drug}_{dose_str}"

                # 获取该条件的预计算sigma
                sigma = condition_sigmas.get(condition_name, None)

                try:
                    inf_mask = cellline_adata.obs['condition'] == condition_name
                    condition_pred = predictions[inf_mask]

                    if condition_pred.shape[0] == 0:
                        logger.warning(f"    条件 {condition_name} 无预测数据")
                        continue

                    test_mask = (
                        (adata_test.obs['cell_line'] == cl) &
                        (adata_test.obs['drug'] == drug) &
                        (adata_test.obs['dose_str'] == dose_str)
                    )
                    actual_cells = adata_test[test_mask]

                    if actual_cells.n_obs == 0:
                        logger.warning(f"    条件 {condition_name} 无真实数据")
                        # 调试信息：检查每个维度的匹配情况
                        cl_match = (adata_test.obs['cell_line'] == cl).sum()
                        drug_match = (adata_test.obs['drug'] == drug).sum()
                        dose_match = (adata_test.obs['dose_str'] == dose_str).sum()
                        logger.debug(f"      test_data中: cell_line={cl}({cl_match}), drug={drug}({drug_match}), dose_str={dose_str}({dose_match})")

                        # 检查是否dose格式问题
                        dose_variants = adata_test.obs[adata_test.obs['cell_line'] == cl]['dose_str'].unique()
                        if len(dose_variants) > 0 and len(dose_variants) < 20:
                            logger.debug(f"      该cell_line的可用dose_str: {sorted(dose_variants[:10])}")
                        continue

                    actual_expr = actual_cells.X.toarray() if hasattr(actual_cells.X, 'toarray') else actual_cells.X

                    from chemcpa_distribution_metrics import compute_condition_distribution_metrics

                    condition_result = compute_condition_distribution_metrics(
                        real_expr=actual_expr,
                        pred_expr=condition_pred,
                        condition_name=condition_name,
                        cell_line=cl,
                        drug=drug,
                        dose=dose_str,
                        subsample=5000,
                        rng=42,
                        mmd_sigma=sigma,  # 使用预计算的固定sigma
                        sw_projections=128,
                        sw_grid_size=400,
                        ot_reg=None,
                        ot_subsample=2000
                    )

                    # 记录使用的sigma到结果中
                    condition_result['mmd_sigma_used'] = sigma

                    cellline_condition_metrics.append(condition_result)

                    if condition_result["status"] == "success":
                        logger.info(f"    {condition_name}: MMD={condition_result['MMD_RBF']:.4f}, "
                                   f"E-dist={condition_result['E_distance']:.4f}, "
                                   f"SW={condition_result['Wasserstein_Sliced']:.4f}")

                except Exception as e:
                    logger.warning(f"    计算条件 {condition_name} 失败: {e}")
                    continue

            if cellline_condition_metrics:
                successful_metrics = [c for c in cellline_condition_metrics if c.get("status") == "success"]

                if successful_metrics:
                    dist_metrics_cols = ['MMD_RBF', 'E_distance', 'Wasserstein_Sliced', 'Wasserstein_OT']
                    avg_dist_metrics = {}

                    for metric in dist_metrics_cols:
                        values = [c[metric] for c in successful_metrics if not np.isnan(c[metric])]
                        if values:
                            avg_dist_metrics[f"avg_{metric}"] = round(float(np.mean(values)), 6)
                            avg_dist_metrics[f"std_{metric}"] = round(float(np.std(values)), 6)
                        else:
                            avg_dist_metrics[f"avg_{metric}"] = None
                            avg_dist_metrics[f"std_{metric}"] = None

                    cellline_result = {
                        "cell_line": cell_line,
                        "status": "completed",
                        "n_conditions": len(cellline_condition_metrics),
                        "n_successful_conditions": len(successful_metrics),
                        "total_cells_analyzed": sum(c.get('n_pred_cells', 0) for c in successful_metrics),
                        "total_real_cells": sum(c.get('n_real_cells', 0) for c in successful_metrics),
                        **avg_dist_metrics
                    }
                    all_cellline_metrics.append(cellline_result)

                    logger.info(f"  ✅ {cell_line} 完成: {len(successful_metrics)} 成功条件")
                else:
                    logger.warning(f"  ⚠️ {cell_line} 无成功条件")

            all_condition_metrics.extend(cellline_condition_metrics)
            total_inference_cells += cellline_adata.n_obs

            # 彻底清理当前cell line的所有变量（不删除adata_test）
            del cellline_adata, predictions
            if 'actual_expr' in locals():
                del actual_expr
            if 'condition_pred' in locals():
                del condition_pred
            if 'actual_cells' in locals():
                del actual_cells
            if 'cellline_condition_metrics' in locals():
                del cellline_condition_metrics
            if 'successful_metrics' in locals():
                del successful_metrics
            if 'unique_conditions' in locals():
                del unique_conditions
            if 'condition_result' in locals():
                del condition_result
            gc.collect()

            logger.info(f"  🧹 {cell_line} 内存已清理")

        except Exception as e:
            logger.error(f"处理 {cell_line} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")

            # 即使出错也要清理内存（只删除循环内的临时变量，不删除adata_test）
            try:
                if 'cellline_adata' in locals():
                    del cellline_adata
                if 'predictions' in locals():
                    del predictions
                if 'actual_expr' in locals():
                    del actual_expr
                if 'condition_pred' in locals():
                    del condition_pred
                if 'actual_cells' in locals():
                    del actual_cells
                if 'cellline_condition_metrics' in locals():
                    del cellline_condition_metrics
                if 'successful_metrics' in locals():
                    del successful_metrics
                if 'unique_conditions' in locals():
                    del unique_conditions
                if 'condition_result' in locals():
                    del condition_result
                gc.collect()
                logger.info(f"  🧹 异常后内存已清理")
            except:
                pass
            continue

    del adata_test
    gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info("内存高效的分cell line分布度量分析完成!")
    logger.info(f"处理了 {len(h5ad_files)} 个cell lines")
    logger.info(f"计算了 {len(all_condition_metrics)} 个condition metrics")
    logger.info(f"总推理细胞数: {total_inference_cells}")
    logger.info(f"{'='*60}")

    return all_cellline_metrics, all_condition_metrics

def analyze_distribution_similarity_results(cell_line_results, all_condition_metrics):
    """分析分布相似性结果"""
    logger.info("分析分布相似性结果...")
    successful_results = [r for r in cell_line_results if r.get("status") == "completed"]

    if not successful_results:
        logger.error("没有成功的结果可供分析")
        return {}

    logger.info(f"成功分析了 {len(successful_results)} 个细胞系")
    logger.info(f"总条件分析: {len(all_condition_metrics)}")

    df_cellline = pd.DataFrame(successful_results)
    dist_metrics_cols = ['avg_MMD_RBF', 'avg_E_distance', 'avg_Wasserstein_Sliced', 'avg_Wasserstein_OT']

    cellline_stats = {}
    logger.info("\n细胞系级别分布相似性统计:")

    for col in dist_metrics_cols:
        if col in df_cellline.columns:
            valid_values = df_cellline[col].dropna()
            if len(valid_values) > 0:
                mean_val = float(valid_values.mean())
                std_val = float(valid_values.std())
                median_val = float(valid_values.median())
                min_val = float(valid_values.min())
                max_val = float(valid_values.max())

                cellline_stats[f"cellline_{col}"] = {
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'min': min_val,
                    'max': max_val
                }

                logger.info(f"  {col}:")
                logger.info(f"    平均: {mean_val:.4f} ± {std_val:.4f}")
                logger.info(f"    中位数: {median_val:.4f}")
                logger.info(f"    范围: [{min_val:.4f}, {max_val:.4f}]")

    condition_stats = {}
    if all_condition_metrics:
        df_condition = pd.DataFrame(all_condition_metrics)
        condition_dist_cols = ['MMD_RBF', 'E_distance', 'Wasserstein_Sliced', 'Wasserstein_OT']

        logger.info("\n条件级别分布相似性统计:")

        for col in condition_dist_cols:
            if col in df_condition.columns:
                valid_values = df_condition[col].dropna()
                if len(valid_values) > 0:
                    mean_val = float(valid_values.mean())
                    std_val = float(valid_values.std())
                    median_val = float(valid_values.median())

                    condition_stats[f"condition_{col}"] = {
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val
                    }

                    logger.info(f"  {col}: 平均={mean_val:.4f} ± {std_val:.4f}, 中位数={median_val:.4f}")

    if 'avg_MMD_RBF' in df_cellline.columns:
        best_mmd_idx = df_cellline['avg_MMD_RBF'].idxmin()
        worst_mmd_idx = df_cellline['avg_MMD_RBF'].idxmax()

        best_cellline = df_cellline.loc[best_mmd_idx]
        worst_cellline = df_cellline.loc[worst_mmd_idx]

        logger.info(f"\n最佳分布匹配: {best_cellline['cell_line']} (MMD={best_cellline['avg_MMD_RBF']:.4f})")
        logger.info(f"最差分布匹配: {worst_cellline['cell_line']} (MMD={worst_cellline['avg_MMD_RBF']:.4f})")

    if 'avg_MMD_RBF' in df_cellline.columns:
        mmd_values = df_cellline['avg_MMD_RBF'].dropna()
        excellent = (mmd_values <= 0.1).sum()
        good = ((mmd_values > 0.1) & (mmd_values <= 0.3)).sum()
        poor = (mmd_values > 0.3).sum()

        logger.info("\n分布相似性表现分布:")
        logger.info(f"  优秀 (MMD ≤ 0.1): {excellent} 个细胞系 ({excellent/len(mmd_values)*100:.1f}%)")
        logger.info(f"  良好 (0.1 < MMD ≤ 0.3): {good} 个细胞系 ({good/len(mmd_values)*100:.1f}%)")
        logger.info(f"  较差 (MMD > 0.3): {poor} 个细胞系 ({poor/len(mmd_values)*100:.1f}%)")

    all_stats = {**cellline_stats, **condition_stats}

    return all_stats

def get_device(device_preference="auto"):
    """获取设备"""
    if device_preference == "auto":
        return "cuda:3" if torch.cuda.is_available() else "cpu"
    elif device_preference.startswith("cuda"):
        if torch.cuda.is_available():
            return device_preference
        else:
            logger.warning("GPU不可用，回退到CPU")
            return "cpu"
    return device_preference

def load_inference_data():
    """智能数据加载：只使用分cell line预计算结果"""
    logger.info("🔍 检测数据源...")
    data_source = detect_data_source()

    if data_source == "precomputed_cellline":
        inference_adata, adata_test, mapping_df = load_precomputed_cellline_data()
        if inference_adata is not None:
            logger.info("🚀 成功加载预计算的分cell line推理结果")
            return inference_adata, adata_test, mapping_df
        else:
            raise FileNotFoundError(f"预计算分cell line数据加载失败，请确保 {PRECOMPUTED_DIR}/ 存在")
    else:
        raise FileNotFoundError(
            "未检测到预计算结果！请先运行 cpa_random_counterfactual_inference_final_full_modified.py 生成推理结果。\n"
            f"需要的文件夹: {PRECOMPUTED_DIR}/"
        )

def apply_training_consistent_mapping(adata_all, training_drug_to_idx, training_covariate_mappings):
    """
    将inference数据的映射调整为与训练时一致的索引
    处理inference数据中可能缺失的药物/细胞系问题
    """
    logger.info("应用训练时一致的数据映射...")
    drug_indices = []
    missing_drugs = []
    
    for drug in adata_all.obs["drug"]:
        if drug in training_drug_to_idx:
            drug_indices.append(training_drug_to_idx[drug])
        else:
            missing_drugs.append(drug)
            drug_indices.append(0)  
    
    adata_all.obs['drug_idx'] = drug_indices
    
    if missing_drugs:
        unique_missing = list(set(missing_drugs))
        logger.warning(f"检测到 {len(unique_missing)} 个训练时未见的药物: {unique_missing[:10]}...")
        logger.warning(f"这些药物将使用默认索引 0")
    
    if training_covariate_mappings and len(training_covariate_mappings) > 0:
        cell_line_to_idx = training_covariate_mappings[0]
        cell_line_indices = []
        missing_cell_lines = []
        
        for cell_line in adata_all.obs["cell_line"]:
            if cell_line in cell_line_to_idx:
                cell_line_indices.append(cell_line_to_idx[cell_line])
            else:
                missing_cell_lines.append(cell_line)
                cell_line_indices.append(0)
        
        adata_all.obs['cell_line_idx'] = cell_line_indices
        
        if missing_cell_lines:
            unique_missing_cells = list(set(missing_cell_lines))
            logger.warning(f"检测到 {len(unique_missing_cells)} 个训练时未见的细胞系: {unique_missing_cells[:5]}...")
            logger.warning(f"这些细胞系将使用默认索引 0")
        
        logger.info(f"数据映射完成: drug_idx范围[{min(drug_indices)}, {max(drug_indices)}], cell_line_idx范围[{min(cell_line_indices)}, {max(cell_line_indices)}]")
    else:
        logger.warning("未找到协变量映射，跳过cell_line_idx处理")
        logger.info(f"数据映射完成: drug_idx范围[{min(drug_indices)}, {max(drug_indices)}]")
    
    return training_drug_to_idx, training_covariate_mappings

def find_control_group(adata_all):
    """查找控制组 - CPA风格简化"""
    logger.info("查找控制组...")
    global_control = "DMSO-TF"
    
    control_mask = adata_all.obs["drug"] == global_control
    control_count = np.sum(control_mask)
    
    if control_count == 0:
        logger.warning(f"控制组 {global_control} 未找到")
        logger.info("检查可用药物类型:")
        all_drugs = set(adata_all.obs["drug"].unique())
        logger.info(f"可用药物: {sorted(all_drugs)}")
        
        dmso_alternatives = [drug for drug in all_drugs if "DMSO" in drug.upper()]
        if dmso_alternatives:
            global_control = dmso_alternatives[0]
            control_count = np.sum(adata_all.obs["drug"] == global_control)
            logger.info(f"使用替代控制组: {global_control} ({control_count} 样本)")
        else:
            logger.error("未找到合适的控制组")
            raise ValueError("未找到合适的控制组")
    else:
        logger.info(f"使用控制组: {global_control} ({control_count} 样本)")
    
    del control_mask
    return global_control

def load_trained_chemcpa_model(model_path, adata_all, global_control, prefer_best=True):
    """加载训练好的ChemCPA模型 - 优先加载最佳模型"""
    logger.info(f"加载训练好的ChemCPA模型...")
    actual_model_path = model_path
    
    if prefer_best:
        best_model_path = model_path.replace("chemcpa_pretrain_model.pth", "chemcpa_pretrain_model_best.pth")
        
        if os.path.exists(best_model_path):
            actual_model_path = best_model_path
            logger.info(f"🎯 找到最佳模型，将加载: {best_model_path}")
        elif os.path.exists(model_path):
            logger.info(f"未找到专用最佳模型文件，加载主模型: {model_path}")
            
            try:
                with open(model_path, 'rb') as f:
                    temp_state = pickle.load(f)
                if 'best_epoch' in temp_state:
                    logger.info(f"✨ 主模型包含最佳状态 (epoch {temp_state['best_epoch']})")
                else:
                    logger.info("主模型包含当前训练状态")
            except:
                pass
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        logger.info(f"加载指定模型: {model_path}")
    
    try:
        with open(actual_model_path, 'rb') as f:
            model_state = pickle.load(f)
        
        logger.info(f"模型类型: {model_state.get('model_type', 'unknown')}")
        logger.info(f"SMILES特征: {model_state.get('n_smiles_features', 0)}")
        
        if model_state.get('is_best_model', False):
            logger.info(f"🏆 加载最佳模型状态:")
            logger.info(f"   最佳epoch: {model_state.get('best_epoch', 'N/A')}")
            logger.info(f"   最佳验证loss: {model_state.get('best_val_loss', 'N/A')}")
            logger.info(f"   最佳训练loss: {model_state.get('best_train_loss', 'N/A')}")
        elif 'best_epoch' in model_state:
            logger.info(f"📊 主模型包含最佳状态:")
            logger.info(f"   最佳epoch: {model_state.get('best_epoch', 'N/A')}")
            logger.info(f"   最佳验证loss: {model_state.get('best_val_loss', 'N/A')}")
        
        import warnings
        from io import StringIO
        
        warnings.filterwarnings('ignore')
        sc.settings.verbosity = 0
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            from chemcpa_implementation import ChemCPAWithSMILES
            
            ChemCPAWithSMILES.split_key = "split"
            ChemCPAWithSMILES.setup_anndata(
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
        
        device = get_device("auto")
        chemcpa_model = ChemCPAWithSMILES(
            adata=adata_all,
            n_smiles_features=model_state.get('n_smiles_features', 0),
            device=device
        )
        
        training_drug_to_idx = None
        training_covariate_mappings = None
        
        if 'drug_to_idx' in model_state and 'covariate_mappings' in model_state:
            training_drug_to_idx = model_state['drug_to_idx']
            training_covariate_mappings = model_state['covariate_mappings']
            logger.info(f"从模型中恢复直接映射: {len(training_drug_to_idx)} 药物, {[len(m) for m in training_covariate_mappings]} 协变量")
        elif 'registry' in model_state:
            logger.info("从模型 registry 中恢复映射...")
            registry = model_state['registry']
            training_drug_to_idx = registry.get("drug_mapping", {})
            training_covariate_mappings = registry.get("covariate_mappings", [])
            logger.info(f"从 registry 恢复映射: {len(training_drug_to_idx)} 药物, {[len(m) for m in training_covariate_mappings]} 协变量")
        
        if training_drug_to_idx and training_covariate_mappings:
            drug_to_idx, covariate_mappings = apply_training_consistent_mapping(
                adata_all, training_drug_to_idx, training_covariate_mappings
            )
            
            chemcpa_model.drug_categories = list(training_drug_to_idx.keys())
            if training_covariate_mappings:
                chemcpa_model.cell_categories = list(training_covariate_mappings[0].keys())
            
            logger.info("✅ 训练时一致的映射已应用")
        else:
            logger.warning("⚠️ 模型文件中未找到映射信息 - 预测可能会有一致性问题")
            logger.warning("将使用inference数据创建新映射，可能导致索引不一致")
            
            from chemcpa_implementation import prepare_chemcpa_data
            drug_to_idx, covariate_mappings = prepare_chemcpa_data(
                adata_all, "drug", global_control, "dose", ["cell_line"]
            )

        if 'model_state_dict' in model_state:
            logger.info("恢复模型权重...")
            try:
                from chemcpa_adversarial import AdversarialStandardLossTrainer
                
                chemcpa_model.trainer = AdversarialStandardLossTrainer(
                    adata=adata_all,
                    n_smiles_features=model_state.get('n_smiles_features', 0),
                    device=device
                )
                chemcpa_model.trainer.initialize_model()
                chemcpa_model.trainer.model.load_state_dict(model_state['model_state_dict'], strict=False)
                chemcpa_model.model = chemcpa_model.trainer.model
                
                if model_state.get('is_best_model', False) or 'best_epoch' in model_state:
                    logger.info("✅ 最佳模型权重恢复成功")
                else:
                    logger.info("✅ 模型权重恢复成功")
                    
            except Exception as e:
                logger.warning(f"恢复模型权重失败: {e}")
                logger.info("将在预测时重新初始化模型结构")
        
        if 'loss_history' in model_state:
            from chemcpa_training import LossHistory
            loss_history = LossHistory()
            history_data = model_state['loss_history']
            loss_history.epochs = history_data.get('epochs', [])
            loss_history.train_losses = history_data.get('train_losses', [])
            loss_history.val_losses = history_data.get('val_losses', [])
            chemcpa_model.loss_history = loss_history
        
        logger.info("✅ ChemCPA模型加载成功")
        
        if hasattr(chemcpa_model, 'drug_categories') and hasattr(chemcpa_model, 'cell_categories'):
            logger.info(f"最终映射状态: {len(chemcpa_model.drug_categories)} 药物, {len(chemcpa_model.cell_categories)} 细胞系")
        
        return chemcpa_model
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise

def load_drug_metadata_for_smiles(metadata_path):
    """加载药物元数据用于SMILES特征生成"""
    logger.info(f"加载药物元数据用于SMILES: {metadata_path}")
    try:
        drug_metadata = pd.read_csv(metadata_path)
        logger.info(f"加载了 {len(drug_metadata)} 个药物的元数据")
        
        if 'canonical_smiles' not in drug_metadata.columns:
            raise ValueError("未找到 'canonical_smiles' 列")
        
        drug_metadata['drug_clean'] = (
            drug_metadata['drug']
            .astype(str)
            .str.replace("_", "-", regex=False)
        )
        
        return drug_metadata
        
    except Exception as e:
        logger.error(f"加载药物元数据失败: {e}")
        raise

def integrate_smiles_for_inference(adata_all, drug_metadata, config):
    """为推理数据集成SMILES特征"""
    logger.info("为推理数据集成SMILES特征...")
    smiles_encoder = SMILESEncoder(
        method=config.smiles.encoding_method,
        n_bits=config.smiles.morgan_n_bits,
        radius=config.smiles.morgan_radius,
        n_descriptors=config.smiles.rdkit_n_descriptors
    )
    
    all_drugs = set(adata_all.obs['drug'].unique())
    logger.info(f"发现 {len(all_drugs)} 个不同的药物")
    
    drug_to_smiles = {}
    missing_drugs = []
    
    for drug in all_drugs:
        matches = drug_metadata[drug_metadata['drug_clean'] == drug]
        if len(matches) > 0:
            smiles = matches.iloc[0]['canonical_smiles']
            drug_to_smiles[drug] = smiles
        else:
            matches = drug_metadata[drug_metadata['drug'] == drug]
            if len(matches) > 0:
                smiles = matches.iloc[0]['canonical_smiles']
                drug_to_smiles[drug] = smiles
            else:
                missing_drugs.append(drug)
                drug_to_smiles[drug] = None
        del matches
    
    logger.info(f"找到SMILES的药物: {len(drug_to_smiles) - len(missing_drugs)}")
    if missing_drugs:
        logger.warning(f"未找到SMILES的药物 ({len(missing_drugs)}): {missing_drugs[:5]}...")
    
    all_smiles = [drug_to_smiles[drug] for drug in all_drugs]
    smiles_features = smiles_encoder.fit_transform(all_smiles)
    
    drug_to_features = {}
    for drug, features in zip(all_drugs, smiles_features):
        drug_to_features[drug] = features
    
    n_obs = adata_all.n_obs
    n_features = smiles_features.shape[1]
    smiles_matrix = np.zeros((n_obs, n_features), dtype=np.float32)
    
    for j, drug in enumerate(adata_all.obs['drug']):
        smiles_matrix[j] = drug_to_features[drug]
    
    adata_all.obsm['smiles_features'] = smiles_matrix
    logger.info(f"添加SMILES特征形状: {smiles_matrix.shape}")
    
    del all_smiles, drug_to_features, missing_drugs, smiles_matrix
    return drug_to_smiles, smiles_features.shape[1]

def perform_single_cell_counterfactual_analysis_by_cellline(chemcpa_model, adata_all, global_control):
    """按cell_line分批执行counterfactual分析 - 优化内存使用"""
    logger.info("开始按cell_line分批的单细胞counterfactual分析...")
    logger.info(f"原始总样本数: {adata_all.n_obs}")
    is_valid_drug_mask = adata_all.obs['drug'] != 'nan'
    adata_all = adata_all[is_valid_drug_mask].copy()
    logger.info(f"清理'nan' drug后，剩余总样本数: {adata_all.n_obs}")

    ctrl_data = adata_all[adata_all.obs["split"] == "ctrl"].copy()
    test_data = adata_all[adata_all.obs["split"] == "test"].copy()

    logger.info(f"有效控制样本: {ctrl_data.n_obs}")
    logger.info(f"有效测试样本: {test_data.n_obs}")

    test_conditions_df = test_data.obs[["cell_line", "drug", "dose_str", "dose"]].drop_duplicates()
    test_conditions = [tuple(row) for row in test_conditions_df.itertuples(index=False)]
    unique_cell_lines = test_conditions_df['cell_line'].unique()

    logger.info(f"发现 {len(test_conditions)} 个唯一测试条件")
    logger.info(f"需要处理 {len(unique_cell_lines)} 个cell_lines: {unique_cell_lines}")

    output_dir = "./distribution_similarity_result"
    cellline_results_dir = f"{output_dir}/cellline_results"
    os.makedirs(cellline_results_dir, exist_ok=True)

    all_cellline_metrics = []
    all_condition_metrics = []
    all_inference_metadata = []

    for cell_line_idx, cell_line in enumerate(unique_cell_lines, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"处理Cell Line [{cell_line_idx}/{len(unique_cell_lines)}]: {cell_line}")
        logger.info(f"{'='*60}")

        try:
            cellline_ctrl_mask = (ctrl_data.obs['cell_line'] == cell_line)
            cellline_controls = ctrl_data[cellline_ctrl_mask]

            if cellline_controls.n_obs == 0:
                logger.warning(f"Cell line {cell_line} 没有控制细胞，跳过")
                continue

            cellline_conditions = [(cl, drug, dose_str, dose) for cl, drug, dose_str, dose in test_conditions if cl == cell_line]
            logger.info(f"Cell line {cell_line}: {cellline_controls.n_obs} 控制细胞, {len(cellline_conditions)} 条件")

            cellline_inference_cells = []
            cellline_metadata = []
            cellline_mapping = []

            for condition_idx, (_, drug, dose_str, dose) in enumerate(cellline_conditions, 1):
                logger.info(f"  处理条件 [{condition_idx}/{len(cellline_conditions)}]: {drug} @ {dose_str}")

                try:
                    test_mask = (
                        (test_data.obs['cell_line'] == cell_line) &
                        (test_data.obs['drug'] == drug) &
                        (test_data.obs['dose_str'] == dose_str)
                    )
                    actual_perturbed_cells = test_data[test_mask]

                    if actual_perturbed_cells.n_obs == 0:
                        logger.warning(f"    条件 {drug}@{dose_str} 没有实际细胞，跳过")
                        continue

                    logger.info(f"    实际扰动细胞数: {actual_perturbed_cells.n_obs}")

                    for ctrl_idx in range(cellline_controls.n_obs):
                        control_cell = cellline_controls[ctrl_idx:ctrl_idx+1].copy()
                        ctrl_cell_idx = cellline_controls.obs.index[ctrl_idx]

                        ctrl_expr = control_cell.X.toarray() if hasattr(control_cell.X, 'toarray') else control_cell.X
                        cellline_inference_cells.append(ctrl_expr.astype(np.float32)[0])

                        metadata = {
                            "cell_line": cell_line,
                            "drug": drug,
                            "dose_str": dose_str,
                            "dose": float(dose),
                            "condition": f"{cell_line}_{drug}_{dose_str}",
                            "original_ctrl_cell_idx": ctrl_cell_idx,
                            "n_real_condition_cells": actual_perturbed_cells.n_obs,
                            "inference_cell_idx": len(cellline_metadata)
                        }
                        cellline_metadata.append(metadata)

                        cellline_mapping.append({
                            "inference_idx": len(cellline_metadata) - 1,
                            "condition": f"{cell_line}_{drug}_{dose_str}",
                            "ctrl_cell_idx": ctrl_cell_idx,
                            "cell_line": cell_line,
                            "drug": drug,
                            "dose_str": dose_str
                        })

                        del ctrl_expr

                except Exception as e:
                    logger.warning(f"    处理条件 {drug}@{dose_str} 失败: {e}")
                    continue

            if not cellline_inference_cells:
                logger.warning(f"Cell line {cell_line} 没有有效的inference数据，跳过")
                continue

            logger.info(f"Cell line {cell_line} 收集完成: {len(cellline_inference_cells)} inference cells")

            logger.info(f"为Cell line {cell_line} 创建inference AnnData...")
            cellline_inference_matrix = np.vstack(cellline_inference_cells).astype(np.float32)
            cellline_metadata_df = pd.DataFrame(cellline_metadata)

            cellline_inference_adata = sc.AnnData(X=cellline_inference_matrix, obs=cellline_metadata_df)
            cellline_inference_adata.var_names = adata_all.var_names

            try:
                warnings.filterwarnings('ignore')
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = StringIO(), StringIO()
                try:
                    ChemCPAWithSMILES.setup_anndata(
                        adata=cellline_inference_adata,
                        perturbation_key="drug",
                        control_group=global_control,
                        dosage_key="dose",
                        categorical_covariate_keys=["cell_line"]
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    warnings.filterwarnings('default')
            except Exception as e:
                logger.error(f"设置Cell line {cell_line} inference_adata失败: {e}")
                continue

            if 'smiles_features' in adata_all.obsm:
                n_smiles_features = adata_all.obsm['smiles_features'].shape[1]
                smiles_matrix = np.zeros((cellline_inference_adata.n_obs, n_smiles_features), dtype=np.float32)
                for j, drug in enumerate(cellline_inference_adata.obs['drug']):
                    drug_mask = adata_all.obs['drug'] == drug
                    drug_indices = np.where(drug_mask)[0]
                    if len(drug_indices) > 0:
                        smiles_matrix[j] = adata_all.obsm['smiles_features'][drug_indices[0]]
                cellline_inference_adata.obsm['smiles_features'] = smiles_matrix
                del smiles_matrix

            def _get_training_vocab(model):
                cats_drug = getattr(model, "drug_categories", None)
                cats_cell = getattr(model, "cell_categories", None)
                if cats_drug is not None:
                    return list(cats_drug), (list(cats_cell) if cats_cell is not None else None)
                raise RuntimeError("找不到训练词表（drug/cell_line）。请确认模型保存了 registry。")
            cats_drug, cats_cell = _get_training_vocab(chemcpa_model)

            original_drugs = cellline_inference_adata.obs["drug"].astype(str)
            cellline_inference_adata.obs["drug"] = pd.Categorical(original_drugs, categories=cats_drug)
            cellline_inference_adata.obs["drug_idx"] = cellline_inference_adata.obs["drug"].cat.codes.astype("int64")
            
            drug_mask = cellline_inference_adata.obs["drug_idx"] == -1
            if drug_mask.any():
                n_missing = drug_mask.sum()
                missing_drugs = original_drugs[drug_mask].unique().tolist()
                logger.warning(f"    为 {cell_line} 推理时发现 {n_missing} 个样本含有训练时未见的药物: {missing_drugs[:5]}")
                logger.warning(f"    这些药物将被映射到默认索引 0")
                cellline_inference_adata.obs.loc[drug_mask, "drug_idx"] = 0
            
            if cats_cell:
                original_cell_lines = cellline_inference_adata.obs["cell_line"].astype(str)
                cellline_inference_adata.obs["cell_line"] = pd.Categorical(original_cell_lines, categories=cats_cell)
                cellline_inference_adata.obs["cell_line_idx"] = cellline_inference_adata.obs["cell_line"].cat.codes.astype("int64")
            
                cell_mask = cellline_inference_adata.obs["cell_line_idx"] == -1
                if cell_mask.any():
                    n_missing = cell_mask.sum()
                    missing_cells = original_cell_lines[cell_mask].unique().tolist()
                    logger.warning(f"    为 {cell_line} 推理时发现 {n_missing} 个样本含有训练时未见的细胞系: {missing_cells[:5]}")
                    logger.warning(f"    这些细胞系将被映射到默认索引 0")
                    cellline_inference_adata.obs.loc[cell_mask, "cell_line_idx"] = 0
            
            cellline_inference_adata.obs["dose"] = cellline_inference_adata.obs["dose"].astype(np.float32)
            if "dose_value" in getattr(chemcpa_model, "required_obs_keys", []):
                cellline_inference_adata.obs["dose_value"] = cellline_inference_adata.obs["dose"]
                
            logger.info(f"执行Cell line {cell_line} 的counterfactual inference...")
            try:
                cellline_inference_with_pred = chemcpa_model.predict(cellline_inference_adata)
                predictions = cellline_inference_with_pred.obsm["CPA_pred"]
                if not isinstance(predictions, np.ndarray):
                    predictions = predictions.toarray()
                predictions = predictions.astype(np.float32)

                logger.info(f"Cell line {cell_line} inference完成: {predictions.shape}")

            except Exception as e:
                logger.error(f"Cell line {cell_line} inference失败: {e}")
                continue

            logger.info(f"计算Cell line {cell_line} 的metrics...")
            cellline_condition_metrics = []

            for condition in cellline_conditions:
                _, drug, dose_str, dose = condition
                condition_name = f"{cell_line}_{drug}_{dose_str}"

                try:
                    inf_mask = cellline_inference_with_pred.obs['condition'] == condition_name
                    condition_inf_pred = predictions[inf_mask]

                    test_mask = (
                        (test_data.obs['cell_line'] == cell_line) &
                        (test_data.obs['drug'] == drug) &
                        (test_data.obs['dose_str'] == dose_str)
                    )
                    actual_cells = test_data[test_mask]

                    if actual_cells.n_obs == 0 or condition_inf_pred.shape[0] == 0:
                        continue

                    actual_expr = actual_cells.X.toarray() if hasattr(actual_cells.X, 'toarray') else actual_cells.X
                    actual_expr = actual_expr.astype(np.float32)

                    condition_result = compute_condition_distribution_metrics(
                        real_expr=actual_expr,
                        pred_expr=condition_inf_pred,
                        condition_name=condition_name,
                        cell_line=cell_line,
                        drug=drug,
                        dose=dose_str,
                        subsample=5000,
                        rng=42,
                        mmd_sigma=None,
                        sw_projections=128,
                        sw_grid_size=400,
                        ot_reg=None,
                        ot_subsample=2000
                    )

                    cellline_condition_metrics.append(condition_result)

                    if condition_result["status"] == "success":
                        logger.info(f"    {condition_name}: MMD={condition_result['MMD_RBF']:.4f}, "
                                   f"E-dist={condition_result['E_distance']:.4f}, "
                                   f"SW={condition_result['Wasserstein_Sliced']:.4f}")
                    else:
                        logger.warning(f"    {condition_name}: {condition_result['status']}")

                except Exception as e:
                    logger.warning(f"    计算条件 {condition_name} metrics失败: {e}")
                    continue

            logger.info(f"保存Cell line {cell_line} 的结果...")

            cellline_inference_with_pred.X = cellline_inference_with_pred.X.astype(np.float32)
            if hasattr(cellline_inference_with_pred.X, 'toarray'):
                cellline_inference_with_pred.X = cellline_inference_with_pred.X.toarray().astype(np.float32)

            if (cellline_inference_with_pred.obs.index.name and
                cellline_inference_with_pred.obs.index.name in cellline_inference_with_pred.obs.columns):
                original_name = cellline_inference_with_pred.obs.index.name
                if not cellline_inference_with_pred.obs.index.equals(cellline_inference_with_pred.obs[original_name]):
                    cellline_inference_with_pred.obs = cellline_inference_with_pred.obs.drop(columns=[original_name])
                cellline_inference_with_pred.obs.index.name = f"{original_name}_cell_index"

            inf_output_path = f"{cellline_results_dir}/{cell_line}_inference_results.h5ad"
            cellline_inference_with_pred.write_h5ad(inf_output_path)

            mapping_df = pd.DataFrame(cellline_mapping)
            mapping_path = f"{cellline_results_dir}/{cell_line}_mapping.csv"
            mapping_df.to_csv(mapping_path, index=False)

            logger.info(f"Cell line {cell_line} 结果已保存: {inf_output_path}")

            all_condition_metrics.extend(cellline_condition_metrics)
            all_inference_metadata.extend(cellline_metadata)

            if cellline_condition_metrics:
                successful_metrics = [c for c in cellline_condition_metrics if c.get("status") == "success"]

                if successful_metrics:
                    dist_metrics_cols = ['MMD_RBF', 'E_distance', 'Wasserstein_Sliced', 'Wasserstein_OT']
                    avg_dist_metrics = {}

                    for metric in dist_metrics_cols:
                        values = [c[metric] for c in successful_metrics if not np.isnan(c[metric])]
                        if values:
                            avg_dist_metrics[f"avg_{metric}"] = round(float(np.mean(values)), 6)
                            avg_dist_metrics[f"std_{metric}"] = round(float(np.std(values)), 6)
                            avg_dist_metrics[f"min_{metric}"] = round(float(np.min(values)), 6)
                            avg_dist_metrics[f"max_{metric}"] = round(float(np.max(values)), 6)
                        else:
                            avg_dist_metrics[f"avg_{metric}"] = None
                            avg_dist_metrics[f"std_{metric}"] = None
                            avg_dist_metrics[f"min_{metric}"] = None
                            avg_dist_metrics[f"max_{metric}"] = None

                    cellline_result = {
                        "cell_line": cell_line,
                        "status": "completed",
                        "n_conditions": len(cellline_condition_metrics),
                        "n_successful_conditions": len(successful_metrics),
                        "total_cells_analyzed": sum(c.get('n_pred_cells', 0) for c in successful_metrics),
                        "total_real_cells": sum(c.get('n_real_cells', 0) for c in successful_metrics),
                        **avg_dist_metrics,
                        "condition_details": cellline_condition_metrics
                    }
                else:
                    cellline_result = {
                        "cell_line": cell_line,
                        "status": "no_successful_conditions",
                        "n_conditions": len(cellline_condition_metrics),
                        "n_successful_conditions": 0,
                        "condition_details": cellline_condition_metrics
                    }
                all_cellline_metrics.append(cellline_result)

            del cellline_inference_cells, cellline_metadata, cellline_mapping
            del cellline_inference_adata, cellline_inference_with_pred, predictions
            del cellline_inference_matrix, cellline_metadata_df
            gc.collect()

            logger.info(f"Cell line {cell_line} 处理完成，内存已清理")

        except Exception as e:
            logger.error(f"处理Cell line {cell_line} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            continue

    del ctrl_data, test_data
    gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info("按cell_line的counterfactual分析完成!")
    logger.info(f"处理了 {len(unique_cell_lines)} 个cell_lines")
    logger.info(f"计算了 {len(all_condition_metrics)} 个condition metrics")
    logger.info(f"{'='*60}")

    return all_cellline_metrics, all_condition_metrics, all_inference_metadata

def analyze_distribution_similarity_results(cell_line_results, all_condition_metrics):
    """分析分布相似性度量结果"""
    logger.info("分析分布相似性度量结果...")
    successful_results = [r for r in cell_line_results if r.get("status") == "completed"]

    if not successful_results:
        logger.error("没有成功的结果可供分析")
        return {}

    logger.info(f"成功分析了 {len(successful_results)} 个细胞系")
    logger.info(f"总条件分析: {len(all_condition_metrics)}")

    df_cellline = pd.DataFrame(successful_results)
    dist_metrics_cols = ['avg_MMD_RBF', 'avg_E_distance', 'avg_Wasserstein_Sliced', 'avg_Wasserstein_OT']

    cellline_stats = {}
    logger.info("\n细胞系级别分布相似性统计:")

    for col in dist_metrics_cols:
        if col in df_cellline.columns:
            valid_values = df_cellline[col].dropna()
            if len(valid_values) > 0:
                mean_val = float(valid_values.mean())
                std_val = float(valid_values.std())
                median_val = float(valid_values.median())
                min_val = float(valid_values.min())
                max_val = float(valid_values.max())

                cellline_stats[f"cellline_{col}"] = {
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'min': min_val,
                    'max': max_val
                }

                logger.info(f"  {col}:")
                logger.info(f"    平均: {mean_val:.4f} ± {std_val:.4f}")
                logger.info(f"    中位数: {median_val:.4f}")
                logger.info(f"    范围: [{min_val:.4f}, {max_val:.4f}]")

    condition_stats = {}
    if all_condition_metrics:
        successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

        if successful_conditions:
            df_condition = pd.DataFrame(successful_conditions)

            logger.info("\n条件级别分布相似性统计:")

            dist_cols = ['MMD_RBF', 'E_distance', 'Wasserstein_Sliced', 'Wasserstein_OT']
            for col in dist_cols:
                if col in df_condition.columns:
                    valid_values = df_condition[col].dropna()
                    if len(valid_values) > 0:
                        mean_val = float(valid_values.mean())
                        std_val = float(valid_values.std())
                        median_val = float(valid_values.median())

                        condition_stats[f"condition_{col}"] = {
                            'mean': mean_val,
                            'std': std_val,
                            'median': median_val
                        }

                        logger.info(f"  {col}: 平均={mean_val:.4f} ± {std_val:.4f}, 中位数={median_val:.4f}")

    if 'avg_MMD_RBF' in df_cellline.columns:
        best_mmd_idx = df_cellline['avg_MMD_RBF'].idxmin()
        worst_mmd_idx = df_cellline['avg_MMD_RBF'].idxmax()

        best_cellline = df_cellline.loc[best_mmd_idx]
        worst_cellline = df_cellline.loc[worst_mmd_idx]

        logger.info(f"\n最佳分布匹配: {best_cellline['cell_line']} (MMD={best_cellline['avg_MMD_RBF']:.4f}, {best_cellline['n_conditions']} 条件)")
        logger.info(f"最差分布匹配: {worst_cellline['cell_line']} (MMD={worst_cellline['avg_MMD_RBF']:.4f}, {worst_cellline['n_conditions']} 条件)")

    if 'avg_MMD_RBF' in df_cellline.columns:
        mmd_values = df_cellline['avg_MMD_RBF'].dropna()
        if len(mmd_values) > 0:
            excellent_performance = (mmd_values <= 0.1).sum()
            good_performance = ((mmd_values > 0.1) & (mmd_values <= 0.3)).sum()
            poor_performance = (mmd_values > 0.3).sum()

            logger.info("\n分布相似性分析 (基于MMD RBF):")
            logger.info(f"  优秀匹配 (MMD ≤ 0.1): {excellent_performance} 个细胞系 ({excellent_performance/len(mmd_values)*100:.1f}%)")
            logger.info(f"  良好匹配 (0.1 < MMD ≤ 0.3): {good_performance} 个细胞系 ({good_performance/len(mmd_values)*100:.1f}%)")
            logger.info(f"  较差匹配 (MMD > 0.3): {poor_performance} 个细胞系 ({poor_performance/len(mmd_values)*100:.1f}%)")

    all_stats = {**cellline_stats, **condition_stats}

    return all_stats

def save_single_cell_inference_results(inference_adata_with_pred, inference_to_actual_mapping):
    """保存单细胞inference结果的adata和映射信息"""
    logger.info("保存单细胞inference结果...")
    os.makedirs("./distribution_similarity_result", exist_ok=True)
    data_dir = Path(RESULT_ROOT) / "data"
    metadata_dir = Path(RESULT_ROOT) / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("确保所有数据为float32格式...")
    
    if hasattr(inference_adata_with_pred.X, 'toarray'):
        inference_adata_with_pred.X = inference_adata_with_pred.X.toarray().astype(np.float32)
    else:
        inference_adata_with_pred.X = inference_adata_with_pred.X.astype(np.float32)
    
    if "CPA_pred" in inference_adata_with_pred.obsm:
        pred = inference_adata_with_pred.obsm["CPA_pred"]
        if hasattr(pred, 'toarray'):
            pred = pred.toarray()
        inference_adata_with_pred.obsm["CPA_pred"] = pred.astype(np.float32)
    
    if 'dose' in inference_adata_with_pred.obs.columns:
        inference_adata_with_pred.obs['dose'] = inference_adata_with_pred.obs['dose'].astype(np.float32)
    if 'dose_value' in inference_adata_with_pred.obs.columns:
        inference_adata_with_pred.obs['dose_value'] = inference_adata_with_pred.obs['dose_value'].astype(np.float32)
    
    if 'smiles_features' in inference_adata_with_pred.obsm:
        inference_adata_with_pred.obsm['smiles_features'] = inference_adata_with_pred.obsm['smiles_features'].astype(np.float32)
    
    adata_path = str(data_dir / "inference_results_full.h5ad")
    logger.info(f"保存inference AnnData到: {adata_path}")
    sc.write(adata_path, inference_adata_with_pred)
    
    mapping_df = pd.DataFrame(inference_to_actual_mapping)
    mapping_path = str(metadata_dir / "inference_to_actual_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"保存映射关系到: {mapping_path}")
    
    metadata_summary = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "single_cell_counterfactual_inference",
        "description": "Single-cell level counterfactual inference results with complete metadata",
        "data_format": "float32_optimized",
        "total_cells": inference_adata_with_pred.n_obs,
        "total_genes": inference_adata_with_pred.n_vars,
        "unique_conditions": len(set(inference_adata_with_pred.obs['condition'])),
        "unique_drugs": len(set(inference_adata_with_pred.obs['drug'])),
        "unique_cell_lines": len(set(inference_adata_with_pred.obs['cell_line'])),
        "data_columns": {
            "obs_columns": list(inference_adata_with_pred.obs.columns),
            "obsm_keys": list(inference_adata_with_pred.obsm.keys()),
            "var_names_sample": list(inference_adata_with_pred.var_names[:10])
        },
        "file_paths": {
            "inference_adata": adata_path,
            "mapping_file": mapping_path
        }
    }
    
    summary_path = str(metadata_dir / "inference_metadata_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metadata_summary, f, indent=2, default=json_default)
    
    logger.info("单细胞inference结果保存完成:")
    logger.info(f"  完整AnnData: {adata_path}")
    logger.info(f"  映射关系: {mapping_path}")
    logger.info(f"  元数据摘要: {summary_path}")
    logger.info(f"  总细胞数: {inference_adata_with_pred.n_obs}")
    logger.info(f"  总基因数: {inference_adata_with_pred.n_vars}")
    logger.info(f"  数据格式: float32")
    
    return metadata_summary

def save_counterfactual_results(cell_line_results, all_condition_metrics, stats):
    """保存counterfactual分析结果"""
    logger.info("保存counterfactual分析结果...")
    os.makedirs(f"{RESULT_ROOT}/results", exist_ok=True)

    cellline_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                               for result in cell_line_results])
    cellline_df.to_csv(f"{RESULT_ROOT}/results/cellline_distribution_metrics.csv", index=False)

    condition_df = pd.DataFrame(all_condition_metrics)
    condition_df.to_csv(f"{RESULT_ROOT}/results/condition_distribution_metrics.csv", index=False)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "chemcpa_single_cell_counterfactual_analysis",
        "description": "ChemCPA single-cell counterfactual analysis with complete cell-level inference",
        "methodology": "Apply perturbations to individual control cells and compare predictions to actual perturbed cells",
        "optimization": "Single batch inference for all individual cells",
        "data_type": "float32_optimized",
        "perturbation_approach": "drug_based_single_cell_level",
        "total_cell_lines": len(cell_line_results),
        "successful_cell_lines": len([r for r in cell_line_results if r.get("status") == "completed"]),
        "total_conditions": len(all_condition_metrics),
        "performance_statistics": stats
    }
    
    with open(f"{RESULT_ROOT}/chemcpa_distribution_similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)
    
    logger.info("结果保存:")
    logger.info(f"  细胞系结果: {RESULT_ROOT}/results/cellline_distribution_metrics.csv")
    logger.info(f"  条件结果: {RESULT_ROOT}/results/condition_distribution_metrics.csv")
    logger.info(f"  完整摘要: {RESULT_ROOT}/chemcpa_distribution_similarity_summary.json")
    
    return summary

def create_distribution_visualization_plots(cell_line_results, all_condition_metrics, output_dir=RESULT_ROOT):
    """创建分布相似性度量可视化图表"""
    logger.info("创建分布相似性度量可视化图表...")
    try:
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        if cell_line_results:
            cl_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                                 for result in cell_line_results])

            if not cl_df.empty:
                dist_metrics_to_plot = ['avg_MMD_RBF', 'avg_E_distance', 'avg_Wasserstein_Sliced', 'avg_Wasserstein_OT']
                available_metrics = [m for m in dist_metrics_to_plot if m in cl_df.columns]

                if available_metrics:
                    plt.figure(figsize=(15, 10))

                    heatmap_data = cl_df.set_index('cell_line')[available_metrics]
                    sns.heatmap(heatmap_data, annot=True, cmap='viridis_r', fmt='.4f', cbar_kws={'label': 'Distance (smaller=better)'})
                    plt.title('ChemCPA Distribution Similarity: Cell Line Performance Heatmap')
                    plt.ylabel('Cell Lines')
                    plt.xlabel('Distribution Similarity Metrics')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/cellline_distribution_heatmap.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("细胞系分布相似性热图已保存")

        if all_condition_metrics:
            successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

            if successful_conditions:
                condition_df = pd.DataFrame(successful_conditions)

                if 'MMD_RBF' in condition_df.columns and len(condition_df) > 0:
                    plt.figure(figsize=(10, 6))

                    mmd_values = condition_df['MMD_RBF'].dropna()
                    if len(mmd_values) > 0:
                        plt.hist(mmd_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                        plt.axvline(mmd_values.mean(), color='red', linestyle='--',
                                   label=f'Mean: {mmd_values.mean():.4f}')
                        plt.axvline(mmd_values.median(), color='orange', linestyle='--',
                                   label=f'Median: {mmd_values.median():.4f}')

                        plt.title('ChemCPA Distribution Similarity: MMD RBF Distribution')
                        plt.xlabel('MMD RBF (smaller = better)')
                        plt.ylabel('Frequency')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/plots/mmd_distribution.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        logger.info("MMD分布图已保存")

        if all_condition_metrics:
            successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

            if successful_conditions:
                condition_df = pd.DataFrame(successful_conditions)

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                dist_metrics = ['MMD_RBF', 'E_distance', 'Wasserstein_Sliced', 'Wasserstein_OT']

                for i, metric in enumerate(dist_metrics):
                    if metric in condition_df.columns:
                        row, col = i // 2, i % 2
                        values = condition_df[metric].dropna()

                        if len(values) > 0:
                            axes[row, col].hist(values, bins=20, alpha=0.7, edgecolor='black')
                            axes[row, col].axvline(values.mean(), color='red', linestyle='--', alpha=0.7)
                            axes[row, col].set_title(f'{metric}\nMean: {values.mean():.4f}')
                            axes[row, col].set_xlabel(f'{metric} (smaller = better)')
                            axes[row, col].set_ylabel('Frequency')
                            axes[row, col].grid(True, alpha=0.3)

                plt.suptitle('ChemCPA Distribution Similarity: All Metrics Comparison', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/plots/all_metrics_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("所有度量比较图已保存")

        if all_condition_metrics:
            successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

            if successful_conditions:
                condition_df = pd.DataFrame(successful_conditions)

                if 'drug' in condition_df.columns and 'MMD_RBF' in condition_df.columns:
                    plt.figure(figsize=(15, 8))

                    drug_performance = condition_df.groupby('drug')['MMD_RBF'].agg(['mean', 'std', 'count']).reset_index()
                    drug_performance = drug_performance.sort_values('mean', ascending=True)

                    top_drugs = drug_performance.head(20)

                    plt.bar(range(len(top_drugs)), top_drugs['mean'],
                           yerr=top_drugs['std'], capsize=5, alpha=0.7, color='lightcoral')
                    plt.xticks(range(len(top_drugs)), top_drugs['drug'], rotation=45, ha='right')
                    plt.title('ChemCPA Distribution Similarity: Top 20 Drug Performance (MMD RBF)')
                    plt.xlabel('Drug')
                    plt.ylabel('Average MMD RBF (smaller = better)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/top_drug_distribution_performance.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("药物分布性能图已保存")
        
    except Exception as e:
        logger.warning(f"创建可视化图表时出现错误: {e}")

def main():
    """主函数 - ChemCPA 分布相似性分析（插件系统）"""
    logger.info("="*60)
    logger.info("ChemCPA 分布相似性分析 - 使用分布度量插件系统")
    logger.info("仅计算：MMD RBF, E-distance, Sliced Wasserstein, OT Wasserstein")
    logger.info("="*60)
    try:
        adata_all, precomputed_test_data, precomputed_mapping = load_inference_data()

        use_precomputed = (precomputed_test_data is not None and precomputed_mapping is not None)

        if use_precomputed:
            logger.info("🚀 检测到预计算结果，使用内存高效的分cell line处理模式")

            if adata_all == "EFFICIENT_MODE":
                logger.info("✅ 使用内存高效的逐cell line处理模式")
                efficient_data = precomputed_mapping

                cell_line_results, all_condition_metrics = process_cellline_distribution_metrics_efficiently(efficient_data)

                if not cell_line_results:
                    logger.error("内存高效处理失败，无结果返回")
                    return

                logger.info(f"✅ 内存高效处理完成: {len(cell_line_results)} cell lines, {len(all_condition_metrics)} conditions")

                logger.info("🎯 跳转到结果保存和分析阶段...")

                logger.info("步骤6: 保存全局汇总结果...")

                if all_condition_metrics:
                    metrics_df = pd.DataFrame(all_condition_metrics)
                    metrics_output_path = f"{RESULT_ROOT}/global_condition_metrics.csv"
                    metrics_df.to_csv(metrics_output_path, index=False)
                    logger.info(f"全局条件metrics已保存: {metrics_output_path}")

                if cell_line_results:
                    cellline_df = pd.DataFrame(cell_line_results)
                    cellline_output_path = f"{RESULT_ROOT}/global_cellline_metrics.csv"
                    cellline_df.to_csv(cellline_output_path, index=False)
                    logger.info(f"全局cell_line汇总已保存: {cellline_output_path}")

                stats = analyze_distribution_similarity_results(cell_line_results, all_condition_metrics)

                summary = save_counterfactual_results(cell_line_results, all_condition_metrics, stats)

                create_distribution_visualization_plots(cell_line_results, all_condition_metrics)

                del cell_line_results, all_condition_metrics, stats
                gc.collect()

                logger.info("="*60)
                logger.info("ChemCPA 内存高效分布相似性分析完成")
                logger.info("="*60)
                logger.info("关键改进:")
                logger.info("  ✅ 内存高效：逐cell line处理，避免449GB数据合并")
                logger.info("  ✅ 即时清理：每个cell line处理完立即释放内存")
                logger.info("  ✅ 进度可见：实时显示处理进度")
                logger.info("  ✅ 分布度量：MMD RBF, E-distance, Sliced Wasserstein, OT Wasserstein")
                logger.info("  ✅ 容错处理：单个cell line失败不影响其他")

                logger.info("\n输出文件:")
                logger.info(f"  📊 分布度量结果: {RESULT_ROOT}/results/condition_distribution_metrics.csv")
                logger.info(f"  📋 细胞系汇总: {RESULT_ROOT}/results/cellline_distribution_metrics.csv")
                logger.info(f"  📖 完整摘要: {RESULT_ROOT}/chemcpa_distribution_similarity_summary.json")
                logger.info(f"  📈 分布可视化: {RESULT_ROOT}/plots/")
                logger.info("="*60)

                return

            else:
                logger.warning("⚠️ 使用旧版预计算逻辑，可能导致内存问题")
                inference_adata_with_pred = adata_all
                test_data = precomputed_test_data
                mapping_df = precomputed_mapping

                if "CPA_pred" not in inference_adata_with_pred.obsm:
                    logger.error("预计算结果中缺少 CPA_pred 数据")
                    raise ValueError("预计算数据格式错误")

                logger.error("❌ 旧版预计算逻辑已禁用，请使用内存高效模式")
                return

        else:
            logger.info("📊 使用原始模式，需要执行完整的模型推理流程")

            global_control = find_control_group(adata_all)

            drug_metadata_path = resolve_drug_metadata_file()
            logger.info("加载SMILES特征用于化学推理...")
            drug_metadata = load_drug_metadata_for_smiles(drug_metadata_path)

            class SimpleConfig:
                class smiles:
                    encoding_method = "combined"
                    morgan_n_bits = 1024
                    morgan_radius = 2
                    rdkit_n_descriptors = 300
            config = SimpleConfig()
            drug_to_smiles, n_smiles_features = integrate_smiles_for_inference(
                adata_all, drug_metadata, config
            )
            logger.info(f"SMILES特征集成完成: {n_smiles_features} 维度")
            del drug_metadata

            base_model_path = "./dose_global_result/models/chemcpa_pretrain_model.pth"
            best_model_path = "./dose_global_result/models/chemcpa_pretrain_model_best.pth"
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
            logger.info(f"🎯 发现最佳模型，优先加载: {best_model_path}")
        elif os.path.exists(base_model_path):
            model_path = base_model_path
            logger.info(f"使用主模型: {base_model_path}")
        else:
            logger.error(f"模型文件不存在:")
            logger.error(f"  - 最佳模型: {best_model_path}")
            logger.error(f"  - 主模型: {base_model_path}")
            logger.error("请确认模型路径正确，或先运行训练脚本")
            return
        
        logger.info("步骤4: 加载训练好的模型并应用一致性映射...")
        chemcpa_model = load_trained_chemcpa_model(
            model_path, adata_all, global_control, prefer_best=False
        )
        logger.info("✅ 模型加载完成，映射一致性已确保")
        
        logger.info("步骤5: 执行按cell_line分批的单细胞counterfactual分析...")
        cell_line_results, all_condition_metrics, all_inference_metadata = perform_single_cell_counterfactual_analysis_by_cellline(
            chemcpa_model, adata_all, global_control
        )

        if not cell_line_results:
            logger.error("按cell_line的分析失败，无结果返回")
            return

        logger.info("步骤6: 保存全局汇总结果...")

        if all_condition_metrics:
            metrics_df = pd.DataFrame(all_condition_metrics)
            metrics_output_path = f"{RESULT_ROOT}/global_condition_metrics.csv"
            metrics_df.to_csv(metrics_output_path, index=False)
            logger.info(f"全局条件metrics已保存: {metrics_output_path}")

        if cell_line_results:
            cellline_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                                       for result in cell_line_results])
            cellline_output_path = f"{RESULT_ROOT}/global_cellline_metrics.csv"
            cellline_df.to_csv(cellline_output_path, index=False)
            logger.info(f"全局cell_line汇总已保存: {cellline_output_path}")

        metadata_summary = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "single_cell_counterfactual_by_cellline",
            "description": "按cell_line分批处理的单细胞counterfactual分析，优化内存使用",
            "data_format": "float32_optimized",
            "total_cells": len(all_inference_metadata),
            "total_inference_cells": len(all_inference_metadata),
            "total_genes": next(iter(all_inference_metadata), {}).get('n_total_genes', 0) if all_inference_metadata else 0,
            "total_conditions": len(all_condition_metrics),
            "total_cell_lines": len(cell_line_results),
            "unique_conditions": len(set(m['condition'] for m in all_inference_metadata)),
            "unique_drugs": len(set(m['drug'] for m in all_inference_metadata)),
            "unique_cell_lines": len(set(m['cell_line'] for m in all_inference_metadata)),
            "memory_optimization": "cell_line_wise_processing",
            "file_structure": {
                "cellline_results_dir": f"{RESULT_ROOT}/cellline_results/",
                "data_dir": str(Path(RESULT_ROOT) / "data"),
                "metadata_dir": f"{RESULT_ROOT}/metadata/",
                "results_dir": f"{RESULT_ROOT}/results/",
                "plots_dir": f"{RESULT_ROOT}/plots/",
                "individual_files": "每个cell_line单独保存inference结果和映射关系",
                "global_metrics": f"{RESULT_ROOT}/global_*_metrics.csv"
            }
        }

        summary_path = f"{RESULT_ROOT}/metadata_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metadata_summary, f, indent=2, default=json_default)
        logger.info(f"元数据摘要已保存: {summary_path}")

        del chemcpa_model, all_inference_metadata
        gc.collect()

        stats = analyze_distribution_similarity_results(cell_line_results, all_condition_metrics)
        
        summary = save_counterfactual_results(cell_line_results, all_condition_metrics, stats)
        
        create_distribution_visualization_plots(cell_line_results, all_condition_metrics)
        
        del cell_line_results, all_condition_metrics, stats
        gc.collect()
        
        logger.info("="*60)
        logger.info("ChemCPA 分布相似性分析完成")
        logger.info("="*60)
        logger.info("关键结果摘要:")
        logger.info(f"  总inference细胞数: {metadata_summary['total_cells']}")
        logger.info(f"  总基因数: {metadata_summary['total_genes']}")
        logger.info(f"  独特条件数: {metadata_summary['unique_conditions']}")
        logger.info(f"  独特药物数: {metadata_summary['unique_drugs']}")
        logger.info(f"  独特细胞系数: {metadata_summary['unique_cell_lines']}")

        if 'performance_statistics' in summary:
            perf_stats = summary['performance_statistics']
            if 'cellline_avg_MMD_RBF' in perf_stats:
                mmd_stats = perf_stats['cellline_avg_MMD_RBF']
                logger.info(f"  细胞系平均MMD RBF: {mmd_stats['mean']:.4f} ± {mmd_stats['std']:.4f}")
                logger.info(f"  MMD范围: [{mmd_stats['min']:.4f}, {mmd_stats['max']:.4f}]")

            if 'cellline_avg_Wasserstein_Sliced' in perf_stats:
                sw_stats = perf_stats['cellline_avg_Wasserstein_Sliced']
                logger.info(f"  细胞系平均Sliced Wasserstein: {sw_stats['mean']:.4f} ± {sw_stats['std']:.4f}")

        logger.info(f"  成功分析: {summary['successful_cell_lines']} 细胞系")
        logger.info(f"  总条件: {summary['total_conditions']}")
        
        if drug_to_smiles:
            total_drugs = len(drug_to_smiles)
            valid_smiles = sum(1 for smiles in drug_to_smiles.values() if smiles is not None)
            logger.info(f"  SMILES覆盖率: {valid_smiles}/{total_drugs} ({valid_smiles/total_drugs*100:.1f}%)")
        
        logger.info("\n关键特性:")
        logger.info("  🔌 插件系统：modular distribution metrics computation")
        logger.info("  📊 分布度量：MMD RBF, E-distance, Sliced Wasserstein, OT Wasserstein")
        logger.info("  🧬 单细胞级别：真实细胞分布 vs 预测细胞分布比较")
        logger.info("  🔄 无传统metrics：移除R², Pearson, MSE等传统度量")
        logger.info("  📈 分布比较：直接比较高维单细胞表达分布的相似性")
        logger.info("  🔢 全float32优化：内存高效的数据处理")
        logger.info("  🎯 最佳模型加载：优先使用验证loss最低的模型")
        logger.info("  📍 条件级分析：每个药物-剂量-细胞系组合独立评估")
        
        logger.info("\n输出文件:")
        logger.info(f"  📊 分布度量结果: {RESULT_ROOT}/results/condition_distribution_metrics.csv")
        logger.info(f"  📋 细胞系汇总: {RESULT_ROOT}/results/cellline_distribution_metrics.csv")
        logger.info(f"  📖 完整摘要: {RESULT_ROOT}/chemcpa_distribution_similarity_summary.json")
        logger.info(f"  📈 分布可视化: {RESULT_ROOT}/plots/ (MMD, Wasserstein分布图)")
        logger.info(f"  🗂️  分组结果: {RESULT_ROOT}/cellline_results/ (按细胞系分组)")
        logger.info(f"  📋 插件元数据: {RESULT_ROOT}/metadata/")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
