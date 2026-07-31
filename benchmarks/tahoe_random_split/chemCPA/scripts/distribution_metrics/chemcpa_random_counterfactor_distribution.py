#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from chemcpa_implementation import SMILESEncoder
from chemcpa_implementation import ChemCPAWithSMILES
from chemcpa_training import LossHistory
import gc
import chemcpa_utils
import sys
import warnings
from io import StringIO
import glob
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Force float32 and optimize performance
os.environ["OMP_NUM_THREADS"] = "90"
os.environ["OPENBLAS_NUM_THREADS"] = "90"
os.environ["MKL_NUM_THREADS"] = "90"
os.environ["NUMEXPR_NUM_THREADS"] = "90"

torch.set_num_threads(90)

def json_default(obj):
    """Handle numpy types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def setup_logging():
    """Setup logging system"""
    log_dir = "./distribution_similarity_result/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/chemcpa_counterfactual_{timestamp}.log"
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

def detect_data_source():
    """检测数据源：预计算结果 vs 原始数据"""
    precomputed_dir = "./counterfactual_result/cellline_results/"

    if os.path.exists(precomputed_dir) and len(glob.glob(f"{precomputed_dir}/*_inference_results.h5ad")) > 0:
        logger.info("🔗 检测到预计算的分cell line推理结果，将使用串联模式")
        return "precomputed_cellline"
    else:
        logger.info("📊 未检测到预计算结果，将使用原始数据")
        return "original"

def load_precomputed_cellline_data_efficiently():
    """内存高效的分cell line数据处理 - 逐个处理而不是全量合并"""
    logger.info("🔗 开始内存高效的分cell line推理数据处理...")
    test_data_path = resolve_benchmark_data_file("test_adata_processed.h5ad")

    try:
        cellline_results_dir = "./counterfactual_result/cellline_results/"

        # 获取所有cell line文件
        h5ad_files = glob.glob(f"{cellline_results_dir}/*_inference_results.h5ad")

        if not h5ad_files:
            logger.warning("未找到cell line推理结果文件")
            return None

        logger.info(f"发现 {len(h5ad_files)} 个cell line推理结果文件")

        # 只加载测试数据一次
        logger.info("加载原始测试数据...")
        adata_test = sc.read_h5ad(test_data_path)

        # 清理测试数据
        for col in ["cell_line", "drug"]:
            adata_test.obs[col] = (
                adata_test.obs[col]
                .astype(str)
                .str.strip()
                .str.replace("_", "-", regex=False)
            )
        adata_test.obs["dose"] = adata_test.obs["dose"].astype(np.float32)

        # 确保测试数据有dose_str列
        if 'dose_str' not in adata_test.obs.columns:
            logger.info("为测试数据创建dose_str列...")
            adata_test.obs['dose_str'] = adata_test.obs['dose'].astype(str).str.replace(".", "-", regex=False)

        if hasattr(adata_test.X, 'toarray'):
            adata_test.X = adata_test.X.astype(np.float32)
        else:
            adata_test.X = adata_test.X.astype(np.float32)

        logger.info(f"测试数据加载完成: {adata_test.n_obs} 细胞, 列: {list(adata_test.obs.columns)}")

        # 返回文件列表和测试数据，让后续函数逐个处理
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

    # 对于需要兼容原接口的调用，返回特殊标记
    return "EFFICIENT_MODE", result["adata_test"], result

def process_cellline_distribution_metrics_efficiently(efficient_data):
    """内存高效的分cell line分布度量计算"""
    logger.info("🚀 开始内存高效的分cell line分布度量分析...")

    h5ad_files = efficient_data["h5ad_files"]
    adata_test = efficient_data["adata_test"]
    cellline_results_dir = efficient_data["cellline_results_dir"]

    # 全局收集变量
    all_condition_metrics = []
    all_cellline_metrics = []
    total_inference_cells = 0

    # 逐个处理每个cell line
    for file_idx, h5ad_file in enumerate(h5ad_files, 1):
        cell_line = os.path.basename(h5ad_file).replace("_inference_results.h5ad", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"处理Cell Line [{file_idx}/{len(h5ad_files)}]: {cell_line}")
        logger.info(f"{'='*60}")

        try:
            # 1. 加载单个cell line的推理数据
            logger.info(f"加载 {cell_line} 推理数据...")
            cellline_adata = sc.read_h5ad(h5ad_file)
            logger.info(f"  推理细胞数: {cellline_adata.n_obs}")

            # 确保数据为float32
            if hasattr(cellline_adata.X, 'toarray'):
                cellline_adata.X = cellline_adata.X.astype(np.float32)
            else:
                cellline_adata.X = cellline_adata.X.astype(np.float32)

            # 数据清理
            for col in ["cell_line", "drug"]:
                cellline_adata.obs[col] = (
                    cellline_adata.obs[col]
                    .astype(str)
                    .str.strip()
                    .str.replace("_", "-", regex=False)
                )
            cellline_adata.obs["dose"] = cellline_adata.obs["dose"].astype(np.float32)

            # 2. 检查是否有预测数据
            if "ChemCPA_pred" not in cellline_adata.obsm:
                logger.warning(f"  {cell_line} 缺少预测数据，跳过")
                continue

            predictions = cellline_adata.obsm["ChemCPA_pred"]
            if not isinstance(predictions, np.ndarray):
                predictions = predictions.toarray()
            predictions = predictions.astype(np.float32)

            logger.info(f"  预测数据形状: {predictions.shape}")

            # 3. 检查并获取该cell line的唯一条件
            logger.info(f"  可用列: {list(cellline_adata.obs.columns)}")

            # 检查dose_str列是否存在，如果不存在则创建
            if 'dose_str' not in cellline_adata.obs.columns:
                logger.info("  创建dose_str列...")
                cellline_adata.obs['dose_str'] = cellline_adata.obs['dose'].astype(str).str.replace(".", "-", regex=False)

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
                        actual_expr = actual_expr.astype(np.float32)

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
                    # 获取该条件的预测结果
                    inf_mask = cellline_adata.obs['condition'] == condition_name
                    condition_pred = predictions[inf_mask]

                    if condition_pred.shape[0] == 0:
                        logger.warning(f"    条件 {condition_name} 无预测数据")
                        continue

                    # 获取对应的真实细胞
                    test_mask = (
                        (adata_test.obs['cell_line'] == cl) &
                        (adata_test.obs['drug'] == drug) &
                        (adata_test.obs['dose_str'] == dose_str)
                    )
                    actual_cells = adata_test[test_mask]

                    if actual_cells.n_obs == 0:
                        logger.warning(f"    条件 {condition_name} 无真实数据")
                        continue

                    actual_expr = actual_cells.X.toarray() if hasattr(actual_cells.X, 'toarray') else actual_cells.X
                    actual_expr = actual_expr.astype(np.float32)

                    # 6. 使用分布相似性度量插件计算（使用预计算的sigma）
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

                    # 记录主要度量
                    if condition_result["status"] == "success":
                        logger.info(f"    {condition_name}: MMD={condition_result['MMD_RBF']:.4f}, "
                                   f"E-dist={condition_result['E_distance']:.4f}, "
                                   f"SW={condition_result['Wasserstein_Sliced']:.4f}")

                except Exception as e:
                    logger.warning(f"    计算条件 {condition_name} 失败: {e}")
                    continue

            # 6. 汇总该cell line的结果
            if cellline_condition_metrics:
                successful_metrics = [c for c in cellline_condition_metrics if c.get("status") == "success"]

                if successful_metrics:
                    # 计算平均分布度量
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

            # 7. 汇总到全局结果
            all_condition_metrics.extend(cellline_condition_metrics)
            total_inference_cells += cellline_adata.n_obs

            # 8. 立即释放内存！
            del cellline_adata, predictions
            if 'actual_expr' in locals():
                del actual_expr
            if 'condition_pred' in locals():
                del condition_pred
            gc.collect()

            logger.info(f"  🧹 {cell_line} 内存已清理")

        except Exception as e:
            logger.error(f"处理 {cell_line} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            continue

    # 释放测试数据
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

    # 细胞系级别统计
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

    # 条件级别统计
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

    # 找出最佳和最差表现（基于MMD距离，越小越好）
    if 'avg_MMD_RBF' in df_cellline.columns:
        best_mmd_idx = df_cellline['avg_MMD_RBF'].idxmin()
        worst_mmd_idx = df_cellline['avg_MMD_RBF'].idxmax()

        best_cellline = df_cellline.loc[best_mmd_idx]
        worst_cellline = df_cellline.loc[worst_mmd_idx]

        logger.info(f"\n最佳分布匹配: {best_cellline['cell_line']} (MMD={best_cellline['avg_MMD_RBF']:.4f})")
        logger.info(f"最差分布匹配: {worst_cellline['cell_line']} (MMD={worst_cellline['avg_MMD_RBF']:.4f})")

    # 性能分布分析（基于MMD）
    if 'avg_MMD_RBF' in df_cellline.columns:
        mmd_values = df_cellline['avg_MMD_RBF'].dropna()
        excellent = (mmd_values <= 0.1).sum()
        good = ((mmd_values > 0.1) & (mmd_values <= 0.3)).sum()
        poor = (mmd_values > 0.3).sum()

        logger.info("\n分布相似性表现分布:")
        logger.info(f"  优秀 (MMD ≤ 0.1): {excellent} 个细胞系 ({excellent/len(mmd_values)*100:.1f}%)")
        logger.info(f"  良好 (0.1 < MMD ≤ 0.3): {good} 个细胞系 ({good/len(mmd_values)*100:.1f}%)")
        logger.info(f"  较差 (MMD > 0.3): {poor} 个细胞系 ({poor/len(mmd_values)*100:.1f}%)")

    # 合并统计
    all_stats = {**cellline_stats, **condition_stats}

    return all_stats

# Removed traditional correlation metrics - using distribution similarity metrics only

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


def load_original_data():
    """
    ⚠️ 废弃函数 - Distribution analysis不使用此函数
    Distribution analysis只需要预计算的推理结果和真实的test数据
    不需要control数据和模型推理
    """
    logger.error("❌ load_original_data() 不应在 distribution analysis 中被调用")
    logger.error("Distribution analysis只使用预计算结果，不需要control数据")
    raise NotImplementedError("此函数已废弃，distribution analysis不需要control数据")

def load_inference_data():
    """智能数据加载：自动检测并选择最佳数据源"""
    logger.info("🔍 检测数据源...")

    data_source = detect_data_source()

    if data_source == "precomputed_cellline":
        # 使用分cell line预计算结果
        inference_adata, adata_test, mapping_df = load_precomputed_cellline_data()
        if inference_adata is not None:
            logger.info("🚀 成功加载预计算的分cell line推理结果，跳过模型推理步骤")
            return inference_adata, adata_test, mapping_df
        else:
            logger.warning("预计算分cell line数据加载失败，回退到原始模式")

    # 回退到原始数据加载模式
    logger.error("❌ 未找到预计算推理结果！")
    logger.error("请先运行 chemcpa_random_counterfactor_full_v2.py 生成预计算结果")
    raise FileNotFoundError("Distribution analysis需要预计算的推理结果")

def analyze_distribution_similarity_results(cell_line_results, all_condition_metrics):
    """分析分布相似性度量结果"""
    logger.info("分析分布相似性度量结果...")

    successful_results = [r for r in cell_line_results if r.get("status") == "completed"]

    if not successful_results:
        logger.error("没有成功的结果可供分析")
        return {}

    logger.info(f"成功分析了 {len(successful_results)} 个细胞系")
    logger.info(f"总条件分析: {len(all_condition_metrics)}")

    # 细胞系级别分布度量统计
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

    # 条件级别分布度量统计
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

    # 找出最佳和最差表现（基于MMD RBF，越小越好）
    if 'avg_MMD_RBF' in df_cellline.columns:
        best_mmd_idx = df_cellline['avg_MMD_RBF'].idxmin()  # 最小MMD最好
        worst_mmd_idx = df_cellline['avg_MMD_RBF'].idxmax()  # 最大MMD最差

        best_cellline = df_cellline.loc[best_mmd_idx]
        worst_cellline = df_cellline.loc[worst_mmd_idx]

        logger.info(f"\n最佳分布匹配: {best_cellline['cell_line']} (MMD={best_cellline['avg_MMD_RBF']:.4f}, {best_cellline['n_conditions']} 条件)")
        logger.info(f"最差分布匹配: {worst_cellline['cell_line']} (MMD={worst_cellline['avg_MMD_RBF']:.4f}, {worst_cellline['n_conditions']} 条件)")

    # 分布相似性分析（基于MMD RBF阈值）
    if 'avg_MMD_RBF' in df_cellline.columns:
        mmd_values = df_cellline['avg_MMD_RBF'].dropna()
        if len(mmd_values) > 0:
            # 定义阈值（越小越好）
            excellent_performance = (mmd_values <= 0.1).sum()
            good_performance = ((mmd_values > 0.1) & (mmd_values <= 0.3)).sum()
            poor_performance = (mmd_values > 0.3).sum()

            logger.info("\n分布相似性分析 (基于MMD RBF):")
            logger.info(f"  优秀匹配 (MMD ≤ 0.1): {excellent_performance} 个细胞系 ({excellent_performance/len(mmd_values)*100:.1f}%)")
            logger.info(f"  良好匹配 (0.1 < MMD ≤ 0.3): {good_performance} 个细胞系 ({good_performance/len(mmd_values)*100:.1f}%)")
            logger.info(f"  较差匹配 (MMD > 0.3): {poor_performance} 个细胞系 ({poor_performance/len(mmd_values)*100:.1f}%)")

    # 合并统计
    all_stats = {**cellline_stats, **condition_stats}

    return all_stats

# ❌ 已废弃：此函数会创建巨大的全局h5ad文件，已被内存高效模式替代
# def save_single_cell_inference_results(inference_adata_with_pred, inference_to_actual_mapping):
#     """保存单细胞inference结果的adata和映射信息 - 已废弃，使用cell_line分批保存"""
#     pass

def save_counterfactual_results(cell_line_results, all_condition_metrics, stats):
    """保存counterfactual分析结果"""
    logger.info("保存counterfactual分析结果...")
    # 确保输出目录存在
    os.makedirs("./distribution_similarity_result/results", exist_ok=True)

    # 保存详细结果
    cellline_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                               for result in cell_line_results])
    cellline_df.to_csv("./distribution_similarity_result/results/cellline_distribution_metrics.csv", index=False)

    condition_df = pd.DataFrame(all_condition_metrics)
    condition_df.to_csv("./distribution_similarity_result/results/condition_distribution_metrics.csv", index=False)
    # 创建最终摘要
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
    # 保存完整摘要
    with open("./distribution_similarity_result/chemcpa_distribution_similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)
    logger.info("结果保存:")
    logger.info("  细胞系结果: ./distribution_similarity_result/results/cellline_distribution_metrics.csv")
    logger.info("  条件结果: ./distribution_similarity_result/results/condition_distribution_metrics.csv")
    logger.info("  完整摘要: ./distribution_similarity_result/chemcpa_distribution_similarity_summary.json")
    return summary

def create_distribution_visualization_plots(cell_line_results, all_condition_metrics, output_dir="./distribution_similarity_result"):
    """创建分布相似性度量可视化图表"""
    logger.info("创建分布相似性度量可视化图表...")

    try:
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        # 细胞系分布相似性热图
        if cell_line_results:
            cl_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                                 for result in cell_line_results])

            if not cl_df.empty:
                dist_metrics_to_plot = ['avg_MMD_RBF', 'avg_E_distance', 'avg_Wasserstein_Sliced', 'avg_Wasserstein_OT']
                available_metrics = [m for m in dist_metrics_to_plot if m in cl_df.columns]

                if available_metrics:
                    plt.figure(figsize=(15, 10))

                    # 标准化数据用于热图显示
                    heatmap_data = cl_df.set_index('cell_line')[available_metrics]
                    sns.heatmap(heatmap_data, annot=True, cmap='viridis_r', fmt='.4f', cbar_kws={'label': 'Distance (smaller=better)'})
                    plt.title('ChemCPA Distribution Similarity: Cell Line Performance Heatmap')
                    plt.ylabel('Cell Lines')
                    plt.xlabel('Distribution Similarity Metrics')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/cellline_distribution_heatmap.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("细胞系分布相似性热图已保存")

        # MMD分布直方图
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

        # 分布度量比较图
        if all_condition_metrics:
            successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

            if successful_conditions:
                condition_df = pd.DataFrame(successful_conditions)

                # 创建多子图比较不同分布度量
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

        # 药物效果分析图（基于MMD）
        if all_condition_metrics:
            successful_conditions = [c for c in all_condition_metrics if c.get("status") == "success"]

            if successful_conditions:
                condition_df = pd.DataFrame(successful_conditions)

                if 'drug' in condition_df.columns and 'MMD_RBF' in condition_df.columns:
                    plt.figure(figsize=(15, 8))

                    # 按药物分组计算平均MMD
                    drug_performance = condition_df.groupby('drug')['MMD_RBF'].agg(['mean', 'std', 'count']).reset_index()
                    drug_performance = drug_performance.sort_values('mean', ascending=True)  # 升序，因为越小越好

                    # 只显示前20个药物
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
        # 1. 智能加载推理数据
        adata_all, precomputed_test_data, precomputed_mapping = load_inference_data()

        # 检查是否使用预计算结果
        use_precomputed = (precomputed_test_data is not None and precomputed_mapping is not None)

        if use_precomputed:
            logger.info("🚀 检测到预计算结果，使用内存高效的分cell line处理模式")

            # 检查是否是高效模式
            if adata_all == "EFFICIENT_MODE":
                logger.info("✅ 使用内存高效的逐cell line处理模式")
                efficient_data = precomputed_mapping  # 包含所有需要的数据

                # 5. 执行内存高效的分布度量计算
                cell_line_results, all_condition_metrics = process_cellline_distribution_metrics_efficiently(efficient_data)

                if not cell_line_results:
                    logger.error("内存高效处理失败，无结果返回")
                    return

                logger.info(f"✅ 内存高效处理完成: {len(cell_line_results)} cell lines, {len(all_condition_metrics)} conditions")

                # 跳转到结果处理阶段
                logger.info("🎯 跳转到结果保存和分析阶段...")

                # 6. 保存全局汇总结果
                logger.info("步骤6: 保存全局汇总结果...")

                # 保存全局metrics
                if all_condition_metrics:
                    metrics_df = pd.DataFrame(all_condition_metrics)
                    metrics_output_path = f"./distribution_similarity_result/global_condition_metrics.csv"
                    metrics_df.to_csv(metrics_output_path, index=False)
                    logger.info(f"全局条件metrics已保存: {metrics_output_path}")

                # 保存cell_line汇总
                if cell_line_results:
                    cellline_df = pd.DataFrame(cell_line_results)
                    cellline_output_path = f"./distribution_similarity_result/global_cellline_metrics.csv"
                    cellline_df.to_csv(cellline_output_path, index=False)
                    logger.info(f"全局cell_line汇总已保存: {cellline_output_path}")

                # 8. 分析结果
                stats = analyze_distribution_similarity_results(cell_line_results, all_condition_metrics)

                # 9. 保存结果摘要
                summary = save_counterfactual_results(cell_line_results, all_condition_metrics, stats)

                # 10. 创建可视化
                create_distribution_visualization_plots(cell_line_results, all_condition_metrics)

                # 释放结果变量
                del cell_line_results, all_condition_metrics, stats
                gc.collect()

                # 11. 最终报告
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
                logger.info("  📊 分布度量结果: ./distribution_similarity_result/results/condition_distribution_metrics.csv")
                logger.info("  📋 细胞系汇总: ./distribution_similarity_result/results/cellline_distribution_metrics.csv")
                logger.info("  📖 完整摘要: ./distribution_similarity_result/chemcpa_distribution_similarity_summary.json")
                logger.info("  📈 分布可视化: ./distribution_similarity_result/plots/")
                logger.info("="*60)

                return  # 高效模式完成，直接返回

            else:
                logger.warning("⚠️ 使用旧版预计算逻辑，可能导致内存问题")
                # 旧版逻辑保持不变但加上警告
                inference_adata_with_pred = adata_all
                test_data = precomputed_test_data
                mapping_df = precomputed_mapping

                if "ChemCPA_pred" not in inference_adata_with_pred.obsm:
                    logger.error("预计算结果中缺少 ChemCPA_pred 数据")
                    raise ValueError("预计算数据格式错误")

                # 使用旧版逻辑进行分析（保持兼容）
                logger.error("❌ 旧版预计算逻辑已禁用，请使用内存高效模式")
                return

        else:
            logger.error("❌ Distribution analysis必须使用预计算结果")
            logger.error("请先运行 chemcpa_random_counterfactor_full_v2.py 生成推理结果")
            raise RuntimeError("Distribution analysis不支持原始数据模式（不需要control和模型）")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
