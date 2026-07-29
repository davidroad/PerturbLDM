#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChemCPA Utilities Module
Helper functions and utilities for ChemCPA model training and evaluation
"""

import numpy as np
import pandas as pd
import scanpy as sc
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def setup_directories():
    """设置必要的目录结构"""
    directories = [
        "./dose_global_result",
        "./dose_global_result/logs", 
        "./dose_global_result/models",
        "./dose_global_result/results",
        "./dose_global_result/analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories


def setup_directories():
    """设置必要的目录结构"""
    directories = [
        "./dose_global_result",
        "./dose_global_result/logs", 
        "./dose_global_result/models",
        "./dose_global_result/results",
        "./dose_global_result/analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories


def validate_data_format(adata: sc.AnnData, required_obs_cols: List[str] = None) -> bool:
    """验证数据格式是否符合要求"""
    if required_obs_cols is None:
        required_obs_cols = ["cell_line", "drug", "dose"]
    
    missing_cols = []
    for col in required_obs_cols:
        if col not in adata.obs.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"缺少必要的obs列: {missing_cols}")
    
    # 检查数据完整性
    if adata.n_obs == 0:
        raise ValueError("数据集为空")
    
    if adata.n_vars == 0:
        raise ValueError("没有基因特征")
    
    return True


def clean_drug_names(adata: sc.AnnData, columns: List[str] = None) -> sc.AnnData:
    """统一清理药物和细胞系名称格式"""
    if columns is None:
        columns = ["cell_line", "drug"]
    
    for col in columns:
        if col in adata.obs.columns:
            adata.obs[col] = (
                adata.obs[col]
                .astype(str)
                .str.strip()
                .str.replace("_", "-", regex=False)
            )
    
    # 创建剂量字符串格式
    if "dose" in adata.obs.columns:
        adata.obs["dose_str"] = (
            adata.obs["dose"]
            .astype(str)
            .str.replace(".", "-", regex=False)
        )
    
    # 创建组合标识符
    if all(col in adata.obs.columns for col in ["cell_line", "drug"]):
        if "dose_str" in adata.obs.columns:
            adata.obs["cov_drug_dose"] = (
                adata.obs["cell_line"]
                + "_"
                + adata.obs["drug"]
                + "_"
                + adata.obs["dose_str"]
            )
        else:
            adata.obs["cov_drug_dose"] = (
                adata.obs["cell_line"]
                + "_"
                + adata.obs["drug"]
            )
    
    return adata


def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    import gc
    
    # 触发垃圾回收
    gc.collect()
    
    # 获取系统内存信息
    memory = psutil.virtual_memory()
    
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent
    }


def log_memory_usage(logger: logging.Logger, stage: str = ""):
    """记录内存使用情况"""
    mem_info = get_memory_usage()
    logger.info(f"💾 内存使用 {stage}: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']:.1f}%)")


def chatterjee_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算 Chatterjee 相关系数
    一种非参数的相关性度量，对异常值更加鲁棒
    """
    n = len(x)
    if n < 2:
        return 0.0
    
    # 按x排序
    order = np.argsort(x)
    y_ordered = y[order]
    
    # 计算y的排名
    ranks = rankdata(y_ordered, method='ordinal')
    
    # 计算相邻排名的差异
    diff = np.abs(np.diff(ranks))
    num = diff.sum()
    
    # Chatterjee相关系数
    return 1 - (3 * num) / (n**2 - 1)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算全面的评估指标"""
    # 确保输入是一维数组
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    # 移除无效值
    valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    if len(y_true_valid) == 0:
        return {
            "n_valid": 0,
            "MSE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Pearson_r": np.nan,
            "Spearman_r": np.nan,
            "Chatterjee": np.nan
        }
    
    # 计算基本指标
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_valid, y_pred_valid)
    
    # 计算相关性指标
    try:
        pearson_r, _ = pearsonr(y_true_valid, y_pred_valid)
    except:
        pearson_r = np.nan
    
    try:
        spearman_r, _ = spearmanr(y_true_valid, y_pred_valid)
    except:
        spearman_r = np.nan
    
    try:
        chatterjee = chatterjee_correlation(y_true_valid, y_pred_valid)
    except:
        chatterjee = np.nan
    
    return {
        "n_valid": len(y_true_valid),
        "MSE": round(mse, 6),
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "R2": round(r2, 6),
        "Pearson_r": round(pearson_r, 6),
        "Spearman_r": round(spearman_r, 6),
        "Chatterjee": round(chatterjee, 6)
    }


def find_common_controls(adata_ctrl: sc.AnnData, cell_lines: np.ndarray) -> str:
    """找到所有细胞系共同的对照组"""
    all_ctrl_labels = set()
    
    for cl in cell_lines:
        ctrl_cl = adata_ctrl[adata_ctrl.obs["cell_line"] == cl]
        if len(ctrl_cl) == 0:
            continue
            
        # 查找DMSO对照组
        mask = (
            ctrl_cl.obs["drug"].str.contains("DMSO", case=False, na=False) & 
            (ctrl_cl.obs["dose_str"] == "0-0")
        )
        
        if mask.any():
            ctrl_labels = ctrl_cl.obs.loc[mask, "cov_drug_dose"].unique()
            all_ctrl_labels.update(ctrl_labels)
    
    if not all_ctrl_labels:
        # 如果没找到DMSO，尝试查找其他对照组
        for cl in cell_lines:
            ctrl_cl = adata_ctrl[adata_ctrl.obs["cell_line"] == cl]
            if len(ctrl_cl) == 0:
                continue
            ctrl_labels = ctrl_cl.obs["cov_drug_dose"].unique()
            all_ctrl_labels.update(ctrl_labels)
    
    if len(all_ctrl_labels) == 0:
        raise ValueError("未找到任何对照组")
    
    # 选择最常见的对照组
    if len(all_ctrl_labels) > 1:
        ctrl_counts = {}
        for label in all_ctrl_labels:
            count = np.sum(adata_ctrl.obs["cov_drug_dose"] == label)
            ctrl_counts[label] = count
        
        global_control = max(ctrl_counts.items(), key=lambda x: x[1])[0]
    else:
        global_control = list(all_ctrl_labels)[0]
    
    return global_control


def create_data_summary(adata_list: List[sc.AnnData], names: List[str] = None) -> pd.DataFrame:
    """创建数据集摘要"""
    if names is None:
        names = [f"Dataset_{i+1}" for i in range(len(adata_list))]
    
    summary_data = []
    
    for adata, name in zip(adata_list, names):
        summary = {
            "dataset": name,
            "n_observations": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_cell_lines": len(adata.obs["cell_line"].unique()) if "cell_line" in adata.obs else 0,
            "n_drugs": len(adata.obs["drug"].unique()) if "drug" in adata.obs else 0,
            "n_conditions": len(adata.obs["cov_drug_dose"].unique()) if "cov_drug_dose" in adata.obs else 0
        }
        
        # 添加剂量信息
        if "dose" in adata.obs:
            summary["n_doses"] = len(adata.obs["dose"].unique())
            summary["dose_range"] = f"{adata.obs['dose'].min():.2f} - {adata.obs['dose'].max():.2f}"
        
        # 添加SMILES特征信息
        if "smiles_features" in adata.obsm:
            summary["smiles_features_dim"] = adata.obsm["smiles_features"].shape[1]
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def filter_low_quality_predictions(predictions: np.ndarray, threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """过滤低质量的预测结果"""
    # 计算预测的方差
    pred_var = np.var(predictions, axis=1)
    
    # 过滤方差过小的预测（可能是模型崩塌）
    valid_mask = pred_var > threshold
    
    return predictions[valid_mask], valid_mask


def compute_condition_metrics(true_values: np.ndarray, predictions: np.ndarray, 
                            conditions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """计算每个condition的平均指标"""
    unique_conditions = np.unique(conditions)
    condition_true_means = []
    condition_pred_means = []
    condition_details = []
    
    for condition in unique_conditions:
        cond_mask = conditions == condition
        cond_indices = np.where(cond_mask)[0]
        
        if len(cond_indices) == 0:
            continue
        
        cond_true = true_values[cond_indices]
        cond_pred = predictions[cond_indices]
        
        # 计算每个基因在该condition下的平均值
        cond_true_mean = np.mean(cond_true, axis=0)
        cond_pred_mean = np.mean(cond_pred, axis=0)
        
        condition_true_means.append(cond_true_mean)
        condition_pred_means.append(cond_pred_mean)
        
        # 解析condition信息
        parts = str(condition).split("_")
        condition_detail = {
            "condition": condition,
            "cell_line": parts[0] if len(parts) > 0 else "",
            "drug": parts[1] if len(parts) > 1 else "",
            "dose": parts[2] if len(parts) > 2 else "",
            "n_cells": int(np.sum(cond_mask))
        }
        condition_details.append(condition_detail)
    
    condition_true_means = np.array(condition_true_means)
    condition_pred_means = np.array(condition_pred_means)
    
    return condition_true_means, condition_pred_means, condition_details


def save_evaluation_results(results: List[Dict], output_dir: str, prefix: str = ""):
    """保存评估结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 分离成功和失败的结果
    successful_results = [r for r in results if r.get("status") == "completed"]
    failed_results = [r for r in results if r.get("status") != "completed"]
    
    # 保存成功的结果
    if successful_results:
        df_results = pd.DataFrame(successful_results)
        success_file = os.path.join(output_dir, f"{prefix}successful_results.csv")
        df_results.to_csv(success_file, index=False)
        
        # 保存condition详情
        all_condition_details = []
        for result in successful_results:
            if 'condition_details' in result:
                for cond_detail in result['condition_details']:
                    cond_detail['cell_line_result'] = result['cell_line']
                    if 'R2' in result:
                        cond_detail['cell_line_r2'] = result['R2']
                    all_condition_details.append(cond_detail)
        
        if all_condition_details:
            condition_file = os.path.join(output_dir, f"{prefix}condition_details.csv")
            condition_df = pd.DataFrame(all_condition_details)
            condition_df.to_csv(condition_file, index=False)
    
    # 保存失败的结果
    if failed_results:
        df_failed = pd.DataFrame(failed_results)
        failed_file = os.path.join(output_dir, f"{prefix}failed_results.csv")
        df_failed.to_csv(failed_file, index=False)
    
    return len(successful_results), len(failed_results)


def generate_performance_report(results: List[Dict], metrics_cols: List[str] = None) -> Dict[str, Any]:
    """生成性能报告"""
    if metrics_cols is None:
        metrics_cols = ['MSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_r', 'Chatterjee']
    
    successful_results = [r for r in results if r.get("status") == "completed"]
    
    if not successful_results:
        return {"status": "no_successful_results"}
    
    df_results = pd.DataFrame(successful_results)
    
    # 计算统计指标
    stats = {}
    for col in metrics_cols:
        if col in df_results.columns:
            stats[col] = {
                'mean': float(df_results[col].mean()),
                'std': float(df_results[col].std()),
                'median': float(df_results[col].median()),
                'min': float(df_results[col].min()),
                'max': float(df_results[col].max()),
                'q25': float(df_results[col].quantile(0.25)),
                'q75': float(df_results[col].quantile(0.75))
            }
    
    # 找出最好和最差的表现
    best_worst = {}
    if 'R2' in df_results.columns:
        best_idx = df_results['R2'].idxmax()
        worst_idx = df_results['R2'].idxmin()
        
        best_worst = {
            'best_performer': {
                'cell_line': df_results.loc[best_idx, 'cell_line'],
                'R2': float(df_results.loc[best_idx, 'R2']),
                'n_conditions': int(df_results.loc[best_idx, 'n_conditions']) if 'n_conditions' in df_results.columns else 0
            },
            'worst_performer': {
                'cell_line': df_results.loc[worst_idx, 'cell_line'],
                'R2': float(df_results.loc[worst_idx, 'R2']),
                'n_conditions': int(df_results.loc[worst_idx, 'n_conditions']) if 'n_conditions' in df_results.columns else 0
            }
        }
    
    # 分析数据量与性能的关系
    correlation_analysis = {}
    if 'train_samples' in df_results.columns and 'R2' in df_results.columns:
        try:
            corr_coef, _ = pearsonr(df_results['train_samples'], df_results['R2'])
            correlation_analysis['train_samples_vs_r2'] = float(corr_coef)
        except:
            pass
    
    if 'n_conditions' in df_results.columns and 'R2' in df_results.columns:
        try:
            corr_coef, _ = pearsonr(df_results['n_conditions'], df_results['R2'])
            correlation_analysis['n_conditions_vs_r2'] = float(corr_coef)
        except:
            pass
    
    return {
        "status": "success",
        "n_successful": len(successful_results),
        "n_failed": len(results) - len(successful_results),
        "performance_statistics": stats,
        "best_worst_performers": best_worst,
        "correlation_analysis": correlation_analysis,
        "summary_stats": {
            "total_cell_lines": len(results),
            "success_rate": len(successful_results) / len(results) if results else 0
        }
    }


def validate_predictions(predictions: np.ndarray, true_values: np.ndarray, 
                        min_variance: float = 1e-8) -> Dict[str, Any]:
    """验证预测结果的质量"""
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    # 检查形状匹配
    if predictions.shape != true_values.shape:
        validation_results["errors"].append(
            f"预测和真实值形状不匹配: {predictions.shape} vs {true_values.shape}"
        )
        validation_results["is_valid"] = False
    
    # 检查是否包含无效值
    if np.any(~np.isfinite(predictions)):
        n_invalid = np.sum(~np.isfinite(predictions))
        validation_results["warnings"].append(
            f"预测中包含 {n_invalid} 个无效值 (NaN/Inf)"
        )
    
    # 检查预测方差
    pred_var = np.var(predictions)
    if pred_var < min_variance:
        validation_results["warnings"].append(
            f"预测方差过小 ({pred_var:.2e})，可能存在模型崩塌"
        )
    
    # 检查预测范围
    pred_range = np.max(predictions) - np.min(predictions)
    true_range = np.max(true_values) - np.min(true_values)
    
    if pred_range < true_range * 0.1:
        validation_results["warnings"].append(
            f"预测范围过小，可能模型未充分学习数据变化"
        )
    
    return validation_results

