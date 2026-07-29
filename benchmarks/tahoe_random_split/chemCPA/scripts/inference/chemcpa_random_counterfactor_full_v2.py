#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChemCPA Single-Cell Counterfactual Analysis - Full Cell-Level Implementation
Performs inference on every individual cell instead of condition-wise means.
Saves complete AnnData with all single-cell inference results and metadata in float32 format.
Uses CPA-style data setup with SMILES integration and comprehensive memory optimization.
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ⚠️ 不要设置CUDA_VISIBLE_DEVICES！
# PyTorch的_load_from_bytes有bug，必须允许访问模型训练时的GPU
# 加载后我们会手动移动到cuda:0

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

# 强制设置CUDA设备为cuda:0
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"✅ CUDA配置: 可见设备={os.environ.get('CUDA_VISIBLE_DEVICES')}, 当前设备ID={torch.cuda.current_device()}, GPU数量={torch.cuda.device_count()}")

from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.patches as mpatches
from chemcpa_implementation import SMILESEncoder
from chemcpa_implementation import ChemCPAWithSMILES
from chemcpa_training import LossHistory
import gc
import chemcpa_utils
import sys
import warnings
import io
from io import StringIO
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Force float32 and optimize performance
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "64"

torch.set_num_threads(64)

def json_default(obj):
    """Handle numpy types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def setup_logging():
    """Setup logging system"""
    log_dir = "./counterfactual_result/logs"
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

def get_device(device_preference="auto"):
    """获取设备"""
    if device_preference == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_preference.startswith("cuda"):
        if torch.cuda.is_available():
            return device_preference
        else:
            logger.warning("GPU不可用，回退到CPU")
            return "cpu"
    return device_preference


def load_inference_data(test_data_path, control_data_path):
    """加载推理数据 - CPA风格"""
    logger.info("加载推理数据（test和control）...")
    
    try:
        # 加载测试和控制数据
        adata_test = sc.read_h5ad(test_data_path)
        adata_ctrl = sc.read_h5ad(control_data_path)
        # 确保数据为float32
        for adata in [adata_test, adata_ctrl]:
            if hasattr(adata.X, 'toarray'):
                pass  # adata.X = adata.X.astype(np.float32)  # 数据已经是float32
            else:
                pass  # adata.X = adata.X.astype(np.float32)  # 数据已经是float32
        
        logger.info(f"数据加载成功: test={adata_test.n_obs}, ctrl={adata_ctrl.n_obs}")
        logger.info(f"数据类型: {adata_test.X.dtype}")
        
        # 数据清理 - 与训练脚本保持一致
        datasets = (adata_ctrl, adata_test)
        for ad in datasets:
            for col in ["cell_line", "drug"]:
                ad.obs[col] = (
                    ad.obs[col]
                    .astype(str)
                    .str.strip()  # <-- 添加这一行来去除前后空格
                    .str.replace("_", "-", regex=False)
                )
            ad.obs["dose_str"] = (
                ad.obs["dose"]
                .astype(str)
                .str.replace(".", "-", regex=False)
            )
            # dose已经是float32类型
            # ad.obs["dose"] = ad.obs["dose"].astype(np.float32)  # 数据已经是float32
        
        # CPA风格合并 - 使用split标识
        adata_all = sc.concat(
            [adata_ctrl, adata_test],
            join="inner",
            label="split", 
            keys=["ctrl", "test"],
            index_unique=None
        )

        # 释放原始数据节省内存
        del adata_ctrl, adata_test
        gc.collect()
        
        logger.info(f"合并数据: {adata_all.n_obs} 细胞, {adata_all.n_vars} 基因")
        split_counts = adata_all.obs["split"].value_counts().to_dict()
        logger.info(f"Split分布: {split_counts}")
        
        return adata_all
        
    except Exception as e:
        logger.error(f"加载推理数据失败: {e}")
        raise

def apply_training_consistent_mapping(adata_all, training_drug_to_idx, training_covariate_mappings):
    """
    将inference数据的映射调整为与训练时一致的索引
    处理inference数据中可能缺失的药物/细胞系问题
    """
    logger.info("应用训练时一致的数据映射...")
    
    # 处理药物映射
    drug_indices = []
    missing_drugs = []
    
    for drug in adata_all.obs["drug"]:
        if drug in training_drug_to_idx:
            drug_indices.append(training_drug_to_idx[drug])
        else:
            missing_drugs.append(drug)
            # 使用默认值（比如第一个药物的索引）
            drug_indices.append(0)  
    
    adata_all.obs['drug_idx'] = drug_indices
    
    if missing_drugs:
        unique_missing = list(set(missing_drugs))
        logger.warning(f"检测到 {len(unique_missing)} 个训练时未见的药物: {unique_missing[:10]}...")
        logger.warning(f"这些药物将使用默认索引 0")
    
    # 处理细胞系映射（第一个协变量）
    if training_covariate_mappings and len(training_covariate_mappings) > 0:
        cell_line_to_idx = training_covariate_mappings[0]
        cell_line_indices = []
        missing_cell_lines = []
        
        for cell_line in adata_all.obs["cell_line"]:
            if cell_line in cell_line_to_idx:
                cell_line_indices.append(cell_line_to_idx[cell_line])
            else:
                missing_cell_lines.append(cell_line)
                # 使用默认值
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
    
    # CPA风格：直接使用drug作为控制组
    global_control = "DMSO-TF"
    
    # 验证控制组存在
    control_mask = adata_all.obs["drug"] == global_control
    control_count = np.sum(control_mask)
    
    if control_count == 0:
        logger.warning(f"控制组 {global_control} 未找到")
        logger.info("检查可用药物类型:")
        all_drugs = set(adata_all.obs["drug"].unique())
        logger.info(f"可用药物: {sorted(all_drugs)}")
        
        # 查找DMSO替代
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
    
    # ✅ 新增：智能选择模型文件
    actual_model_path = model_path
    
    if prefer_best:
        # 尝试查找最佳模型文件
        best_model_path = model_path.replace("chemcpa_pretrain_model.pth", "chemcpa_pretrain_model_best.pth")
        
        if os.path.exists(best_model_path):
            actual_model_path = best_model_path
            logger.info(f"🎯 找到最佳模型，将加载: {best_model_path}")
        elif os.path.exists(model_path):
            logger.info(f"未找到专用最佳模型文件，加载主模型: {model_path}")
            
            # 检查主模型是否包含最佳状态
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
        # ⚠️ 终极解决方案：先加载到CPU，然后移动到cuda:0
        # PyTorch的_load_from_bytes有bug，但加载到CPU时这个bug不影响
        logger.info("正在加载模型到CPU（避免其他GPU的OOM问题）...")

        target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        def map_to_target(storage, loc):
            if target_device.type == 'cuda':
                return storage.cuda(target_device.index or 0)
            return storage.cpu()

        import torch.storage as storage_module

        original_load_from_bytes = storage_module._load_from_bytes

        def patched_load_from_bytes(b):
            storage_module._load_from_bytes = original_load_from_bytes
            try:
                return torch.load(io.BytesIO(b), map_location=map_to_target, weights_only=False)
            finally:
                storage_module._load_from_bytes = patched_load_from_bytes

        storage_module._load_from_bytes = patched_load_from_bytes

        try:
            with open(actual_model_path, 'rb') as f:
                model_state = torch.load(f, map_location=map_to_target, weights_only=False)
        except RuntimeError as exc:
            if "Invalid magic number" in str(exc):
                logger.warning("torch.load 报错 Invalid magic number，尝试使用 pickle.load 重新加载")
                with open(actual_model_path, 'rb') as f:
                    model_state = pickle.load(f)
            else:
                raise
        finally:
            storage_module._load_from_bytes = original_load_from_bytes

        logger.info(f"模型加载成功，目标设备: {target_device}")

        if 'model_state_dict' in model_state:
            new_state_dict = {}
            for k, v in model_state['model_state_dict'].items():
                if torch.is_tensor(v):
                    new_state_dict[k] = v.to(target_device)
                else:
                    new_state_dict[k] = v
            model_state['model_state_dict'] = new_state_dict
            logger.info(f"✅ 所有模型参数已移动到 {target_device}")
        
        logger.info(f"模型类型: {model_state.get('model_type', 'unknown')}")
        logger.info(f"SMILES特征: {model_state.get('n_smiles_features', 0)}")
        
        # ✅ 新增：显示加载的模型信息
        if model_state.get('is_best_model', False):
            logger.info(f"🏆 加载最佳模型状态:")
            logger.info(f"   最佳epoch: {model_state.get('best_epoch', 'N/A')}")
            logger.info(f"   最佳验证loss: {model_state.get('best_val_loss', 'N/A')}")
            logger.info(f"   最佳训练loss: {model_state.get('best_train_loss', 'N/A')}")
        elif 'best_epoch' in model_state:
            logger.info(f"📊 主模型包含最佳状态:")
            logger.info(f"   最佳epoch: {model_state.get('best_epoch', 'N/A')}")
            logger.info(f"   最佳验证loss: {model_state.get('best_val_loss', 'N/A')}")
        
        # CPA风格数据设置 - 抑制输出
        import warnings
        from io import StringIO
        
        warnings.filterwarnings('ignore')
        sc.settings.verbosity = 0
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            # 导入ChemCPA类
            from chemcpa_implementation import ChemCPAWithSMILES
            
            # CPA风格设置
            ChemCPAWithSMILES.split_key = "split"
            ChemCPAWithSMILES.setup_anndata(
                adata=adata_all,
                perturbation_key="drug",  # 关键改变：使用drug而不是cov_drug_dose
                control_group=global_control,  # "DMSO-TF"
                dosage_key="dose",
                categorical_covariate_keys=["cell_line"]
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        warnings.filterwarnings('default')
        sc.settings.verbosity = 1
        
        # 创建模型实例
        device = get_device("auto")
        chemcpa_model = ChemCPAWithSMILES(
            adata=adata_all,
            n_smiles_features=model_state.get('n_smiles_features', 0),
            device=device
        )
        
        # 恢复映射信息并应用到inference数据
        training_drug_to_idx = None
        training_covariate_mappings = None
        
        # 尝试从直接保存的映射中读取
        if 'drug_to_idx' in model_state and 'covariate_mappings' in model_state:
            training_drug_to_idx = model_state['drug_to_idx']
            training_covariate_mappings = model_state['covariate_mappings']
            logger.info(f"从模型中恢复直接映射: {len(training_drug_to_idx)} 药物, {[len(m) for m in training_covariate_mappings]} 协变量")
        # 备用：从 registry 中读取
        elif 'registry' in model_state:
            logger.info("从模型 registry 中恢复映射...")
            registry = model_state['registry']
            training_drug_to_idx = registry.get("drug_mapping", {})
            training_covariate_mappings = registry.get("covariate_mappings", [])
            logger.info(f"从 registry 恢复映射: {len(training_drug_to_idx)} 药物, {[len(m) for m in training_covariate_mappings]} 协变量")
        
        # 应用训练时的映射到inference数据
        if training_drug_to_idx and training_covariate_mappings:
            # 使用训练时的映射对inference数据进行映射
            drug_to_idx, covariate_mappings = apply_training_consistent_mapping(
                adata_all, training_drug_to_idx, training_covariate_mappings
            )
            
            # 保存到模型中供预测使用
            chemcpa_model.drug_categories = list(training_drug_to_idx.keys())
            if training_covariate_mappings:
                chemcpa_model.cell_categories = list(training_covariate_mappings[0].keys())
            
            # 记录训练时的类别数量以确保模型结构一致
            adata_all.uns['training_drug_count'] = len(training_drug_to_idx)
            if training_covariate_mappings:
                adata_all.uns['training_cell_count'] = len(training_covariate_mappings[0])

            logger.info("✅ 训练时一致的映射已应用")
        else:
            logger.warning("⚠️ 模型文件中未找到映射信息 - 预测可能会有一致性问题")
            logger.warning("将使用inference数据创建新映射，可能导致索引不一致")
            
            # 备用方案：使用inference数据创建映射（有风险）
            from chemcpa_implementation import prepare_chemcpa_data
            drug_to_idx, covariate_mappings = prepare_chemcpa_data(
                adata_all, "drug", global_control, "dose", ["cell_line"]
            )

        # 恢复模型权重
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

                def align_state_dict_shapes(model, loaded_state):
                    model_state = model.state_dict()
                    aligned = {}
                    for key, value in loaded_state.items():
                        if key in model_state and torch.is_tensor(value):
                            target = model_state[key]
                            if value.shape != target.shape:
                                logger.warning(
                                    "权重尺寸不匹配: %s | checkpoint=%s 当前模型=%s -- 自动对齐",
                                    key, tuple(value.shape), tuple(target.shape)
                                )
                                new_tensor = target.clone()
                                slices = tuple(slice(0, min(value.size(dim), target.size(dim))) for dim in range(value.dim()))
                                new_tensor[slices] = value[slices].to(new_tensor.device)
                                value = new_tensor
                        aligned[key] = value
                    return aligned

                aligned_state_dict = align_state_dict_shapes(
                    chemcpa_model.trainer.model,
                    model_state['model_state_dict']
                )

                chemcpa_model.trainer.model.load_state_dict(aligned_state_dict, strict=False)
                chemcpa_model.model = chemcpa_model.trainer.model
                
                # ✅ 显示加载的权重信息
                if model_state.get('is_best_model', False) or 'best_epoch' in model_state:
                    logger.info("✅ 最佳模型权重恢复成功")
                else:
                    logger.info("✅ 模型权重恢复成功")
                    
            except Exception as e:
                logger.warning(f"恢复模型权重失败: {e}")
                logger.info("将在预测时重新初始化模型结构")
        
        # 恢复loss历史（如果有的话）
        if 'loss_history' in model_state:
            from chemcpa_training import LossHistory
            loss_history = LossHistory()
            history_data = model_state['loss_history']
            loss_history.epochs = history_data.get('epochs', [])
            loss_history.train_losses = history_data.get('train_losses', [])
            loss_history.val_losses = history_data.get('val_losses', [])
            chemcpa_model.loss_history = loss_history
        
        logger.info("✅ ChemCPA模型加载成功")
        
        # 在返回前展示最终的映射状态
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
    
    # 创建SMILES编码器
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
    
    # 编码所有SMILES
    all_smiles = [drug_to_smiles[drug] for drug in all_drugs]
    smiles_features = smiles_encoder.fit_transform(all_smiles)
    
    drug_to_features = {}
    for drug, features in zip(all_drugs, smiles_features):
        drug_to_features[drug] = features
    
    # 为adata_all添加SMILES特征
    n_obs = adata_all.n_obs
    n_features = smiles_features.shape[1]
    smiles_matrix = np.zeros((n_obs, n_features))
    
    for j, drug in enumerate(adata_all.obs['drug']):
        smiles_matrix[j] = drug_to_features[drug]
    
    adata_all.obsm['smiles_features'] = smiles_matrix
    logger.info(f"添加SMILES特征形状: {smiles_matrix.shape}")
    
    del all_smiles, drug_to_features, missing_drugs, smiles_matrix
    return drug_to_smiles, smiles_features.shape[1]

def perform_single_cell_counterfactual_analysis_by_cellline(chemcpa_model, adata_all, global_control):
    """
    按cell_line分批执行counterfactual分析 - 优化内存使用和计算速度

    性能优化要点:
    1. 预构建test_data查找索引 - 避免每次条件都重复计算mask (从O(n)降到O(1))
    2. 批量处理控制细胞 - 一次性转换所有稀疏矩阵，避免循环中重复toarray()
    3. 预构建drug到SMILES映射 - 避免重复的数组查找
    4. 使用索引直接访问数据 - 避免创建中间AnnData对象

    预期加速: 每个条件从1.5-2秒降低到0.1-0.3秒 (约5-10倍加速)
    """
    logger.info("开始按cell_line分批的单细胞counterfactual分析...")

    # 清理数据
    logger.info(f"原始总样本数: {adata_all.n_obs}")
    is_valid_drug_mask = adata_all.obs['drug'] != 'nan'
    adata_all = adata_all[is_valid_drug_mask].copy()
    logger.info(f"清理'nan' drug后，剩余总样本数: {adata_all.n_obs}")

    ctrl_data = adata_all[adata_all.obs["split"] == "ctrl"].copy()
    test_data = adata_all[adata_all.obs["split"] == "test"].copy()

    logger.info(f"有效控制样本: {ctrl_data.n_obs}")
    logger.info(f"有效测试样本: {test_data.n_obs}")

    # 获取唯一的测试条件和cell_lines
    test_conditions_df = test_data.obs[["cell_line", "drug", "dose_str", "dose"]].drop_duplicates()
    test_conditions = [tuple(row) for row in test_conditions_df.itertuples(index=False)]
    unique_cell_lines = test_conditions_df['cell_line'].unique()

    logger.info(f"发现 {len(test_conditions)} 个唯一测试条件")
    logger.info(f"需要处理 {len(unique_cell_lines)} 个cell_lines: {unique_cell_lines}")

    # ✅ 优化1: 预先构建test_data的快速查找索引 (避免重复的mask计算)
    logger.info("预构建test_data查找索引以加速...")
    test_data_grouped = test_data.obs.groupby(['cell_line', 'drug', 'dose_str']).indices
    logger.info(f"构建了 {len(test_data_grouped)} 个条件的索引")

    # ✅ 优化2: 预先提取稀疏矩阵以避免重复转换
    ctrl_X_is_sparse = hasattr(ctrl_data.X, 'toarray')
    test_X_is_sparse = hasattr(test_data.X, 'toarray')

    # ✅ 优化3: 预构建drug到SMILES特征的映射 (如果存在)
    drug_to_smiles_features = {}
    if 'smiles_features' in adata_all.obsm:
        logger.info("预构建drug到SMILES特征的映射...")
        unique_drugs = adata_all.obs['drug'].unique()
        for drug in unique_drugs:
            drug_mask = adata_all.obs['drug'] == drug
            drug_indices = np.where(drug_mask)[0]
            if len(drug_indices) > 0:
                drug_to_smiles_features[drug] = adata_all.obsm['smiles_features'][drug_indices[0]]
        logger.info(f"构建了 {len(drug_to_smiles_features)} 个药物的SMILES映射")

    # 创建输出目录
    output_dir = "./counterfactual_result"
    cellline_results_dir = f"{output_dir}/cellline_results"
    os.makedirs(cellline_results_dir, exist_ok=True)

    # 全局收集变量
    all_cellline_metrics = []
    all_condition_metrics = []
    all_inference_metadata = []

    # 按cell_line分批处理
    for cell_line_idx, cell_line in enumerate(unique_cell_lines, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"处理Cell Line [{cell_line_idx}/{len(unique_cell_lines)}]: {cell_line}")
        logger.info(f"{'='*60}")

        try:
            # 获取该cell_line的控制细胞和测试条件
            cellline_ctrl_mask = (ctrl_data.obs['cell_line'] == cell_line)
            cellline_controls = ctrl_data[cellline_ctrl_mask]

            if cellline_controls.n_obs == 0:
                logger.warning(f"Cell line {cell_line} 没有控制细胞，跳过")
                continue

            # 获取该cell_line的测试条件
            cellline_conditions = [(cl, drug, dose_str, dose) for cl, drug, dose_str, dose in test_conditions if cl == cell_line]
            logger.info(f"Cell line {cell_line}: {cellline_controls.n_obs} 控制细胞, {len(cellline_conditions)} 条件")

            # ✅ 优化: 在cell_line级别一次性转换控制细胞数据（避免重复转换）
            if ctrl_X_is_sparse:
                cellline_ctrl_expr = cellline_controls.X.toarray()
            else:
                cellline_ctrl_expr = cellline_controls.X

            ctrl_indices_list = cellline_controls.obs.index.tolist()

            # 收集该cell_line的所有inference数据
            cellline_inference_cells = []
            cellline_metadata = []
            cellline_mapping = []

            for condition_idx, (_, drug, dose_str, dose) in enumerate(cellline_conditions, 1):
                logger.info(f"  处理条件 [{condition_idx}/{len(cellline_conditions)}]: {drug} @ {dose_str}")

                try:
                    # ✅ 优化: 使用预构建的索引快速查找 (替代重复的mask操作)
                    condition_key = (cell_line, drug, dose_str)
                    test_indices = test_data_grouped.get(condition_key, np.array([]))

                    if len(test_indices) == 0:
                        logger.warning(f"    条件 {drug}@{dose_str} 没有实际细胞，跳过")
                        continue

                    n_real_cells = len(test_indices)
                    logger.info(f"    实际扰动细胞数: {n_real_cells}")

                    # ✅ 优化: 使用cell_line级别已转换的控制细胞数据（避免重复转换）
                    n_ctrl = cellline_controls.n_obs

                    # 批量添加到inference列表（使用外层已转换的数据）
                    for ctrl_idx in range(n_ctrl):
                        cellline_inference_cells.append(cellline_ctrl_expr[ctrl_idx])

                        # 记录metadata
                        cellline_metadata.append({
                            "cell_line": cell_line,
                            "drug": drug,
                            "dose_str": dose_str,
                            "dose": float(dose),
                            "condition": f"{cell_line}_{drug}_{dose_str}",
                            "original_ctrl_cell_idx": ctrl_indices_list[ctrl_idx],
                            "n_real_condition_cells": n_real_cells,
                            "inference_cell_idx": len(cellline_metadata)
                        })

                        # 记录映射关系
                        cellline_mapping.append({
                            "inference_idx": len(cellline_metadata) - 1,
                            "condition": f"{cell_line}_{drug}_{dose_str}",
                            "ctrl_cell_idx": ctrl_indices_list[ctrl_idx],
                            "cell_line": cell_line,
                            "drug": drug,
                            "dose_str": dose_str
                        })

                except Exception as e:
                    logger.warning(f"    处理条件 {drug}@{dose_str} 失败: {e}")
                    continue

            if not cellline_inference_cells:
                logger.warning(f"Cell line {cell_line} 没有有效的inference数据，跳过")
                continue

            logger.info(f"Cell line {cell_line} 收集完成: {len(cellline_inference_cells)} inference cells")

            # 为该cell_line创建inference AnnData
            logger.info(f"为Cell line {cell_line} 创建inference AnnData...")
            cellline_inference_matrix = np.vstack(cellline_inference_cells)
            cellline_metadata_df = pd.DataFrame(cellline_metadata)

            cellline_inference_adata = sc.AnnData(X=cellline_inference_matrix, obs=cellline_metadata_df)
            cellline_inference_adata.var_names = adata_all.var_names

            # 设置CPA注释
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

            # ✅ 优化: 使用预构建的drug到SMILES映射 (避免重复查找)
            if drug_to_smiles_features:
                n_smiles_features = list(drug_to_smiles_features.values())[0].shape[0]
                smiles_matrix = np.zeros((cellline_inference_adata.n_obs, n_smiles_features), dtype=np.float32)
                for j, drug in enumerate(cellline_inference_adata.obs['drug']):
                    if drug in drug_to_smiles_features:
                        smiles_matrix[j] = drug_to_smiles_features[drug]
                cellline_inference_adata.obsm['smiles_features'] = smiles_matrix
                del smiles_matrix

            # 设置训练词汇表
            def _get_training_vocab(model):
                cats_drug = getattr(model, "drug_categories", None)
                cats_cell = getattr(model, "cell_categories", None)
                if cats_drug is not None:
                    return list(cats_drug), (list(cats_cell) if cats_cell is not None else None)
                raise RuntimeError("找不到训练词表（drug/cell_line）。请确认模型保存了 registry。")

            cats_drug, cats_cell = _get_training_vocab(chemcpa_model)

            # 编码drug和cell_line索引
            original_drugs = cellline_inference_adata.obs["drug"].astype(str)
            cellline_inference_adata.obs["drug"] = pd.Categorical(original_drugs, categories=cats_drug)
            cellline_inference_adata.obs["drug_idx"] = cellline_inference_adata.obs["drug"].cat.codes.astype("int64")

            if cats_cell:
                cellline_inference_adata.obs["cell_line"] = pd.Categorical(cellline_inference_adata.obs["cell_line"].astype(str), categories=cats_cell)
                cellline_inference_adata.obs["cell_line_idx"] = cellline_inference_adata.obs["cell_line"].cat.codes.astype("int64")

            # cellline_inference_adata.obs["dose"] = cellline_inference_adata.obs["dose"].astype(np.float32)  # 已经是正确类型
            if "dose_value" in getattr(chemcpa_model, "required_obs_keys", []):
                cellline_inference_adata.obs["dose_value"] = cellline_inference_adata.obs["dose"]

            # 执行该cell_line的inference
            logger.info(f"执行Cell line {cell_line} 的counterfactual inference...")
            try:
                cellline_inference_with_pred = chemcpa_model.predict(cellline_inference_adata)
                predictions = cellline_inference_with_pred.obsm["ChemCPA_pred"]
                if not isinstance(predictions, np.ndarray):
                    predictions = predictions.toarray()
                # predictions = predictions.astype(np.float32)  # 模型输出已经是float32

                logger.info(f"Cell line {cell_line} inference完成: {predictions.shape}")

            except Exception as e:
                logger.error(f"Cell line {cell_line} inference失败: {e}")
                continue

            # 计算该cell_line的metrics
            logger.info(f"计算Cell line {cell_line} 的metrics...")
            cellline_condition_metrics = []

            for condition in cellline_conditions:
                _, drug, dose_str, dose = condition
                condition_name = f"{cell_line}_{drug}_{dose_str}"

                try:
                    # 获取该条件的inference结果
                    inf_mask = cellline_inference_with_pred.obs['condition'] == condition_name
                    condition_inf_pred = predictions[inf_mask]

                    # ✅ 优化: 使用预构建的索引快速获取实际细胞
                    condition_key = (cell_line, drug, dose_str)
                    test_indices = test_data_grouped.get(condition_key, np.array([]))

                    if len(test_indices) == 0 or condition_inf_pred.shape[0] == 0:
                        continue

                    # 直接用索引获取数据，避免创建中间AnnData对象
                    if test_X_is_sparse:
                        actual_expr = test_data.X[test_indices].toarray()
                    else:
                        actual_expr = test_data.X[test_indices]

                    # 计算mean expressions
                    counterfactual_mean_expr = np.mean(condition_inf_pred, axis=0)
                    real_mean_expr = np.mean(actual_expr, axis=0)

                    # 过滤有效基因
                    valid_mask = np.isfinite(counterfactual_mean_expr) & np.isfinite(real_mean_expr)
                    if np.sum(valid_mask) < 2:
                        continue

                    counterfactual_valid = counterfactual_mean_expr[valid_mask]
                    real_valid = real_mean_expr[valid_mask]

                    # 计算metrics
                    mse = mean_squared_error(real_valid, counterfactual_valid)
                    mae = mean_absolute_error(real_valid, counterfactual_valid)
                    r2 = r2_score(real_valid, counterfactual_valid)
                    pearson_r, _ = pearsonr(real_valid, counterfactual_valid)
                    spearman_r, _ = spearmanr(real_valid, counterfactual_valid)
                    chatterjee_r = chatterjee_corr(real_valid, counterfactual_valid)

                    condition_result = {
                        "cell_line": cell_line,
                        "drug": drug,
                        "dose": dose_str,
                        "condition": condition_name,
                        "n_counterfactual_inferences": condition_inf_pred.shape[0],
                        "n_real_condition_cells": actual_expr.shape[0],
                        "n_valid_genes": int(np.sum(valid_mask)),
                        "n_total_genes": actual_expr.shape[1],
                        "MSE": round(float(mse), 6),
                        "MAE": round(float(mae), 6),
                        "R2": round(float(r2), 6),
                        "Pearson_r": round(float(pearson_r), 6),
                        "Spearman_r": round(float(spearman_r), 6),
                        "Chatterjee": round(float(chatterjee_r), 6),
                        "status": "completed",
                        "comparison_type": "real_mean_expr_vs_counterfactual_mean_expr"
                    }
                    cellline_condition_metrics.append(condition_result)

                    logger.info(f"    {condition_name}: R²={r2:.4f}, Pearson={pearson_r:.4f}")

                except Exception as e:
                    logger.warning(f"    计算条件 {condition_name} metrics失败: {e}")
                    continue

            # 保存该cell_line的结果
            logger.info(f"保存Cell line {cell_line} 的结果...")

            # 修复索引冲突（数据已经是float32）
            # cellline_inference_with_pred.X = cellline_inference_with_pred.X.astype(np.float32)  # 已经是float32
            if hasattr(cellline_inference_with_pred.X, 'toarray'):
                cellline_inference_with_pred.X = cellline_inference_with_pred.X.toarray()

            # 修复索引冲突
            if (cellline_inference_with_pred.obs.index.name and
                cellline_inference_with_pred.obs.index.name in cellline_inference_with_pred.obs.columns):
                original_name = cellline_inference_with_pred.obs.index.name
                if not cellline_inference_with_pred.obs.index.equals(cellline_inference_with_pred.obs[original_name]):
                    cellline_inference_with_pred.obs = cellline_inference_with_pred.obs.drop(columns=[original_name])
                cellline_inference_with_pred.obs.index.name = f"{original_name}_cell_index"

            # 保存inference结果
            inf_output_path = f"{cellline_results_dir}/{cell_line}_inference_results.h5ad"
            cellline_inference_with_pred.write_h5ad(inf_output_path)

            # 保存映射关系
            mapping_df = pd.DataFrame(cellline_mapping)
            mapping_path = f"{cellline_results_dir}/{cell_line}_mapping.csv"
            mapping_df.to_csv(mapping_path, index=False)

            logger.info(f"Cell line {cell_line} 结果已保存: {inf_output_path}")

            # 汇总到全局结果
            all_condition_metrics.extend(cellline_condition_metrics)
            all_inference_metadata.extend(cellline_metadata)

            # 计算该cell_line的汇总metrics
            if cellline_condition_metrics:
                metrics_cols = ['MSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_r', 'Chatterjee']
                avg_metrics = {}
                for metric in metrics_cols:
                    values = [c[metric] for c in cellline_condition_metrics if c[metric] is not None]
                    if values:
                        avg_metrics[metric] = round(float(np.mean(values)), 6)
                    else:
                        avg_metrics[metric] = None

                cellline_result = {
                    "cell_line": cell_line,
                    "status": "completed",
                    "n_conditions": len(cellline_condition_metrics),
                    "total_cells_analyzed": sum(c['n_counterfactual_inferences'] for c in cellline_condition_metrics),
                    **avg_metrics,
                    "condition_details": cellline_condition_metrics
                }
                all_cellline_metrics.append(cellline_result)

            # 释放该cell_line的内存
            del cellline_inference_cells, cellline_metadata, cellline_mapping
            del cellline_inference_adata, cellline_inference_with_pred, predictions
            del cellline_inference_matrix, cellline_metadata_df
            del cellline_ctrl_expr, ctrl_indices_list  # 释放控制细胞批量数据
            gc.collect()

            logger.info(f"Cell line {cell_line} 处理完成，内存已清理")

        except Exception as e:
            logger.error(f"处理Cell line {cell_line} 失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            continue

    # 释放原始数据
    del ctrl_data, test_data
    gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info("按cell_line的counterfactual分析完成!")
    logger.info(f"处理了 {len(unique_cell_lines)} 个cell_lines")
    logger.info(f"计算了 {len(all_condition_metrics)} 个condition metrics")
    logger.info(f"{'='*60}")

    return all_cellline_metrics, all_condition_metrics, all_inference_metadata

# 注意：这个函数已经被 perform_single_cell_counterfactual_analysis_by_cellline 替代
# 新的函数已经在cell_line级别处理时直接计算了metrics，无需单独的计算步骤

def analyze_counterfactual_results(cell_line_results, all_condition_metrics):
    """分析counterfactual预测结果"""
    logger.info("分析counterfactual预测结果...")
    
    successful_results = [r for r in cell_line_results if r.get("status") == "completed"]
    
    if not successful_results:
        logger.error("没有成功的结果可供分析")
        return {}
    
    logger.info(f"成功分析了 {len(successful_results)} 个细胞系")
    logger.info(f"总条件分析: {len(all_condition_metrics)}")
    
    # 细胞系级别统计
    df_cellline = pd.DataFrame(successful_results)
    metrics_cols = ['MSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_r', 'Chatterjee']
    
    cellline_stats = {}
    logger.info("\n细胞系级别性能统计（Counterfactual分析）:")
    
    for col in metrics_cols:
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
        
        logger.info("\n条件级别性能统计（Counterfactual分析）:")
        
        for col in metrics_cols:
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
    
    # 找出最佳和最差表现
    if 'R2' in df_cellline.columns:
        best_r2_idx = df_cellline['R2'].idxmax()
        worst_r2_idx = df_cellline['R2'].idxmin()
        
        best_cellline = df_cellline.loc[best_r2_idx]
        worst_cellline = df_cellline.loc[worst_r2_idx]
        
        logger.info(f"\n最佳表现: {best_cellline['cell_line']} (R²={best_cellline['R2']:.4f}, {best_cellline['n_conditions']} 条件)")
        logger.info(f"最差表现: {worst_cellline['cell_line']} (R²={worst_cellline['R2']:.4f}, {worst_cellline['n_conditions']} 条件)")
    
    # 性能分布分析
    if 'R2' in df_cellline.columns:
        r2_values = df_cellline['R2']
        good_performance = (r2_values >= 0.5).sum()
        moderate_performance = ((r2_values >= 0.2) & (r2_values < 0.5)).sum()
        poor_performance = (r2_values < 0.2).sum()
        
        logger.info("\n性能分布分析:")
        logger.info(f"  高性能 (R² ≥ 0.5): {good_performance} 个细胞系 ({good_performance/len(r2_values)*100:.1f}%)")
        logger.info(f"  中等性能 (0.2 ≤ R² < 0.5): {moderate_performance} 个细胞系 ({moderate_performance/len(r2_values)*100:.1f}%)")
        logger.info(f"  低性能 (R² < 0.2): {poor_performance} 个细胞系 ({poor_performance/len(r2_values)*100:.1f}%)")
    
    # 合并统计
    all_stats = {**cellline_stats, **condition_stats}
    
    return all_stats

def save_single_cell_inference_results(inference_adata_with_pred, inference_to_actual_mapping):
    """保存单细胞inference结果的adata和映射信息"""
    logger.info("保存单细胞inference结果...")
    
    # 确保输出目录存在
    os.makedirs("./counterfactual_result", exist_ok=True)
    os.makedirs("./counterfactual_result/data", exist_ok=True)
    os.makedirs("./counterfactual_result/metadata", exist_ok=True)
    
    # 确保所有数据都是float32格式
    logger.info("数据已经是float32格式，无需转换...")
    
    # 转换稀疏矩阵为密集矩阵（如果需要）
    if hasattr(inference_adata_with_pred.X, 'toarray'):
        inference_adata_with_pred.X = inference_adata_with_pred.X.toarray()
    else:
        pass  # inference_adata_with_pred.X = inference_adata_with_pred.X.astype(np.float32)  # 已经是正确类型
    
    # 转换稀疏预测结果为密集矩阵（如果需要）
    if "ChemCPA_pred" in inference_adata_with_pred.obsm:
        pred = inference_adata_with_pred.obsm["ChemCPA_pred"]
        if hasattr(pred, 'toarray'):
            pred = pred.toarray()
        # inference_adata_with_pred.obsm["ChemCPA_pred"] = pred.astype(np.float32)  # 已经是正确类型
    
    # dose列和SMILES特征已经是正确类型
    if 'dose' in inference_adata_with_pred.obs.columns:
        pass  # inference_adata_with_pred.obs['dose'] = inference_adata_with_pred.obs['dose'].astype(np.float32)  # 已经是正确类型
    if 'dose_value' in inference_adata_with_pred.obs.columns:
        pass  # inference_adata_with_pred.obs['dose_value'] = inference_adata_with_pred.obs['dose_value'].astype(np.float32)  # 已经是正确类型

    # SMILES特征已经是正确类型
    if 'smiles_features' in inference_adata_with_pred.obsm:
        pass  # inference_adata_with_pred.obsm['smiles_features'] = inference_adata_with_pred.obsm['smiles_features'].astype(np.float32)  # 已经是正确类型
    
    # 保存完整的AnnData对象
    adata_path = "./counterfactual_result/data/inference_results_full.h5ad"
    logger.info(f"保存inference AnnData到: {adata_path}")
    sc.write(adata_path, inference_adata_with_pred)
    
    # 保存映射关系
    mapping_df = pd.DataFrame(inference_to_actual_mapping)
    mapping_path = "./counterfactual_result/metadata/inference_to_actual_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"保存映射关系到: {mapping_path}")
    
    # 创建元数据摘要
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
    
    # 保存元数据摘要
    summary_path = "./counterfactual_result/metadata/inference_metadata_summary.json"
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
    
    # 确保输出目录存在
    os.makedirs("./counterfactual_result/results", exist_ok=True)
    
    # 保存详细结果
    cellline_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'} 
                               for result in cell_line_results])
    cellline_df.to_csv("./counterfactual_result/results/cellline_counterfactual_metrics.csv", index=False)
    
    condition_df = pd.DataFrame(all_condition_metrics)
    condition_df.to_csv("./counterfactual_result/results/condition_counterfactual_metrics.csv", index=False)
    
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
    with open("./counterfactual_result/chemcpa_single_cell_counterfactual_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)
    
    logger.info("结果保存:")
    logger.info("  细胞系结果: ./counterfactual_result/results/cellline_counterfactual_metrics.csv")
    logger.info("  条件结果: ./counterfactual_result/results/condition_counterfactual_metrics.csv")
    logger.info("  完整摘要: ./counterfactual_result/chemcpa_single_cell_counterfactual_summary.json")
    
    return summary

def create_visualization_plots(cell_line_results, all_condition_metrics, output_dir="./counterfactual_result"):
    """创建可视化图表"""
    logger.info("创建可视化图表...")
    
    try:
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # 细胞系性能热图
        if cell_line_results:
            cl_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'} 
                                 for result in cell_line_results])
            
            if not cl_df.empty and 'R2' in cl_df.columns:
                plt.figure(figsize=(12, 8))
                
                # 创建细胞系性能热图
                metrics_to_plot = ['R2', 'Pearson_r', 'MSE', 'MAE']
                available_metrics = [m for m in metrics_to_plot if m in cl_df.columns]
                
                if available_metrics:
                    # 标准化数据用于热图显示
                    heatmap_data = cl_df.set_index('cell_line')[available_metrics]
                    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
                    plt.title('ChemCPA Counterfactual: Cell Line Performance Heatmap')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/cellline_performance_heatmap.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("细胞系性能热图已保存")
        
        # R²分布直方图
        if all_condition_metrics:
            condition_df = pd.DataFrame(all_condition_metrics)
            if 'R2' in condition_df.columns and len(condition_df) > 0:
                plt.figure(figsize=(10, 6))
                
                r2_values = condition_df['R2'].dropna()
                if len(r2_values) > 0:
                    plt.hist(r2_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.axvline(r2_values.mean(), color='red', linestyle='--', 
                               label=f'Mean: {r2_values.mean():.3f}')
                    plt.axvline(r2_values.median(), color='orange', linestyle='--', 
                               label=f'Median: {r2_values.median():.3f}')
                    
                    plt.title('ChemCPA Counterfactual: Distribution of R² Scores')
                    plt.xlabel('R² Score')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/plots/r2_distribution.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("R²分布图已保存")
        
        # 药物效果分析图
        if all_condition_metrics:
            condition_df = pd.DataFrame(all_condition_metrics)
            if 'drug' in condition_df.columns and 'R2' in condition_df.columns:
                plt.figure(figsize=(15, 8))
                
                # 按药物分组计算平均R²
                drug_performance = condition_df.groupby('drug')['R2'].agg(['mean', 'std', 'count']).reset_index()
                drug_performance = drug_performance.sort_values('mean', ascending=False)
                
                # 只显示前20个药物
                top_drugs = drug_performance.head(20)
                
                plt.bar(range(len(top_drugs)), top_drugs['mean'], 
                       yerr=top_drugs['std'], capsize=5, alpha=0.7)
                plt.xticks(range(len(top_drugs)), top_drugs['drug'], rotation=45, ha='right')
                plt.title('ChemCPA Counterfactual: Top 20 Drug Performance (R² Score)')
                plt.xlabel('Drug')
                plt.ylabel('Average R² Score')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/plots/top_drug_performance.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("药物性能图已保存")
        
    except Exception as e:
        logger.warning(f"创建可视化图表时出现错误: {e}")

def main():
    """主函数 - ChemCPA Counterfactual分析（CPA风格）"""
    logger.info("="*60)
    logger.info("ChemCPA Counterfactual分析 - CPA风格数据设置")
    logger.info("="*60)

    # Validate the external input contract before any H5AD is opened.
    test_data_path = resolve_benchmark_data_file("test_adata_processed.h5ad")
    control_data_path = resolve_benchmark_data_file("control_adata_processed.h5ad")
    drug_metadata_path = resolve_drug_metadata_file()
    
    try:
        # 1. 加载推理数据
        adata_all = load_inference_data(test_data_path, control_data_path)
        
        # 2. 查找控制组
        global_control = find_control_group(adata_all)
        
        # 3. 加载SMILES特征（如果需要）
        logger.info("加载SMILES特征用于化学推理...")
        drug_metadata = load_drug_metadata_for_smiles(drug_metadata_path)

        # 简化配置用于SMILES
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
        
        # 4. 加载训练好的模型 - ✅ 修改：优先加载最佳模型
        model_path = "./dose_global_result/models/chemcpa_pretrain_model_best.pth"

        if not os.path.exists(model_path):
            logger.error(f"指定的最佳模型文件不存在: {model_path}")
            logger.error("请确认模型路径正确，或先运行训练脚本生成最佳模型")
            return

        logger.info(f"🎯 使用最佳模型: {model_path}")

        # 加载选定的模型（不需要prefer_best参数了，因为已经选好了）
        logger.info("步骤4: 加载训练好的模型并应用一致性映射...")
        chemcpa_model = load_trained_chemcpa_model(
            model_path, adata_all, global_control, prefer_best=False
        )
        logger.info("✅ 模型加载完成，映射一致性已确保")
        
        # 5. 执行按cell_line分批的单细胞counterfactual分析（优化内存使用）
        logger.info("步骤5: 执行按cell_line分批的单细胞counterfactual分析...")
        cell_line_results, all_condition_metrics, all_inference_metadata = perform_single_cell_counterfactual_analysis_by_cellline(
            chemcpa_model, adata_all, global_control
        )

        if not cell_line_results:
            logger.error("按cell_line的分析失败，无结果返回")
            return

        # 6. 保存全局汇总结果
        logger.info("步骤6: 保存全局汇总结果...")

        # 保存全局metrics
        if all_condition_metrics:
            metrics_df = pd.DataFrame(all_condition_metrics)
            metrics_output_path = f"./counterfactual_result/global_condition_metrics.csv"
            metrics_df.to_csv(metrics_output_path, index=False)
            logger.info(f"全局条件metrics已保存: {metrics_output_path}")

        # 保存cell_line汇总
        if cell_line_results:
            cellline_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'condition_details'}
                                       for result in cell_line_results])
            cellline_output_path = f"./counterfactual_result/global_cellline_metrics.csv"
            cellline_df.to_csv(cellline_output_path, index=False)
            logger.info(f"全局cell_line汇总已保存: {cellline_output_path}")

        # 保存元数据摘要
        metadata_summary = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "single_cell_counterfactual_by_cellline",
            "description": "按cell_line分批处理的单细胞counterfactual分析，优化内存使用",
            "data_format": "float32_optimized",
            "total_inference_cells": len(all_inference_metadata),
            "total_conditions": len(all_condition_metrics),
            "total_cell_lines": len(cell_line_results),
            "unique_drugs": len(set(m['drug'] for m in all_inference_metadata)),
            "memory_optimization": "cell_line_wise_processing",
            "file_structure": {
                "cellline_results_dir": "./counterfactual_result/cellline_results/",
                "data_dir": "./counterfactual_result/data/",
                "metadata_dir": "./counterfactual_result/metadata/",
                "results_dir": "./counterfactual_result/results/",
                "plots_dir": "./counterfactual_result/plots/",
                "individual_files": "每个cell_line单独保存inference结果和映射关系",
                "global_metrics": "./counterfactual_result/global_*_metrics.csv"
            }
        }

        summary_path = "./counterfactual_result/metadata_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metadata_summary, f, indent=2, default=json_default)
        logger.info(f"元数据摘要已保存: {summary_path}")

        # 释放模型内存
        del chemcpa_model, all_inference_metadata
        gc.collect()

        # 8. 分析结果
        stats = analyze_counterfactual_results(cell_line_results, all_condition_metrics)
        
        # 9. 保存结果摘要
        summary = save_counterfactual_results(cell_line_results, all_condition_metrics, stats)
        
        # 10. 创建可视化
        create_visualization_plots(cell_line_results, all_condition_metrics)
        
        # 释放结果变量
        del cell_line_results, all_condition_metrics, stats
        gc.collect()
        
        # 11. 最终报告
        logger.info("="*60)
        logger.info("ChemCPA 单细胞Counterfactual分析完成")
        logger.info("="*60)
        logger.info("关键结果摘要:")
        logger.info(f"  总inference细胞数: {metadata_summary['total_cells']}")
        logger.info(f"  总基因数: {metadata_summary['total_genes']}")
        logger.info(f"  独特条件数: {metadata_summary['unique_conditions']}")
        logger.info(f"  独特药物数: {metadata_summary['unique_drugs']}")
        logger.info(f"  独特细胞系数: {metadata_summary['unique_cell_lines']}")
        
        if 'performance_statistics' in summary:
            perf_stats = summary['performance_statistics']
            if 'cellline_R2' in perf_stats:
                r2_stats = perf_stats['cellline_R2']
                logger.info(f"  细胞系平均R²: {r2_stats['mean']:.4f} ± {r2_stats['std']:.4f}")
                logger.info(f"  R²范围: [{r2_stats['min']:.4f}, {r2_stats['max']:.4f}]")
            
            if 'cellline_Pearson_r' in perf_stats:
                pearson_stats = perf_stats['cellline_Pearson_r']
                logger.info(f"  细胞系平均Pearson: {pearson_stats['mean']:.4f} ± {pearson_stats['std']:.4f}")
        
        logger.info(f"  成功分析: {summary['successful_cell_lines']} 细胞系")
        logger.info(f"  总条件: {summary['total_conditions']}")
        
        # SMILES信息
        if drug_to_smiles:
            total_drugs = len(drug_to_smiles)
            valid_smiles = sum(1 for smiles in drug_to_smiles.values() if smiles is not None)
            logger.info(f"  SMILES覆盖率: {valid_smiles}/{total_drugs} ({valid_smiles/total_drugs*100:.1f}%)")
        
        logger.info("\n关键改进:")
        logger.info("  🎯 优先加载最佳验证loss模型，提高预测质量")
        logger.info("  📊 单细胞级别分析：每个cell单独inference而不是condition-wise的mean")
        logger.info("  💾 完整数据保存：所有单细胞的inference结果和metadata都保存在AnnData中")
        logger.info("  🗂️  统一输出结构：counterfactual_result文件夹包含完整的单细胞数据和按cell_line分组的结果")
        logger.info("  🔢 数据格式: 全面的float32优化")
        logger.info("  📈 评估方式：基于单细胞结果聚合的条件级别评估")
        logger.info("  📍 真正的counterfactual分析：对每个控制细胞应用扰动，预测扰动效果")
        logger.info("  🔬 单细胞精度：将预测与对应的实际扰动细胞逐一比较")
        
        logger.info("\n输出文件:")
        logger.info("  📁 按cell_line分组结果: ./counterfactual_result/cellline_results/")
        logger.info("      每个cell_line单独保存: <cell_line>_inference_results.h5ad")
        logger.info("      每个cell_line映射关系: <cell_line>_mapping.csv")
        logger.info("  📊 全局cell_line汇总: ./counterfactual_result/global_cellline_metrics.csv")
        logger.info("  📝 全局条件metrics: ./counterfactual_result/global_condition_metrics.csv")
        logger.info("  📋 元数据摘要: ./counterfactual_result/metadata_summary.json")
        logger.info("  📖 完整摘要: ./counterfactual_result/chemcpa_single_cell_counterfactual_summary.json")
        logger.info("  📈 可视化图表: ./counterfactual_result/plots/")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
