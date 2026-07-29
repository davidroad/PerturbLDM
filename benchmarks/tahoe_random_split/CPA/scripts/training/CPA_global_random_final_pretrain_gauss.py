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

import time
import logging
import traceback
import sys
from datetime import datetime
import gc

# 在文件开头添加这些强制CPU设置

# 1. 在导入任何深度学习库之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 完全禁用CUDA
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # 修复OpenBLAS问题
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import torch
# 2. 强制PyTorch使用CPU
torch.cuda.is_available = lambda: False  # 欺骗PyTorch认为没有CUDA
torch.set_num_threads(16)
torch.set_num_interop_threads(16)

def json_default(obj):
    """Handle numpy types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# 导入增强分析工具
try:
    from cpa_enhanced_analyzer import CPAEnhancedAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
    print("Enhanced analyzer imported successfully")
except ImportError as e:
    ENHANCED_ANALYZER_AVAILABLE = False
    print(f"Enhanced analyzer import failed: {e}")
    print("Using basic analysis functionality")

def setup_logging():
    """设置日志系统"""
    log_dir = "./random_gauss_result/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/cpa_global_training_gauss_{timestamp}.log"
    
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
logger.info(f"torch threads: {torch.get_num_threads()}, interop threads: {torch.get_num_interop_threads()}")

def analyze_data_distribution(adata_train, adata_test):
    """分析数据分布"""
    logger.info("Analyzing data distribution...")
    
    train_counts = adata_train.obs['cell_line'].value_counts()
    test_counts = adata_test.obs['cell_line'].value_counts()
    
    distribution_df = pd.DataFrame({
        'cell_line': train_counts.index,
        'train_samples': train_counts.values,
        'test_samples': test_counts.reindex(train_counts.index).fillna(0).astype(int)
    })
    
    distribution_df['total_samples'] = distribution_df['train_samples'] + distribution_df['test_samples']
    distribution_df = distribution_df.sort_values('total_samples', ascending=False)
    
    logger.info(f"Data distribution stats:")
    logger.info(f"  Total cell lines: {len(distribution_df)}")
    logger.info(f"  Train samples range: {distribution_df['train_samples'].min()} - {distribution_df['train_samples'].max()}")
    logger.info(f"  Test samples range: {distribution_df['test_samples'].min()} - {distribution_df['test_samples'].max()}")
    logger.info(f"  Average train samples: {distribution_df['train_samples'].mean():.1f}")
    logger.info(f"  Average test samples: {distribution_df['test_samples'].mean():.1f}")
    
    # 保存分布信息
    os.makedirs("./random_gauss_result/analysis", exist_ok=True)
    distribution_df.to_csv("./random_gauss_result/analysis/data_distribution.csv", index=False)
    
    return distribution_df

def prepare_global_data(adata_all):
    """准备全局训练数据"""
    logger.info("Preparing global training data...")
    
    # 分析数据分布
    train_data = adata_all[adata_all.obs["split"] == "train"]
    test_data = adata_all[adata_all.obs["split"] == "test"]
    distribution_df = analyze_data_distribution(train_data, test_data)
    
    # 使用简化的控制组设置
    global_control = "DMSO-TF"
    logger.info(f"Using global control group: {global_control}")
    
    # 验证控制组在数据中的存在性
    control_mask = adata_all.obs["drug"] == global_control
    control_count = np.sum(control_mask)
    
    if control_count == 0:
        logger.warning(f"Warning: Control group {global_control} not found in data")
        logger.info("Checking available drug types:")
        all_drugs = set(adata_all.obs["drug"].unique())
        logger.info(f"Available drugs: {sorted(all_drugs)}")
        
        # 如果没有找到DMSO-TF，选择其他控制组
        dmso_alternatives = [drug for drug in all_drugs if "DMSO" in drug.upper()]
        if dmso_alternatives:
            global_control = dmso_alternatives[0]
            control_count = np.sum(adata_all.obs["drug"] == global_control)
            logger.info(f"Using alternative control group: {global_control} ({control_count} samples)")
        else:
            logger.error("No suitable control group found")
            raise ValueError("No suitable control group found")
    else:
        logger.info(f"Control group validation successful: {global_control} ({control_count} samples)")
    
    return distribution_df, global_control

def train_global_model(adata_all, global_control):
    """训练全局CPA模型"""
    logger.info("Starting global CPA model training...")
    start_time = time.time()
    
    try:
        logger.info(f"Training data info:")
        split_counts = adata_all.obs["split"].value_counts().to_dict()
        logger.info(f"  Split counts: {split_counts}")
        logger.info(f"  Total samples: {adata_all.n_obs}, genes: {adata_all.n_vars}")
        
        # 临时抑制进度条和详细输出
        import warnings
        import sys
        from io import StringIO
        
        warnings.filterwarnings('ignore')
        
        # 设置scanpy的详细度级别为最低
        sc.settings.verbosity = 0
        
        # 如果CPA有详细度设置，也降低它
        if hasattr(CPA, 'verbosity'):
            CPA.verbosity = 0
        
        # 临时重定向stdout和stderr来完全抑制进度条
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            # 设置CPA
            logger.info("Setting up CPA model...")
            CPA.split_key = "split"
            CPA.setup_anndata(
                adata=adata_all,
                perturbation_key="drug",
                control_group=global_control,
                dosage_key="dose",
                categorical_covariate_keys=["cell_line"]
            )
        finally:
            # 恢复正常的输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # 恢复正常的详细度级别
        sc.settings.verbosity = 1
        warnings.filterwarnings('default')
        
        # 强制使用CPU
        CPA.device = "cpu"
        logger.info("Using CPU for training")
        
        # 创建模型实例
        cpa_model = CPA(adata=adata_all, recon_loss="gauss")
        
        # 训练模型
        logger.info("Starting training...")
        
        # 在训练期间抑制部分输出
        old_stdout_train = sys.stdout
        old_stderr_train = sys.stderr
        
        # 创建一个过滤器，只保留重要信息
        class FilteredOutput:
            def __init__(self, original):
                self.original = original
                
            def write(self, text):
                # 只输出包含epoch, loss等关键信息的行
                if any(keyword in text.lower() for keyword in ['epoch', 'loss', 'validation', 'early stopping']):
                    self.original.write(text)
                    
            def flush(self):
                self.original.flush()
        
        try:
            sys.stdout = FilteredOutput(old_stdout_train)
            sys.stderr = FilteredOutput(old_stderr_train)
            
            # 训练模型 - 增大batch size并调整loss weights
            cpa_model.train(
                max_epochs=10,
                batch_size=131072,  # 增大batch size
                plan_kwargs={
                    "lr": 1e-3,
                    "reg_adv": 200.0,    #balance to 1/3 of recon loss ~6000, adv loss ~10
                    "pen_adv": 400.0,    #two times of reg_adv
                },
                early_stopping=True,
                early_stopping_patience=10,
                early_stopping_min_delta=0.0001,
                enable_checkpointing=True,
                check_val_every_n_epoch=10
            )
        finally:
            sys.stdout = old_stdout_train
            sys.stderr = old_stderr_train
        
        training_time = time.time() - start_time
        logger.info(f"Global model training completed, time: {training_time:.2f} seconds")
        
        # 保存模型
        os.makedirs("./random_gauss_result/models", exist_ok=True)
        model_path = "./random_gauss_result/models/cpa_global_model_gauss.pth"
        cpa_model.save(model_path, overwrite=True)
        logger.info(f"Model saved: {model_path}")
        
        return cpa_model, training_time
        
    except Exception as e:
        logger.error(f"Global model training failed: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

def enhanced_training_analysis(cpa_model, adata_all):
    """使用增强分析工具进行训练过程分析"""
    if not ENHANCED_ANALYZER_AVAILABLE:
        logger.warning("Enhanced analyzer not available, skipping training analysis")
        return None
    
    try:
        logger.info("Starting enhanced training analysis...")
        
        # 创建增强分析器
        analyzer = CPAEnhancedAnalyzer(cpa_model)
        
        # 执行完整分析
        analysis_results = analyzer.get_complete_analysis(adata_all)
        
        # 生成所有分析图片
        os.makedirs("./random_gauss_result/enhanced_analysis", exist_ok=True)
        plots_generated = analyzer.generate_all_plots("./random_gauss_result/enhanced_analysis/")
        
        # 保存分析结果
        with open("./random_gauss_result/enhanced_analysis/complete_analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=json_default)
        
        # 打印关键分析结果
        logger.info("Training analysis results:")
        
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            logger.info(f"  Overall status: {summary.get('overall_status', 'unknown')}")
            
            for finding in summary.get('key_findings', []):
                logger.info(f"  {finding}")
            
            for recommendation in summary.get('recommendations', []):
                logger.info(f"  Recommendation: {recommendation}")
        
        # CPA指标详情
        if 'cpa_metric' in analysis_results and 'summary' in analysis_results['cpa_metric']:
            cpa_summary = analysis_results['cpa_metric']['summary']
            logger.info(f"  Final CPA metric: {cpa_summary.get('final_cpa_metric', 'N/A'):.4f}")
            logger.info(f"  Best CPA metric: {cpa_summary.get('best_cpa_metric', 'N/A'):.4f}")
        
        # 解耦效果
        if 'disentanglement' in analysis_results and 'summary' in analysis_results['disentanglement']:
            disent_summary = analysis_results['disentanglement']['summary']
            logger.info(f"  Disentanglement improvement: {disent_summary.get('final_improvement', 'N/A'):.4f}")
        
        logger.info(f"Generated {len(plots_generated)} analysis plots")
        logger.info("Complete analysis results saved in: ./random_gauss_result/enhanced_analysis/")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Enhanced training analysis failed: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

def create_training_summary(training_time, enhanced_analysis, distribution_df):
    """创建训练总结"""
    logger.info("Creating training summary...")
    
    # 创建训练总结
    summary = {
        "timestamp": datetime.now().isoformat(),
        "training_approach": "global_train_only_gauss",
        "training_time_seconds": round(training_time, 2),
        "data_distribution": {
            "total_cell_lines": len(distribution_df),
            "train_samples_range": [int(distribution_df['train_samples'].min()), int(distribution_df['train_samples'].max())],
            "test_samples_range": [int(distribution_df['test_samples'].min()), int(distribution_df['test_samples'].max())],
            "avg_train_samples": float(distribution_df['train_samples'].mean()),
            "avg_test_samples": float(distribution_df['test_samples'].mean())
        },
        "model_info": {
            "model_type": "CPA",
            "saved_path": "./random_gauss_result/models/cpa_global_model_gauss.pth"
        }
    }
    
    # 添加增强分析结果
    if enhanced_analysis is not None:
        summary['enhanced_training_analysis'] = enhanced_analysis
        logger.info("Enhanced training analysis integrated to final results")
    
    # 保存总结
    with open("./random_gauss_result/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)
    
    logger.info("Training summary saved to ./random_gauss_result/training_summary.json")
    
    return summary

def main():
    logger.info("="*60)
    logger.info("Starting CPA Global Training (Training Only)")
    logger.info("="*60)
    
    # 载入预处理的数据
    logger.info("Loading preprocessed data...")
    try:
        adata_all = sc.read_h5ad('./cpa_preprocessed_data/adata_all_concatenated_random.h5ad')
        
        # Convert expression data to float32 for better CPU performance
        if hasattr(adata_all.X, 'toarray'):
            adata_all.X = adata_all.X.astype(np.float32)
        else:
            adata_all.X = adata_all.X.astype(np.float32)
            
        logger.info(f"Data loaded successfully: {adata_all.n_obs} cells, {adata_all.n_vars} genes (dtype: {adata_all.X.dtype})")
        
        split_counts = adata_all.obs["split"].value_counts().to_dict()
        logger.info(f"Split distribution: {split_counts}")
        
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        logger.error("Please run the preprocessing script first to create the concatenated dataset")
        return
    
    try:
        # 准备数据
        distribution_df, global_control = prepare_global_data(adata_all)
        
        # 训练全局模型
        cpa_model, training_time = train_global_model(adata_all, global_control)
        
        # 训练过程增强分析
        enhanced_analysis = enhanced_training_analysis(cpa_model, adata_all)
        
        # 创建训练总结
        summary = create_training_summary(training_time, enhanced_analysis, distribution_df)
        
        logger.info("="*60)
        logger.info("CPA Global Training Completed")
        logger.info("="*60)
        logger.info("Key results summary:")
        logger.info(f"  Training time: {summary['training_time_seconds']:.2f} seconds")
        logger.info(f"  Total cell lines: {summary['data_distribution']['total_cell_lines']}")
        
        # 增强分析摘要
        if enhanced_analysis and 'summary' in enhanced_analysis:
            enhanced_summary = enhanced_analysis['summary']
            logger.info("Enhanced analysis summary:")
            for finding in enhanced_summary.get('key_findings', [])[:3]:  # 显示前3个关键发现
                logger.info(f"  {finding}")
        
        logger.info("\nOutput files:")
        logger.info("  Training summary: ./random_gauss_result/training_summary.json")
        logger.info("  Saved model: ./random_gauss_result/models/cpa_global_model_gauss.pth")
        logger.info("  Data distribution: ./random_gauss_result/analysis/data_distribution.csv")
        if ENHANCED_ANALYZER_AVAILABLE:
            logger.info("  Analysis plots: ./random_gauss_result/enhanced_analysis/")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main()
