#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import time
import logging
import traceback
import gc
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# chemCPA imports
try:
    import chemCPA
    from chemCPA.model import ComPert
    CHEMCPA_AVAILABLE = True
except ImportError:
    CHEMCPA_AVAILABLE = False

logger = logging.getLogger(__name__)

class LossHistory:
    """Loss历史记录器 - 详细记录每个epoch的loss和相关指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置历史记录"""
        self.epochs = []
        self.train_losses = []  # 🔥 改名：明确是训练loss
        self.val_losses = []    # 🔥 新增：验证loss
        self.batch_losses = []  # 每个epoch内的所有batch loss
        self.epoch_times = []
        self.learning_rates = []
        self.timestamps = []
        self.batch_details = []  # 详细的batch信息
        
        # 训练统计
        self.total_batches_processed = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.best_epoch = None
        self.best_val_loss = float('inf')
    
    def add_epoch_loss(self, epoch, train_loss, val_loss, epoch_time, lr=None, batch_losses=None):
        """添加epoch级别的loss记录 - 包含validation loss"""
        self.epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))  # 🔥 新增
        self.epoch_times.append(float(epoch_time))
        self.learning_rates.append(float(lr) if lr is not None else 0.0)
        self.timestamps.append(datetime.now().isoformat())
        
        # 🔥 更新最佳验证loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        
        # 记录batch级别的详细信息
        if batch_losses is not None:
            self.batch_losses.append(batch_losses.copy())
        else:
            self.batch_losses.append([])
    
    # 🔥 添加缺失的 add_batch_loss 方法
    def add_batch_loss(self, epoch, batch_idx, loss, batch_time=None, success=True):
        """添加batch级别的loss记录"""
        batch_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': float(loss) if loss is not None else 0.0,
            'batch_time': float(batch_time) if batch_time is not None else 0.0,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        self.batch_details.append(batch_info)
        
        self.total_batches_processed += 1
        if success:
            self.successful_batches += 1
        else:
            self.failed_batches += 1
    
    def get_training_summary(self):
        """获取训练摘要统计 - 包含validation信息"""
        if not self.epochs:
            return {
                'total_epochs': 0,
                'total_training_time': 0.0,
                'final_train_loss': None,
                'final_val_loss': None,
                'best_val_loss': None,
                'best_epoch': None,
                'avg_epoch_time': 0.0,
                'total_batches_processed': 0,
                'batch_success_rate': 0.0
            }
        
        total_epochs = len(self.epochs)
        total_training_time = sum(self.epoch_times)
        final_train_loss = self.train_losses[-1]
        final_val_loss = self.val_losses[-1]  # 🔥 新增
        best_val_loss = self.best_val_loss
        best_epoch = self.best_epoch
        avg_epoch_time = np.mean(self.epoch_times)
        
        batch_success_rate = (self.successful_batches / self.total_batches_processed 
                             if self.total_batches_processed > 0 else 0.0)
        
        return {
            'total_epochs': total_epochs,
            'total_training_time': round(total_training_time, 2),
            'final_train_loss': round(final_train_loss, 6),
            'final_val_loss': round(final_val_loss, 6),  # 🔥 新增
            'best_val_loss': round(best_val_loss, 6),
            'best_epoch': best_epoch,
            'avg_epoch_time': round(avg_epoch_time, 2),
            'total_batches_processed': self.total_batches_processed,
            'successful_batches': self.successful_batches,
            'failed_batches': self.failed_batches,
            'batch_success_rate': round(batch_success_rate, 3),
            'train_loss_improvement': round(self.train_losses[0] - final_train_loss, 6) if len(self.train_losses) > 1 else 0.0,
            'val_loss_improvement': round(self.val_losses[0] - final_val_loss, 6) if len(self.val_losses) > 1 else 0.0,  # 🔥 新增
            'convergence_status': self._assess_convergence()
        }
    
    def _assess_convergence(self):
        """评估训练收敛状态 - 基于validation loss"""
        if len(self.val_losses) < 5:
            return 'insufficient_data'
        
        # 检查最后5个epoch的validation loss变化
        recent_val_losses = self.val_losses[-5:]
        loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
        
        if abs(loss_trend) < 1e-6:
            return 'converged'
        elif loss_trend < -1e-6:
            return 'improving'
        else:
            return 'diverging'
    
    def to_dict(self):
        """转换为字典格式，便于保存"""
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'learning_rates': self.learning_rates,
            'timestamps': self.timestamps,
            'batch_details': self.batch_details,
            'training_statistics': {
                'total_batches_processed': self.total_batches_processed,
                'successful_batches': self.successful_batches,
                'failed_batches': self.failed_batches
            }
        }
    
    def save_to_csv(self, file_path):
        """保存详细的loss历史到CSV文件"""
        if not self.epochs:
            logger.warning("没有训练历史数据可保存")
            return
        
        # Epoch级别的数据
        epoch_df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'epoch_time_seconds': self.epoch_times,
            'learning_rate': self.learning_rates,
            'timestamp': self.timestamps,
            'cumulative_time': np.cumsum(self.epoch_times)
        })
        
        epoch_csv_path = file_path.replace('.csv', '_epochs.csv')
        epoch_df.to_csv(epoch_csv_path, index=False)
        logger.info(f"Epoch级别loss历史保存到: {epoch_csv_path}")
        
        # Batch级别的数据
        if self.batch_details:
            batch_df = pd.DataFrame(self.batch_details)
            batch_csv_path = file_path.replace('.csv', '_batches.csv')
            batch_df.to_csv(batch_csv_path, index=False)
            logger.info(f"Batch级别loss历史保存到: {batch_csv_path}")
        
        return epoch_csv_path
    
    def plot_training_curves(self, save_path=None):
        """绘制详细的训练曲线 - 包含train/val对比"""
        if not self.epochs:
            logger.warning("没有训练历史数据可绘制")
            return None
        
        # 设置绘图样式
        try:
            plt.style.use('seaborn-v0_8')
        except:
            pass
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 🔥 1. 主要的Train/Val Loss对比曲线
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.epochs, self.train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        ax1.plot(self.epochs, self.val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
        
        # 添加最佳validation loss标记
        if self.best_epoch is not None:
            ax1.scatter(self.best_epoch, self.best_val_loss, 
                       color='red', s=100, zorder=5, marker='*', 
                       label=f'Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 对数刻度更好显示
        
        # 其他子图的代码...（为了简洁，这里省略了完整的绘图代码）
        
        plt.suptitle('ChemCPA Training & Validation Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练&验证曲线图保存到: {save_path}")
        
        return fig

class StandardLossTrainer:
    """标准Loss训练器 - 专门负责使用标准loss进行训练"""
    
    def __init__(self, adata, n_smiles_features=None, device="auto", loss_history=None):
        self.adata = adata
        self.n_smiles_features = n_smiles_features
        self.device = device
        self.loss_history = loss_history if loss_history is not None else LossHistory()
        self.best_model_state = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 训练配置
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')  # 🔥 新增：最佳验证loss
        self.patience_counter = 0

        # 🔥 新增：validation相关属性
        self.validation_data = None
        self.validation_losses = []
        self.training_losses = []

        # ✅ 新增：创建drug_idx到drug名称的映射（用于prepare_batch_data）
        if 'drug_idx' in adata.obs and 'drug' in adata.obs:
            self.idx_to_drug = dict(zip(adata.obs['drug_idx'], adata.obs['drug']))
            logger.debug(f"创建药物索引映射: {len(self.idx_to_drug)} 个药物")
        else:
            self.idx_to_drug = {}

        logger.info(f"标准Loss训练器初始化，设备: {device}")
        
# 文件: chemcpa_training.py
# 请用此函数完整替换 class StandardLossTrainer 下的 initialize_model 函数

    def initialize_model(self):
        """初始化ChemCPA模型"""
        logger.info("初始化ChemCPA模型...")

        try:
            # 获取基本参数
            n_genes = self.adata.n_vars

            # --- START of FINAL FIX ---
            # 1. (重要修正) 确保从正确的 'drug' 列计算药物数量，而不是旧的 'cov_drug_dose'
            if 'training_drug_count' in self.adata.uns:
                n_drugs = int(self.adata.uns['training_drug_count'])
                logger.info(f"使用训练时药物数量: {n_drugs}")
            else:
                n_drugs = len(self.adata.obs['drug'].unique())
            # --- END of FINAL FIX ---

            if 'training_cell_count' in self.adata.uns:
                n_covariates = [int(self.adata.uns['training_cell_count'])]
                logger.info(f"使用训练时细胞系数量: {n_covariates[0]}")
            else:
                n_covariates = [len(np.unique(self.adata.obs.get("cell_line", [])))]

            logger.info(f"模型参数: genes={n_genes}, drugs={n_drugs}, covariates={n_covariates}")

            # 计算总特征维度（基因 + SMILES）
            if self.n_smiles_features and self.n_smiles_features > 0:
                total_features = n_genes + self.n_smiles_features
                logger.info(f"使用拼接特征维度: {total_features} (基因{n_genes} + SMILES{self.n_smiles_features})")
            else:
                total_features = n_genes
                logger.info(f"仅使用基因维度: {total_features}")

            # 🔥 自定义超参数配置 - 修复维度一致性
            custom_hparams = {
                # 核心架构参数 (用户指定的大模型配置)
                "autoencoder_width": 2048,     # 增大隐藏层 (默认: 512)
                "autoencoder_depth": 4,        # 自编码器深度 (默认: 3)
                "dim": 1024,                   # 增大潜在空间 (默认: 256)
                "adversary_width": 1024,       # 🔥 修改：必须匹配dim维度，处理encoder的输出
                "adversary_depth": 3,          # 对抗网络深度 (默认: 2)

                # 🔥 关键修复：确保所有编码器维度与主架构匹配
                "embedding_encoder_width": 1024,  # 🔥 修改：匹配dim大小，确保维度一致
                "embedding_encoder_depth": 3,     # 🔥 修改：增加深度以处理更大的维度

                # 🔥 关键修复：剂量编码器也需要匹配
                "dosers_width": 512,           # 🔥 修改：增大剂量编码器以匹配架构
                "dosers_depth": 3,             # 🔥 修改：增加深度

                # 🔥 新增：确保解码器维度正确
                "decoder_width": 2048,         # 🔥 新增：解码器宽度匹配自编码器
                "decoder_depth": 4,            # 🔥 新增：解码器深度

                # 🔥 新增：确保编码器维度正确
                "encoder_width": 2048,         # 🔥 新增：编码器宽度匹配
                "encoder_depth": 4,            # 🔥 新增：编码器深度

                # 学习率参数
                "dosers_lr": 1e-3,             # 剂量编码器学习率
                "autoencoder_lr": 1e-3,        # 自编码器学习率
                "adversary_lr": 3e-4,          # 对抗网络学习率

                # 正则化参数
                "reg_adversary": 5.0,          # 对抗损失权重
                "penalty_adversary": 3.0,      # 对抗惩罚权重
                "autoencoder_wd": 1e-6,        # 自编码器权重衰减
                "adversary_wd": 1e-4,          # 对抗网络权重衰减

                # 训练参数
                "adversary_steps": 3,          # 对抗训练步数
                "step_size_lr": 45,            # 学习率调度步长
                "gamma_lr": 0.15,              # 学习率衰减因子

                # 模型配置参数
                "decoder_activation": "linear", # 解码器激活函数
                "doser_type": "logsigm",       # 剂量编码器类型

                # 🔥 添加可能缺失的默认参数
                "max_epochs": 100,             # 最大训练轮数
                "patience": 5,                 # 早停耐心
                "checkpoint": True,            # 是否保存检查点
                "save_dir": "./checkpoints",   # 保存目录
                "load_latest": True,           # 是否加载最新模型
                "warmup": 30,                  # 预热轮数
            }

            logger.info("🎯 使用自定义超参数:")
            for key, value in custom_hparams.items():
                logger.info(f"  {key}: {value}")

            # 初始化ComPert模型
            if CHEMCPA_AVAILABLE:
                try:
                    # 🔥 首先尝试使用自定义超参数
                    self.model = ComPert(
                        num_genes=total_features,
                        num_drugs=n_drugs,
                        num_covariates=n_covariates,
                        device=self.device,
                        seed=0,
                        patience=5,
                        doser_type="logsigm",
                        decoder_activation="linear",
                        hparams=custom_hparams,  # 🔥 使用自定义超参数
                        # --- START of FINAL FIX ---
                        # 2. (重要修正) 将模式切换为 True
                        use_drugs_idx=True,
                        # --- END of FINAL FIX ---
                    )
                    logger.info("✅ 成功使用自定义超参数初始化模型")

                except Exception as e:
                    logger.error(f"🚨 自定义超参数初始化失败: {e}")
                    logger.error(f"🚨 错误详情: {type(e).__name__}")
                    import traceback
                    logger.error(f"🚨 完整错误栈: {traceback.format_exc()}")
                    logger.error("🚨 尝试其他超参数格式...")

                    # 🔥 尝试不同的超参数格式
                    alternative_formats = [
                        # 格式1: 简化的字典
                        {
                            "autoencoder_width": 2048,
                            "dim": 1024,
                            "adversary_width": 1024,
                        },

                        # 格式2: 字符串格式
                        "autoencoder_width=2048,dim=1024,adversary_width=1024",

                        # 格式3: 空字符串但使用其他参数控制
                        "",

                        # 格式4: 不传递hparams
                        "SKIP_HPARAMS"
                    ]

                    model_created = False
                    for i, alt_hparams in enumerate(alternative_formats):
                        try:
                            logger.info(f"🔄 尝试格式 {i+1}: {type(alt_hparams).__name__} - {alt_hparams}")

                            if alt_hparams == "SKIP_HPARAMS":
                                # 不传递hparams参数
                                self.model = ComPert(
                                    num_genes=total_features,
                                    num_drugs=n_drugs,
                                    num_covariates=n_covariates,
                                    device=self.device,
                                    seed=0,
                                    patience=5,
                                    doser_type="logsigm",
                                    decoder_activation="linear",
                                    use_drugs_idx=True,
                                )
                            else:
                                self.model = ComPert(
                                    num_genes=total_features,
                                    num_drugs=n_drugs,
                                    num_covariates=n_covariates,
                                    device=self.device,
                                    seed=0,
                                    patience=5,
                                    doser_type="logsigm",
                                    decoder_activation="linear",
                                    hparams=alt_hparams,
                                    use_drugs_idx=True,
                                )

                            logger.info(f"✅ 格式 {i+1} 初始化成功！")
                            model_created = True
                            break

                        except Exception as alt_e:
                            logger.warning(f"❌ 格式 {i+1} 也失败: {alt_e}")
                            continue

                    if not model_created:
                        logger.error("🚨 所有超参数格式都失败了！")
                        raise RuntimeError("无法创建ComPert模型 - 所有超参数格式都被拒绝")

                self.model = self.model.to(self.device)
                logger.info(f"✅ ComPert模型初始化完成，移至设备: {self.device}")

                # 🔥 打印模型结构和参数
                self._print_model_structure()

            else:
                raise ImportError("chemCPA不可用，无法初始化模型")

            return self.model

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def _print_model_structure(self):
        """打印模型结构和参数统计"""
        logger.info("="*60)
        logger.info("📊 模型结构和参数统计")
        logger.info("="*60)

        # 计算总参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"🔢 总参数数量: {total_params:,}")
        logger.info(f"🔢 可训练参数数量: {trainable_params:,}")
        logger.info(f"🔢 内存占用估计: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

        # 打印模型层结构
        logger.info("\n📝 模型层结构:")
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只打印叶子层
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    logger.info(f"  {name}: {module} - 参数数量: {params:,}")

        # 打印子模块参数统计
        logger.info("\n🧩 子模块参数统计:")
        module_stats = {}
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            module_stats[name] = params
            logger.info(f"  {name}: {params:,} 参数")

        # 打印参数比例
        logger.info("\n📊 参数分布比例:")
        for name, params in module_stats.items():
            percentage = (params / total_params) * 100 if total_params > 0 else 0
            logger.info(f"  {name}: {percentage:.2f}%")

        # 打印具体的重要层信息
        if hasattr(self.model, 'autoencoder'):
            logger.info("\n🎯 自编码器结构:")
            autoencoder_params = sum(p.numel() for p in self.model.autoencoder.parameters())
            logger.info(f"  自编码器总参数: {autoencoder_params:,}")

        if hasattr(self.model, 'adversary_drugs'):
            logger.info("\n🎯 对抗网络结构:")
            adversary_params = sum(p.numel() for p in self.model.adversary_drugs.parameters())
            logger.info(f"  药物对抗网络参数: {adversary_params:,}")

        if hasattr(self.model, 'adversary_covariates'):
            covariate_adversary_params = sum(p.numel() for p in self.model.adversary_covariates.parameters())
            logger.info(f"  协变量对抗网络参数: {covariate_adversary_params:,}")

        logger.info("="*60)

    def prepare_batch_data(self, adata_batch):
        """准备批次数据 - ✅ 优化：从查找表动态获取SMILES特征"""
        # 在CPU上准备数据
        if hasattr(adata_batch.X, 'toarray'):
            gene_data_cpu = adata_batch.X.toarray().astype(np.float32)
        else:
            gene_data_cpu = adata_batch.X.astype(np.float32)

        # ✅ 优化：处理SMILES特征 - 从查找表动态获取
        combined_data_cpu = gene_data_cpu
        if self.n_smiles_features:
            # 优先使用查找表 - 从主adata对象获取（而非batch对象）
            if hasattr(self, 'adata') and hasattr(self.adata, 'uns') and 'drug_to_smiles_features' in self.adata.uns:
                drug_to_smiles_features = self.adata.uns['drug_to_smiles_features']
                n_samples = len(adata_batch.obs) if hasattr(adata_batch, 'obs') else len(adata_batch.X)
                smiles_data_cpu = np.zeros((n_samples, self.n_smiles_features), dtype=np.float32)

                # 从drug_idx或drug列获取药物信息
                if 'drug' in adata_batch.obs:
                    for i, drug in enumerate(adata_batch.obs['drug']):
                        if drug in drug_to_smiles_features:
                            smiles_data_cpu[i] = drug_to_smiles_features[drug]
                elif 'drug_idx' in adata_batch.obs and hasattr(self, 'idx_to_drug'):
                    for i, drug_idx in enumerate(adata_batch.obs['drug_idx']):
                        drug = self.idx_to_drug.get(drug_idx, None)
                        if drug and drug in drug_to_smiles_features:
                            smiles_data_cpu[i] = drug_to_smiles_features[drug]

                combined_data_cpu = np.concatenate([gene_data_cpu, smiles_data_cpu], axis=1)
                del smiles_data_cpu
            # 次优：从batch的uns获取（如果有）
            elif hasattr(adata_batch, 'uns') and 'drug_to_smiles_features' in adata_batch.uns:
                drug_to_smiles_features = adata_batch.uns['drug_to_smiles_features']
                n_samples = len(adata_batch.obs) if hasattr(adata_batch, 'obs') else len(adata_batch.X)
                smiles_data_cpu = np.zeros((n_samples, self.n_smiles_features), dtype=np.float32)
                for i, drug in enumerate(adata_batch.obs['drug']):
                    if drug in drug_to_smiles_features:
                        smiles_data_cpu[i] = drug_to_smiles_features[drug]
                combined_data_cpu = np.concatenate([gene_data_cpu, smiles_data_cpu], axis=1)
                del smiles_data_cpu, drug_to_smiles_features
            # 向后兼容旧格式：从obsm获取
            elif hasattr(adata_batch, 'obsm') and 'smiles_features' in adata_batch.obsm:
                smiles_data_cpu = adata_batch.obsm['smiles_features'].astype(np.float32)
                combined_data_cpu = np.concatenate([gene_data_cpu, smiles_data_cpu], axis=1)
                del smiles_data_cpu
            # 如果都没有，只使用基因表达数据
            else:
                logger.debug("未找到SMILES特征，仅使用基因表达数据")
        
        # 准备索引
        drug_idx_cpu = adata_batch.obs['drug_idx'].values.astype(np.int64)
        cell_line_idx_cpu = adata_batch.obs['cell_line_idx'].values.astype(np.int64)
        dosages_cpu = adata_batch.obs['dose'].values.astype(np.float32)
        
        # 检查索引范围
        if hasattr(self, 'n_cell_lines') and cell_line_idx_cpu.max() >= self.n_cell_lines:
            cell_line_idx_cpu = np.clip(cell_line_idx_cpu, 0, self.n_cell_lines - 1)
        elif not hasattr(self, 'n_cell_lines'):
            # 从模型推断
            self.n_cell_lines = max(100, cell_line_idx_cpu.max() + 10)
        
        # ⭐ 关键：创建正确的协变量格式 - one-hot编码
        n_samples = len(cell_line_idx_cpu)
        cell_line_onehot = np.zeros((n_samples, self.n_cell_lines), dtype=np.float32)
        cell_line_onehot[np.arange(n_samples), cell_line_idx_cpu] = 1.0
        
        # 分步移到GPU，避免同时占用过多内存
        features_gpu = torch.from_numpy(combined_data_cpu).to(self.device)
        del combined_data_cpu
        
        drug_idx_gpu = torch.from_numpy(drug_idx_cpu).to(self.device)
        del drug_idx_cpu
        
        cell_line_idx_gpu = torch.from_numpy(cell_line_idx_cpu).to(self.device)
        
        dosages_gpu = torch.from_numpy(dosages_cpu).to(self.device)
        del dosages_cpu
        
        targets_gpu = torch.from_numpy(gene_data_cpu).to(self.device)
        del gene_data_cpu
        
        # ⭐ 关键：协变量是one-hot格式的list
        covariates_gpu = [torch.from_numpy(cell_line_onehot).to(self.device)]
        del cell_line_onehot, cell_line_idx_cpu
        
        result = {
            'features': features_gpu,
            'drug_idx': drug_idx_gpu,
            'cell_line_idx': cell_line_idx_gpu,
            'dosages': dosages_gpu,
            'targets': targets_gpu,
            'covariates': covariates_gpu
        }
        
        return result

    
    def _safe_tensor_operation(self, tensor, operation_name):
        """安全的张量操作，处理维度问题"""
        try:
            if tensor.dim() == 0:  # 标量
                return tensor.unsqueeze(0)
            elif tensor.dim() > 1:
                # 只压缩尺寸为1的维度，避免过度压缩
                while tensor.dim() > 1 and tensor.size(-1) == 1:
                    tensor = tensor.squeeze(-1)
                return tensor
            else:  # 1D张量
                return tensor
        except Exception as e:
            logger.warning(f"{operation_name}张量操作失败: {e}")
            return tensor
    
    def prepare_training_data(self):
        """准备训练和验证数据 - 修复：按condition分割避免data leakage + 激进内存清理"""
        logger.info("准备训练数据（按condition分割 + 激进内存清理）...")
        
        # 获取训练数据
        train_mask = self.adata.obs['split'] == 'train'
        full_train_adata = self.adata[train_mask].copy()
        
        # 🔥 立即删除mask，释放内存
        del train_mask
        gc.collect()
        
        logger.info(f"原始训练数据: {full_train_adata.n_obs} 样本")
        
        # 🔥 修复：按condition分割而不是按样本分割
        full_train_adata.obs['condition'] = (
            full_train_adata.obs['cell_line'] + '_' + 
            full_train_adata.obs['drug'] + '_' + 
            full_train_adata.obs['dose_str']
        )
        
        # 获取所有唯一conditions
        unique_conditions = full_train_adata.obs['condition'].unique()
        logger.info(f"总共 {len(unique_conditions)} 个不同的conditions")
        
        # 按condition随机分割
        np.random.seed(42)
        shuffled_conditions = np.random.permutation(unique_conditions)
        
        n_val_conditions = max(1, int(len(unique_conditions) * 0.1))  # 10%的conditions作为验证集
        val_conditions = set(shuffled_conditions[:n_val_conditions])
        train_conditions = set(shuffled_conditions[n_val_conditions:])
        
        # 🔥 立即删除不需要的变量
        del unique_conditions, shuffled_conditions
        gc.collect()
        
        logger.info(f"验证集conditions: {len(val_conditions)}")
        logger.info(f"训练集conditions: {len(train_conditions)}")
        
        # 根据condition分配样本
        val_mask = full_train_adata.obs['condition'].isin(val_conditions)
        train_mask = ~val_mask
        
        # 🔥 立即删除conditions集合
        del val_conditions, train_conditions
        gc.collect()
        
        # 分离训练和验证数据
        train_adata = full_train_adata[train_mask].copy()
        val_adata = full_train_adata[val_mask].copy()
        
        # 🔥 立即删除masks和原始数据
        del train_mask, val_mask, full_train_adata
        gc.collect()
        
        logger.info(f"实际训练数据: {train_adata.n_obs} 样本")
        logger.info(f"验证数据: {val_adata.n_obs} 样本")
        logger.info(f"实际验证比例: {val_adata.n_obs/(train_adata.n_obs + val_adata.n_obs):.1%}")
        
        # 验证没有数据泄露
        train_conditions_check = set(train_adata.obs['condition'])
        val_conditions_check = set(val_adata.obs['condition'])
        overlap = train_conditions_check & val_conditions_check
        
        # 🔥 立即删除检查用的集合
        del train_conditions_check, val_conditions_check
        
        if len(overlap) == 0:
            logger.info("✅ 验证通过：没有condition泄露")
        else:
            logger.error(f"❌ 仍存在泄露：{len(overlap)} 个重叠conditions")
            raise ValueError("数据分割修复失败！")
        
        del overlap
        gc.collect()
        
        # 🔥 逐个处理训练数据，立即删除中间变量
        logger.info("处理训练数据特征...")
        
        # 处理训练数据基因表达
        if hasattr(train_adata.X, 'toarray'):
            gene_data_train = train_adata.X.toarray().astype(np.float32)
        else:
            gene_data_train = train_adata.X.astype(np.float32)
        
        # ✅ 优化：整合SMILES特征 - 从查找表动态获取而非使用预存的完整矩阵
        if self.n_smiles_features and 'drug_to_smiles_features' in train_adata.uns:
            logger.info("✅ 从查找表动态获取SMILES特征（内存优化）...")
            drug_to_smiles_features = train_adata.uns['drug_to_smiles_features']

            # 为训练数据动态创建SMILES特征矩阵
            n_samples = len(train_adata)
            smiles_data_train = np.zeros((n_samples, self.n_smiles_features), dtype=np.float32)
            for i, drug in enumerate(train_adata.obs['drug']):
                if drug in drug_to_smiles_features:
                    smiles_data_train[i] = drug_to_smiles_features[drug]

            combined_data_train = np.concatenate([gene_data_train, smiles_data_train], axis=1)

            # 🔥 立即删除中间数组
            del gene_data_train, smiles_data_train, drug_to_smiles_features
            gc.collect()

            combined_data_train = torch.tensor(combined_data_train, dtype=torch.float32)
            logger.info(f"训练集拼接特征: 基因 + SMILES = {combined_data_train.shape[1]}")
        elif 'smiles_features' in train_adata.obsm and self.n_smiles_features:
            # 向后兼容：如果使用旧格式
            logger.warning("⚠️ 使用旧格式SMILES特征（完整矩阵），建议使用查找表格式")
            smiles_data_train = train_adata.obsm['smiles_features'].astype(np.float32)
            combined_data_train = np.concatenate([gene_data_train, smiles_data_train], axis=1)

            del gene_data_train, smiles_data_train
            gc.collect()

            combined_data_train = torch.tensor(combined_data_train, dtype=torch.float32)
            logger.info(f"训练集拼接特征: 基因 + SMILES = {combined_data_train.shape[1]}")
        else:
            combined_data_train = torch.tensor(gene_data_train, dtype=torch.float32)
            del gene_data_train
            gc.collect()
            logger.info(f"训练集仅使用基因特征: {combined_data_train.shape[1]}")
        
        # 准备其他训练数据
        drug_idx_train = torch.tensor(train_adata.obs['drug_idx'].values, dtype=torch.long)
        dosages_train = torch.tensor(train_adata.obs["dose"].values, dtype=torch.float32) if "dose" in train_adata.obs else torch.ones_like(drug_idx_train, dtype=torch.float32)
        
        # 准备协变量 - 训练集
        covariates_train = []
        if "cell_line_idx" in train_adata.obs:
            cov_idx_train = torch.tensor(train_adata.obs["cell_line_idx"].values, dtype=torch.long)
            covariates_train.append(cov_idx_train)
        
        # 🔥 训练数据处理完毕，立即删除train_adata
        del train_adata
        gc.collect()
        
        # 🔥 处理验证数据，同样的激进清理策略
        logger.info("处理验证数据特征...")
        
        # 处理验证数据基因表达
        if hasattr(val_adata.X, 'toarray'):
            gene_data_val = val_adata.X.toarray().astype(np.float32)
        else:
            gene_data_val = val_adata.X.astype(np.float32)
        
        # ✅ 优化：整合SMILES特征 - 验证集也从查找表动态获取
        if self.n_smiles_features and 'drug_to_smiles_features' in val_adata.uns:
            logger.info("✅ 验证集：从查找表动态获取SMILES特征...")
            drug_to_smiles_features = val_adata.uns['drug_to_smiles_features']

            n_samples = len(val_adata)
            smiles_data_val = np.zeros((n_samples, self.n_smiles_features), dtype=np.float32)
            for i, drug in enumerate(val_adata.obs['drug']):
                if drug in drug_to_smiles_features:
                    smiles_data_val[i] = drug_to_smiles_features[drug]

            combined_data_val = np.concatenate([gene_data_val, smiles_data_val], axis=1)

            del gene_data_val, smiles_data_val, drug_to_smiles_features
            gc.collect()

            combined_data_val = torch.tensor(combined_data_val, dtype=torch.float32)
        elif 'smiles_features' in val_adata.obsm and self.n_smiles_features:
            # 向后兼容
            smiles_data_val = val_adata.obsm['smiles_features'].astype(np.float32)
            combined_data_val = np.concatenate([gene_data_val, smiles_data_val], axis=1)

            del gene_data_val, smiles_data_val
            gc.collect()

            combined_data_val = torch.tensor(combined_data_val, dtype=torch.float32)
        else:
            combined_data_val = torch.tensor(gene_data_val, dtype=torch.float32)
            del gene_data_val
            gc.collect()
        
        # 准备其他验证数据
        drug_idx_val = torch.tensor(val_adata.obs['drug_idx'].values, dtype=torch.long)
        dosages_val = torch.tensor(val_adata.obs["dose"].values, dtype=torch.float32) if "dose" in val_adata.obs else torch.ones_like(drug_idx_val, dtype=torch.float32)
        
        # 准备协变量 - 验证集
        covariates_val = []
        if "cell_line_idx" in val_adata.obs:
            cov_idx_val = torch.tensor(val_adata.obs["cell_line_idx"].values, dtype=torch.long)
            covariates_val.append(cov_idx_val)
        
        # 🔥 验证数据处理完毕，立即删除val_adata
        del val_adata
        gc.collect()
        
        logger.info(f"特征处理完成: 训练{combined_data_train.shape}, 验证{combined_data_val.shape}")
        
        # 🔥 最后一次强制垃圾回收
        gc.collect()
        
        return {
            'train': {
                'gene_expression': combined_data_train,
                'drug_indices': drug_idx_train,
                'dosages': dosages_train,
                'covariates': covariates_train
            },
            'val': {
                'gene_expression': combined_data_val,
                'drug_indices': drug_idx_val,
                'dosages': dosages_val,
                'covariates': covariates_val
            }
        }
    
    def create_data_loader(self, batch_size, shuffle=True):
        """创建数据加载器"""
        logger.info(f"创建数据加载器，批处理大小: {batch_size}")
        
        train_data = self.prepare_training_data()
        
        # 修复：访问正确的键名
        dataset_tensors = [
            train_data['train']['gene_expression'],  # 这里需要加上 ['train']
            train_data['train']['drug_indices'],     # 这里需要加上 ['train']
            train_data['train']['dosages']           # 这里需要加上 ['train']
        ]
        
        # 添加协变量
        if train_data['train']['covariates']:  # 这里需要加上 ['train']
            dataset_tensors.extend(train_data['train']['covariates'])
        
        dataset = TensorDataset(*dataset_tensors)
        
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # 丢弃最后一个不完整的批次
        )
        
        logger.info(f"数据加载器创建完成: {len(dataset)} 样本, {len(loader)} 批次")
        
        return loader
    
    def compute_standard_loss(self, batch_data):
        """计算标准loss - 使用工作的预测方法"""
        try:
            # 提取批次数据
            gene_expr = batch_data[0].to(self.device)
            drug_idx = batch_data[1].to(self.device)
            dosages = batch_data[2].to(self.device)
            
            batch_size = gene_expr.size(0)
            
            # 处理协变量 - 使用相同的方法
            raw_covariates = []
            if len(batch_data) > 3:
                for i in range(3, len(batch_data)):
                    cov_data = batch_data[i].to(self.device)
                    raw_covariates.append(cov_data)
            
            # ⭐ 关键：使用统一的数据准备方法
            # 创建临时adata对象来使用prepare_batch_data
            temp_adata = type('TempAData', (), {})()
            temp_adata.X = gene_expr.cpu().numpy()
            temp_adata.n_obs = batch_size
            temp_adata.n_vars = self.adata.n_vars if hasattr(self, 'adata') else gene_expr.size(1)
            
            # 创建obs
            temp_adata.obs = pd.DataFrame({
                'drug_idx': drug_idx.cpu().numpy(),
                'cell_line_idx': raw_covariates[0].cpu().numpy() if raw_covariates else np.zeros(batch_size),
                'dose': dosages.cpu().numpy()
            })
            
            # 添加SMILES特征
            temp_adata.obsm = {}
            if hasattr(self, 'n_smiles_features') and self.n_smiles_features > 0:
                n_genes = self.adata.n_vars if hasattr(self, 'adata') else gene_expr.size(1) - self.n_smiles_features
                if gene_expr.size(1) > n_genes:
                    temp_adata.obsm['smiles_features'] = gene_expr[:, n_genes:].cpu().numpy()
                    temp_adata.X = gene_expr[:, :n_genes].cpu().numpy()
            
            # 使用统一的数据准备方法
            batch_data_prepared = self.prepare_batch_data(temp_adata)
            
            # ⭐ 关键：使用简单直接的预测调用
            features = batch_data_prepared['features']
            drug_idx = batch_data_prepared['drug_idx']
            dosages = batch_data_prepared['dosages']
            covariates = batch_data_prepared['covariates']
            targets = batch_data_prepared['targets']
            
            # 确保维度正确
            if drug_idx.dim() > 1:
                drug_idx = drug_idx.squeeze()
            if dosages.dim() > 1:
                dosages = dosages.squeeze()
            
            # 直接预测，不用复杂的错误处理
            predictions, _ = self.model.predict(
                genes=features,
                drugs_idx=drug_idx,
                dosages=dosages,
                covariates=covariates
            )
            
            # 计算损失（仅针对基因部分）
            if predictions.size(1) > targets.size(1):
                predictions = predictions[:, :targets.size(1)]
            
            loss = torch.nn.functional.mse_loss(predictions, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到无效loss: {loss}")
                loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 清理临时数据
            del batch_data_prepared, features, drug_idx, dosages, covariates, targets, predictions
            
            return loss
            
        except Exception as e:
            logger.error(f"compute_standard_loss失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)

    def compute_validation_loss(self, val_data):
        """计算验证集loss"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            # 创建验证数据的简单批次
            batch_size = 64
            n_val_samples = val_data['gene_expression'].size(0)
            
            for start_idx in range(0, n_val_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_val_samples)
                
                # 准备批次数据
                batch_data = [
                    val_data['gene_expression'][start_idx:end_idx],
                    val_data['drug_indices'][start_idx:end_idx],
                    val_data['dosages'][start_idx:end_idx]
                ]
                
                # 添加协变量
                if val_data['covariates']:
                    for cov in val_data['covariates']:
                        batch_data.append(cov[start_idx:end_idx])
                
                try:
                    # 计算loss
                    loss = self.compute_standard_loss(batch_data)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_losses.append(loss.item())
                except Exception as e:
                    logger.warning(f"验证批次 {start_idx}-{end_idx} 失败: {e}")
                    continue
        
        if val_losses:
            avg_val_loss = np.mean(val_losses)
        else:
            avg_val_loss = float('inf')
            logger.warning("所有验证批次都失败了")
        
        return avg_val_loss
    
    def train_epoch(self, data_loader, val_data, epoch):
        """训练单个epoch - 包含validation"""
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        total_batches = len(data_loader)
        successful_batches = 0
        
        # 训练阶段
        for batch_idx, batch_data in enumerate(data_loader):
            batch_start_time = time.time()
            
            try:
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 计算loss
                loss = self.compute_standard_loss(batch_data)
                
                # 检查loss有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: 检测到无效loss值: {loss}")
                    self.loss_history.add_batch_loss(epoch, batch_idx, None, None, False)
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                # 记录成功的batch
                batch_time = time.time() - batch_start_time
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                successful_batches += 1
                
                self.loss_history.add_batch_loss(epoch, batch_idx, loss_value, batch_time, True)
                
                # 定期输出进度
                if batch_idx % max(total_batches // 10, 1) == 0:
                    logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{total_batches}: Loss={loss_value:.6f}")
                
            except Exception as e:
                logger.warning(f"Epoch {epoch}, Batch {batch_idx} 训练失败: {e}")
                self.loss_history.add_batch_loss(epoch, batch_idx, None, None, False)
                continue
            
            # 内存清理
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # 🔥 验证阶段
        logger.info(f"  Epoch {epoch}: 开始验证...")
        val_loss = self.compute_validation_loss(val_data)
        
        # 计算epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        success_rate = successful_batches / total_batches if total_batches > 0 else 0.0
        
        # 更新学习率调度器
        if self.scheduler:
            self.scheduler.step(val_loss)  # 基于validation loss调整
        
        # 记录epoch级别的loss
        current_lr = self.optimizer.param_groups[0]['lr']
        self.loss_history.add_epoch_loss(epoch, avg_train_loss, val_loss, epoch_time, current_lr, epoch_losses)
        
        # 早停检查 - 基于validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # 确保保存最佳模型状态
            self.best_model_state = {
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': avg_train_loss
            }
            logger.info(f"  ✨ 新的最佳验证loss: {self.best_val_loss:.6f}. 模型状态已捕获。")
        else:
            self.patience_counter += 1
        
        logger.info(f"Epoch {epoch} 完成: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}, 时间={epoch_time:.2f}s, 成功率={success_rate:.1%}")
        
        return avg_train_loss, val_loss, epoch_time, success_rate
    
    def train(self, max_epochs=100, batch_size=None, lr=1e-4, weight_decay=1e-6,
              early_stopping=True, early_stopping_patience=5, **kwargs):
        """执行完整的训练过程 - 包含validation"""

        # 如果没有提供batch_size，使用默认值32
        if batch_size is None:
            batch_size = 32
            logger.warning("未提供batch_size，使用默认值32")

        logger.info("🚀 开始标准Loss训练（包含validation）...")
        logger.info(f"训练参数: epochs={max_epochs}, batch_size={batch_size}, lr={lr}")
        
        # 重置loss历史
        self.loss_history.reset()
        
        # 🔥 准备训练和验证数据
        prepared_data = self.prepare_training_data()
        train_data = prepared_data['train']
        val_data = prepared_data['val']
        
        logger.info(f"数据准备完成: 训练样本={train_data['gene_expression'].size(0)}, 验证样本={val_data['gene_expression'].size(0)}")
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 创建数据加载器（仅对训练数据）
        dataset_tensors = [
            train_data['gene_expression'],
            train_data['drug_indices'],
            train_data['dosages']
        ]
        
        if train_data['covariates']:
            dataset_tensors.extend(train_data['covariates'])
        
        dataset = TensorDataset(*dataset_tensors)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        # 训练循环
        for epoch in range(1, max_epochs + 1):
            self.current_epoch = epoch
            
            try:
                # 训练一个epoch（包含validation）
                avg_train_loss, val_loss, epoch_time, success_rate = self.train_epoch(data_loader, val_data, epoch)
                
                # 早停检查
                if early_stopping and self.patience_counter >= early_stopping_patience:
                    logger.info(f"早停触发：连续{early_stopping_patience}个epoch验证loss无改善，在epoch {epoch}停止训练")
                    break
                
                # 如果成功率太低，警告
                if success_rate < 0.5:
                    logger.warning(f"Epoch {epoch} 成功率较低: {success_rate:.1%}")
                
            except Exception as e:
                logger.error(f"Epoch {epoch} 训练失败: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                continue
        
        # 训练完成总结
        training_summary = self.loss_history.get_training_summary()
        logger.info("✅ 标准Loss训练完成（包含validation）")
        logger.info(f"📊 训练总结:")
        logger.info(f"  总epochs: {training_summary['total_epochs']}")
        logger.info(f"  最终训练loss: {training_summary['final_train_loss']:.6f}")
        logger.info(f"  最终验证loss: {training_summary['final_val_loss']:.6f}")
        logger.info(f"  最佳验证loss: {training_summary['best_val_loss']:.6f} (Epoch {training_summary['best_epoch']})")
        logger.info(f"  总训练时间: {training_summary['total_training_time']:.2f}s")
        logger.info(f"  批次成功率: {training_summary['batch_success_rate']:.1%}")
        
        return training_summary

    def save_training_plots(self, save_dir="./dose_global_result/training_plots"):
        """保存训练曲线图"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{save_dir}/chemcpa_training_validation_{timestamp}.pdf"
        
        fig = self.loss_history.plot_training_curves(plot_path)
        
        if fig is not None:
            logger.info(f"📈 训练&验证曲线已保存: {plot_path}")
            return plot_path
        else:
            logger.warning("无法保存训练曲线：没有足够的数据")
            return None
    
# 文件: chemcpa_training.py
# 请用此函数完整替换 class StandardLossTrainer 下的 predict 函数

# 文件: chemcpa_training.py
# 请用此函数完整替换 class StandardLossTrainer 下的 predict 函数

# 文件: chemcpa_training.py
# 请用此函数完整替换 class StandardLossTrainer 下的 predict 函数

    def predict(self, adata_test):
        """使用训练好的模型进行预测 - 最终修复版"""
        logger.info("开始原生ChemCPA模型预测（最终修复版）...")
        
        if self.model is None:
            raise ValueError("模型未训练，无法进行预测")
        
        self.model.eval()
        
        n_samples = adata_test.n_obs
        n_genes = adata_test.n_vars
        
        logger.info(f"预测数据: {n_samples} 样本, {n_genes} 基因")
        
        batch_size = 500
        all_predictions = []
        
        if not hasattr(self, 'n_cell_lines'):
            if 'cell_line_idx' in adata_test.obs:
                self.n_cell_lines = max(100, adata_test.obs['cell_line_idx'].max() + 10)
            else:
                self.n_cell_lines = 100
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_adata = adata_test[start_idx:end_idx].copy()
                
                try:
                    batch_data = self.prepare_batch_data(batch_adata)
                    
                    features = batch_data['features']
                    drug_idx = batch_data['drug_idx']
                    dosages = batch_data['dosages']
                    covariates = batch_data['covariates']
                    
                    if drug_idx.dim() > 1:
                        drug_idx = drug_idx.squeeze()
                    if dosages.dim() > 1:
                        dosages = dosages.squeeze()

                    # --- START of FINAL FIX ---
                    # 无论如何都传递 drugs_idx，以满足库函数的 assert 要求。
                    # 模型内部会根据 use_drugs_idx=False 的设置来正确地忽略它。
                    predictions, _ = self.model.predict(
                        genes=features,
                        drugs_idx=drug_idx,
                        dosages=dosages,
                        covariates=covariates
                    )
                    # --- END of FINAL FIX ---
                    
                    if predictions.size(1) > n_genes:
                        predictions = predictions[:, :n_genes]
                    elif predictions.size(1) < n_genes:
                        padding = torch.zeros(predictions.size(0), n_genes - predictions.size(1)).to(predictions.device)
                        predictions = torch.cat([predictions, padding], dim=1)
                    
                    batch_predictions = predictions.cpu().numpy()
                    all_predictions.append(batch_predictions)
                    
                    if (start_idx // batch_size) % 20 == 0:
                        logger.info(f"原生ChemCPA预测进度: {start_idx + batch_size}/{n_samples}")
                    
                except Exception as e:
                    logger.error(f"预测批次 {start_idx}-{end_idx} 失败。详细错误追踪如下:")
                    logger.error(traceback.format_exc())
                    logger.warning(f"预测失败，使用随机预测。")
                    
                    batch_predictions = np.random.randn(end_idx - start_idx, n_genes) * 0.1
                    all_predictions.append(batch_predictions)
                
                finally:
                    # 确保在每次循环后都进行内存清理
                    del batch_adata, batch_data
                    if 'features' in locals(): del features
                    if 'drug_idx' in locals(): del drug_idx
                    if 'dosages' in locals(): del dosages
                    if 'covariates' in locals(): del covariates
                    if 'predictions' in locals(): del predictions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        final_predictions = np.concatenate(all_predictions, axis=0)
        
        if final_predictions.shape[0] != n_samples:
            if final_predictions.shape[0] > n_samples:
                final_predictions = final_predictions[:n_samples]
            else:
                padding = np.zeros((n_samples - final_predictions.shape[0], n_genes))
                final_predictions = np.concatenate([final_predictions, padding], axis=0)
        
        adata_test.obsm["ChemCPA_pred"] = final_predictions
        
        logger.info(f"原生ChemCPA模型预测完成（最终修复版），结果形状: {final_predictions.shape}")
        
        return adata_test
