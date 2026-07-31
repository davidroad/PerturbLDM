#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChemCPA Configuration Module
Contains all configuration parameters for ChemCPA training and evaluation
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


def _required_environment_path(variable_name: str) -> Path:
    """Resolve a required input path without embedding a host-specific root."""
    value = os.environ.get(variable_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {variable_name}. "
            "See method_packages/PORTABILITY.md for the input contract."
        )
    return Path(value).expanduser().resolve()


def resolve_benchmark_data_file(filename: str) -> str:
    """Resolve and validate one benchmark AnnData input."""
    path = _required_environment_path("PERTURBLDM_BENCHMARK_DATA_DIR") / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark input does not exist: {path}. "
            "Set PERTURBLDM_BENCHMARK_DATA_DIR to the directory containing "
            "the prepared benchmark H5AD files."
        )
    return str(path)


def resolve_drug_metadata_file() -> str:
    """Resolve and validate the chemCPA drug-metadata CSV input."""
    path = _required_environment_path("PERTURBLDM_CHEMCPA_DRUG_METADATA")
    if not path.is_file():
        raise FileNotFoundError(
            f"Required chemCPA drug metadata does not exist: {path}. "
            "Set PERTURBLDM_CHEMCPA_DRUG_METADATA to the metadata CSV."
        )
    return str(path)


@dataclass
class DataConfig:
    """数据相关配置"""
    # Paths may be supplied explicitly; otherwise they resolve from the
    # environment variables documented in method_packages/PORTABILITY.md.
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    control_data_path: Optional[str] = None
    drug_metadata_path: Optional[str] = None
    
    # 数据预处理参数
    required_obs_columns: List[str] = None
    
    def __post_init__(self):
        if self.train_data_path is None:
            self.train_data_path = resolve_benchmark_data_file("train_adata_processed.h5ad")
        if self.test_data_path is None:
            self.test_data_path = resolve_benchmark_data_file("test_adata_processed.h5ad")
        if self.control_data_path is None:
            self.control_data_path = resolve_benchmark_data_file("control_adata_processed.h5ad")
        if self.drug_metadata_path is None:
            self.drug_metadata_path = resolve_drug_metadata_file()
        if self.required_obs_columns is None:
            self.required_obs_columns = ["cell_line", "drug", "dose"]


@dataclass
class SMILESConfig:
    """SMILES编码配置"""
    # 编码方法: 'morgan', 'rdkit', 'combined'
    encoding_method: str = "combined"
#https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4
#https://pubs.acs.org/doi/full/10.1021/acs.jpcb.4c01875
#https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0109-z    
    # Morgan指纹参数
    morgan_n_bits: int = 1024
    morgan_radius: int = 2
    
    # RDKit描述符参数
    rdkit_n_descriptors: int = 300
    
    # 标准化参数
    standardize_features: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型类型
    model_type: str = "chemcpa_with_smiles"  # 'chemcpa_with_smiles', 'native_chemcpa', 'custom_chemcpa'

    # 设备配置
    device: str = "auto"  # 'cpu', 'cuda', 'auto'

    # ChemCPA特定参数
    doser_type: str = "logsigm"  # 'logsigm', 'linear'
    decoder_activation: str = "linear"
    use_drugs_idx: bool = False

    # 🔥 新增：模型架构参数 - 支持自定义超参数（默认为大模型配置）
    # 核心维度参数
    dim: int = 1024  # latent space维度（使用大模型配置）
    autoencoder_width: int = 2048  # encoder/decoder hidden层宽度
    autoencoder_depth: int = 4  # encoder/decoder深度

    # 对抗网络参数
    adversary_width: int = 1024  # adversarial网络宽度（必须匹配dim）
    adversary_depth: int = 3  # adversarial网络深度

    # 其他组件参数
    embedding_encoder_width: int = 1024  # embedding encoder宽度（必须匹配dim）
    dosers_width: int = 512  # 剂量编码器宽度
    dosers_depth: int = 2  # 剂量编码器深度
    decoder_width: int = 2048  # 解码器宽度
    encoder_width: int = 2048  # 编码器宽度

    # 数据设置参数
    perturbation_key: str = "drug"
    dosage_key: str = "dose"
    categorical_covariate_keys: List[str] = None
    
    def __post_init__(self):
        if self.categorical_covariate_keys is None:
            self.categorical_covariate_keys = ["cell_line"]


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    max_epochs: int = 5
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # 早停参数
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # 验证参数
    check_val_every_n_epoch: int = 1
    validation_split: float = 0.1
    
    # 检查点参数
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./dose_global_result/checkpoints"
    
    # 随机种子
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估方法
    evaluation_method: str = "condition_averaged"  # 'condition_averaged', 'cell_level'
    
    # 评估指标
    metrics: List[str] = None
    
    # 质量控制参数
    min_prediction_variance: float = 1e-8
    filter_low_quality: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['MSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_r', 'Chatterjee']


@dataclass
class OutputConfig:
    """输出配置"""
    # 输出目录
    base_output_dir: str = "./dose_global_result"
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    analysis_dir: str = "analysis"
    
    # 文件名前缀
    file_prefix: str = "chemcpa_global"
    
    # 保存选项
    save_detailed_results: bool = True
    save_condition_details: bool = True
    save_model: bool = True
    save_predictions: bool = False
    
    # 输出格式
    results_format: str = "csv"  # 'csv', 'json', 'both'


@dataclass
class SystemConfig:
    """系统配置"""
    # 线程配置
    n_threads: int = 90
    n_interop_threads: int = 30
    
    # 内存配置
    memory_limit_gb: Optional[float] = None
    
    # 日志配置
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # 环境变量
    env_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_variables is None:
            self.env_variables = {
                "OMP_NUM_THREADS": str(self.n_threads),
                "OPENBLAS_NUM_THREADS": str(self.n_threads),
                "MKL_NUM_THREADS": str(self.n_threads),
                "NUMEXPR_NUM_THREADS": str(self.n_threads)
            }


@dataclass
class ChemCPAConfig:
    """完整的ChemCPA配置"""
    data: DataConfig = None
    smiles: SMILESConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    output: OutputConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.smiles is None:
            self.smiles = SMILESConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def apply_system_config(self):
        """应用系统配置"""
        import torch
        
        # 设置环境变量
        for key, value in self.system.env_variables.items():
            os.environ[key] = value
        
        # 设置torch线程数
        torch.set_num_threads(self.system.n_threads)
        torch.set_num_interop_threads(self.system.n_interop_threads)
    
    def get_output_paths(self) -> Dict[str, str]:
        """获取所有输出路径"""
        base_dir = self.output.base_output_dir
        
        paths = {
            "base": base_dir,
            "models": os.path.join(base_dir, self.output.models_dir),
            "results": os.path.join(base_dir, self.output.results_dir),
            "logs": os.path.join(base_dir, self.output.logs_dir),
            "analysis": os.path.join(base_dir, self.output.analysis_dir),
        }
        
        # 创建目录
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        
        return paths
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        import dataclasses
        
        def asdict_custom(obj):
            if dataclasses.is_dataclass(obj):
                return {f.name: asdict_custom(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            elif isinstance(obj, list):
                return [asdict_custom(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: asdict_custom(value) for key, value in obj.items()}
            else:
                return obj
        
        return asdict_custom(self)
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: str):
        """从文件加载配置"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 重建配置对象
        config = cls()
        
        # 更新配置
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'smiles' in config_dict:
            config.smiles = SMILESConfig(**config_dict['smiles'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        if 'output' in config_dict:
            config.output = OutputConfig(**config_dict['output'])
        if 'system' in config_dict:
            config.system = SystemConfig(**config_dict['system'])
        
        return config


# 默认配置实例
DEFAULT_CONFIG = ChemCPAConfig()

# 预定义配置模板
def get_fast_config() -> ChemCPAConfig:
    """快速训练配置（用于测试）"""
    config = ChemCPAConfig()
    config.training.max_epochs = 10
    config.training.batch_size = 1024  # 小批次快速测试
    config.training.early_stopping_patience = 3
    config.smiles.morgan_n_bits = 512
    return config


def get_production_config() -> ChemCPAConfig:
    """生产环境配置"""
    config = ChemCPAConfig()
    config.training.max_epochs = 300
    config.training.batch_size = 262144
    config.training.early_stopping_patience = 25
    config.smiles.morgan_n_bits = 2048
    config.system.n_threads = 120
    return config


def get_memory_efficient_config() -> ChemCPAConfig:
    """内存优化配置 - 适合GPU内存有限的情况"""
    config = ChemCPAConfig()
    config.training.batch_size = 512  # 大幅减小批次大小以节省内存
    config.training.max_epochs = 50   # 适度减少epochs
    config.smiles.morgan_n_bits = 512 # 减少SMILES特征维度
    config.smiles.rdkit_n_descriptors = 100
    config.system.memory_limit_gb = 32.0
    config.output.save_predictions = False
    return config


def get_high_accuracy_config() -> ChemCPAConfig:
    """高精度配置"""
    config = ChemCPAConfig()
    config.training.max_epochs = 500
    config.training.batch_size = 4096  # 中等批次大小平衡精度和效率
    config.training.learning_rate = 5e-4
    config.training.early_stopping_patience = 50
    config.smiles.morgan_n_bits = 2048
    config.smiles.morgan_radius = 3
    config.smiles.rdkit_n_descriptors = 300
    return config


def get_large_batch_config() -> ChemCPAConfig:
    """大批次配置 - 适合内存充足的环境"""
    config = ChemCPAConfig()
    config.training.batch_size = 65536  # 大批次大小，适合大内存GPU
    config.training.max_epochs = 100
    config.training.learning_rate = 2e-3  # 大批次可以使用稍高的学习率
    config.training.early_stopping_patience = 20
    config.smiles.morgan_n_bits = 2048
    config.system.n_threads = 128
    return config


def get_small_batch_config() -> ChemCPAConfig:
    """小批次配置 - 适合内存有限或精细训练"""
    config = ChemCPAConfig()
    config.training.batch_size = 64     # 小批次大小
    config.training.max_epochs = 200    # 增加epochs补偿小批次
    config.training.learning_rate = 5e-4  # 小批次使用较小学习率
    config.training.early_stopping_patience = 30
    config.smiles.morgan_n_bits = 1024
    return config


# 配置验证函数
def validate_config(config: ChemCPAConfig) -> Dict[str, List[str]]:
    """验证配置的有效性"""
    warnings = []
    errors = []

    # 跳过路径验证 - 只记录警告
    if not os.path.exists(config.data.train_data_path):
        warnings.append(f"训练数据路径不存在: {config.data.train_data_path}")

    if not os.path.exists(config.data.drug_metadata_path):
        warnings.append(f"药物元数据路径不存在: {config.data.drug_metadata_path}")

    # 验证训练参数
    if config.training.batch_size <= 0:
        errors.append("batch_size必须大于0")

    if config.training.learning_rate <= 0:
        errors.append("learning_rate必须大于0")

    if config.training.max_epochs <= 0:
        errors.append("max_epochs必须大于0")

    # 验证SMILES参数
    if config.smiles.morgan_n_bits <= 0:
        errors.append("morgan_n_bits必须大于0")

    if config.smiles.morgan_radius < 0:
        errors.append("morgan_radius不能为负数")

    # 验证系统参数
    if config.system.n_threads <= 0:
        warnings.append("n_threads应该大于0")

    # 验证内存限制
    if config.system.memory_limit_gb is not None and config.system.memory_limit_gb <= 0:
        warnings.append("memory_limit_gb应该大于0或为None")

    # 验证批次大小与内存的关系
    if config.training.batch_size > 1000000:
        warnings.append("批次大小过大，可能导致内存不足")

    return {
        "warnings": warnings,
        "errors": errors
    }
