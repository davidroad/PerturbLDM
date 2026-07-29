import scanpy as sc
import numpy as np
import pandas as pd
import gc
import os
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr, rankdata

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

# -----------------------
# Utility: Memory Monitoring
# -----------------------
def print_mem(stage=""):
    import psutil
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[{stage}] Memory usage: {mem_gb:.2f} GB")

def save_complete_baseline_predictions_simple(adata_test, result, output_dir):
    """
    简单保存baseline预测结果 - 跟CPA/scGen一样的逻辑
    直接从result中获取预测数据并保存
    
    Parameters:
    -----------
    adata_test : AnnData
        原始测试数据
    result : dict
        pipeline返回的结果字典
    output_dir : str
        输出目录路径
    
    Returns:
    --------
    adata_pred_dict : dict
        包含各模型预测结果的字典
    """
    print("💾 保存baseline完整预测结果...")
    
    try:
        import datetime
        import json
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否有预测数据
        if 'prediction_data' not in result:
            print("❌ 未找到预测数据，请先修改pipeline函数")
            return None
        
        pred_data = result['prediction_data']
        test_condition_info = pred_data['test_condition_info']
        baseline_mean = pred_data['baseline_mean']
        
        # 模型预测结果
        model_predictions = {
            'MLP': pred_data['y_pred_mlp'],
            'RF': pred_data['y_pred_rf'], 
            'TrivialZero': pred_data['y_pred_trivial_zero']
        }
        
        saved_models = {}
        
        for model_name, y_pred_delta in model_predictions.items():
            print(f"  处理 {model_name} 模型...")
            
            # 将delta转换为绝对表达值
            n_conditions = len(test_condition_info)
            n_genes = y_pred_delta.shape[1]
            y_pred_absolute = np.zeros((n_conditions, n_genes), dtype=np.float32)
            
            for i in range(n_conditions):
                cell_line = test_condition_info.iloc[i]['cell_line']
                ctrl_mean_vec = baseline_mean[cell_line]
                y_pred_absolute[i] = ctrl_mean_vec + y_pred_delta[i]
            
            # 扩展到每个细胞
            all_predictions = []
            all_metadata = []
            
            for i, row in test_condition_info.iterrows():
                cond_id = row['CondID']
                
                # 获取该condition的所有细胞
                cond_mask = adata_test.obs['CondID'] == cond_id
                n_cells_in_condition = np.sum(cond_mask)
                
                if n_cells_in_condition == 0:
                    continue
                
                # 获取原始细胞metadata
                orig_cells_obs = adata_test.obs[cond_mask].copy()
                
                # 为每个细胞复制相同的预测
                pred_expression = np.tile(y_pred_absolute[i:i+1], (n_cells_in_condition, 1))
                all_predictions.append(pred_expression)
                
                # 为每个细胞创建metadata
                for orig_idx, orig_cell_data in orig_cells_obs.iterrows():
                    pred_metadata = orig_cell_data.to_dict()
                    pred_metadata.update({
                        'prediction_model': model_name,
                        'prediction_method': f'baseline_{model_name.lower()}',
                        'data_type': 'prediction',
                        'original_cell_index': orig_idx  # 保存原始索引
                    })
                    all_metadata.append(pred_metadata)
            
            if not all_predictions:
                print(f"    {model_name}: 没有有效数据")
                continue
            
            # 创建AnnData对象
            combined_predictions = np.vstack(all_predictions)
            metadata_df = pd.DataFrame(all_metadata)
            
            adata_pred = sc.AnnData(
                X=combined_predictions,
                obs=metadata_df,
                var=adata_test.var.copy()
            )
            
            # 设置index
            adata_pred.obs.index = [f"{model_name.lower()}_pred_{i}" for i in range(adata_pred.n_obs)]
            
            # 🔥 修复：正确获取原始表达数据
            try:
                # 使用保存的原始索引来获取对应的原始表达数据
                original_indices = []
                for orig_idx in metadata_df['original_cell_index']:
                    # 在adata_test中找到对应的位置
                    pos = adata_test.obs.index.get_loc(orig_idx)
                    original_indices.append(pos)
                
                adata_pred.obsm["original_expression"] = adata_test.X[original_indices]
                if hasattr(adata_pred.obsm["original_expression"], 'toarray'):
                    adata_pred.obsm["original_expression"] = adata_pred.obsm["original_expression"].toarray()
                    
            except Exception as e:
                print(f"    ⚠️ 无法添加原始表达数据: {e}")
                # 继续执行，不添加原始表达数据
            
            # 添加预测信息
            adata_pred.uns["prediction_info"] = {
                "model_type": f"baseline_{model_name}",
                "prediction_timestamp": datetime.datetime.now().isoformat(),
                "total_predicted_cells": adata_pred.n_obs,
                "total_genes": adata_pred.n_vars
            }
            
            # 保存文件 - 🔥 修改文件名以区别版本 (random版本)
            pred_file = os.path.join(output_dir, f"baseline_{model_name.lower()}_expr_based_predictions_complete.h5ad")
            adata_pred.write_h5ad(pred_file)
            
            # 保存metadata
            obs_file = os.path.join(output_dir, f"baseline_{model_name.lower()}_expr_based_predictions_metadata.csv")
            adata_pred.obs.to_csv(obs_file, index=True)
            
            # 统计信息
            stats = {
                "model_name": model_name,
                "total_predicted_cells": int(adata_pred.n_obs),
                "total_genes": int(adata_pred.n_vars),
                "prediction_mean": float(np.mean(combined_predictions)),
                "prediction_std": float(np.std(combined_predictions)),
                "evaluation_method": "expression_based"
            }
            
            stats_file = os.path.join(output_dir, f"baseline_{model_name.lower()}_expr_based_prediction_statistics.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            
            saved_models[model_name] = adata_pred
            print(f"    ✅ {model_name}: {adata_pred.n_obs:,} 个细胞已保存")
        
        print(f"\n📁 所有预测结果已保存到: {output_dir}")
        print(f"💡 使用方法: adata_mlp = sc.read_h5ad('{output_dir}/baseline_mlp_expr_based_predictions_complete.h5ad')")
        
        return saved_models
        
    except Exception as e:
        print(f"保存预测结果失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None
        
# -----------------------
# Helper: Config recommendations
# -----------------------
def get_enhanced_config_recommendations(data_size_gb, available_memory_gb):
    """
    根据数据规模（GB）和可用内存（GB）返回：
      - target_samples: 采样数量
      - max_features: 最大特征数量
      - chunk_size: 分块处理大小
      - batch_size: 训练批次大小
      - mlp_hidden_layers: MLP 隐藏层结构
      - mlp_epochs: MLP 训练轮数
      - mlp_lr: MLP 学习率
    """
    if data_size_gb > 100:
        ts, mf, bs = 15000, 1500, 15000
        mlp_h = (2048, 1024, 512, 256)
        mlp_ep = 100
        mlp_lr = 5e-4
    elif data_size_gb > 50:
        ts, mf, bs = 25000, 2500, 20000
        mlp_h = (1024, 512, 256)
        mlp_ep = 40
        mlp_lr = 1e-3
    else:
        ts, mf, bs = 35000, 3000, 25000
        mlp_h = (512, 256)
        mlp_ep = 30
        mlp_lr = 1e-3

    if available_memory_gb < 8:
        cs = 1000
        ts = min(ts, 10000)
        mlp_h = (256, 128)
        mlp_ep = min(mlp_ep, 20)
    elif available_memory_gb < 16:
        cs = 2000
    else:
        cs = 5000

    return {
        'target_samples': ts,
        'max_features': mf,
        'chunk_size': cs,
        'batch_size': bs,
        'mlp_hidden_layers': mlp_h,
        'mlp_epochs': mlp_ep,
        'mlp_lr': mlp_lr
    }


# -----------------------
# Load filtered AnnData objects
# -----------------------
adata_ctrl_filter = sc.read_h5ad("./random_data/control_adata_processed.h5ad")
adata_test_filter = sc.read_h5ad("./random_data/test_adata_processed.h5ad")
adata_train_filter = sc.read_h5ad("./random_data/train_adata_processed.h5ad")

# -----------------------
# Dataset Wrapper for PyTorch
# -----------------------
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------
# Define Chatterjee Correlation
# -----------------------
def chatterjee_corr(x, y):
    """
    计算 Chatterjee 相关系数：
    1 - (3 / (n^2 - 1)) * sum_{i=1 to n-1} |rank(y_{(i+1)}) - rank(y_{(i)})|
    其中 y_{(i)} 是在 x 递增排序下对应的 y 值，rank 使用 1..n 的整数秩。
    """
    n = len(x)
    order = np.argsort(x)
    y_ordered = y[order]
    ranks = rankdata(y_ordered, method='ordinal')
    diff = np.abs(np.diff(ranks))
    num = diff.sum()
    return 1 - (3 * num) / (n**2 - 1)


# -----------------------
# 🔥 Modified Evaluation Function for Expression-based metrics
# -----------------------
def evaluate_expression(y_true_expr, y_pred_expr, sample_size=50000):
    """
    使用真实表达值和预测表达值计算评估指标
    处理NaN值：当预测值全为常数时，相关系数会是NaN，此时设为0。
    """
    y_true = y_true_expr.flatten()
    y_pred = y_pred_expr.flatten()

    if len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_s = y_true[idx]
        y_pred_s = y_pred[idx]
    else:
        y_true_s = y_true
        y_pred_s = y_pred

    # 计算相关系数，处理NaN情况
    try:
        pr, pp = pearsonr(y_true_s, y_pred_s)
        # 如果是NaN（通常因为预测值方差为0），设为0
        if np.isnan(pr) or np.isnan(pp):
            pr, pp = 0.0, 1.0
    except:
        pr, pp = 0.0, 1.0
    
    try:
        sr, sp = spearmanr(y_true_s, y_pred_s)
        # 如果是NaN（通常因为预测值方差为0），设为0
        if np.isnan(sr) or np.isnan(sp):
            sr, sp = 0.0, 1.0
    except:
        sr, sp = 0.0, 1.0
    
    try:
        cc = chatterjee_corr(y_true_s, y_pred_s)
        if np.isnan(cc):
            cc = 0.0
    except:
        cc = 0.0

    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pr,
        'Pearson_p': pp,
        'Spearman_r': sr,
        'Spearman_p': sp,
        'Chatterjee': cc
    }


# -----------------------
# 🔥 Modified Per-condition evaluation function for Expression-based metrics
# -----------------------
def evaluate_per_condition_expression(y_true_expr, y_pred_expr, condition_ids):
    """
    使用真实表达值和预测表达值计算每个condition的评估指标
    处理NaN值：当预测值全为常数时，相关系数会是NaN，此时设为0。
    """
    unique_conditions = np.unique(condition_ids)
    condition_metrics = []
    
    for cond_id in unique_conditions:
        # Get indices for this condition
        cond_mask = condition_ids == cond_id
        
        if np.sum(cond_mask) > 0:  # Ensure we have data for this condition
            y_true_cond = y_true_expr[cond_mask].flatten()
            y_pred_cond = y_pred_expr[cond_mask].flatten()
            
            # Calculate metrics for this condition
            try:
                try:
                    pr, pp = pearsonr(y_true_cond, y_pred_cond)
                    if np.isnan(pr) or np.isnan(pp):
                        pr, pp = 0.0, 1.0
                except:
                    pr, pp = 0.0, 1.0
                
                try:
                    sr, sp = spearmanr(y_true_cond, y_pred_cond)
                    if np.isnan(sr) or np.isnan(sp):
                        sr, sp = 0.0, 1.0
                except:
                    sr, sp = 0.0, 1.0
                
                try:
                    cc = chatterjee_corr(y_true_cond, y_pred_cond)
                    if np.isnan(cc):
                        cc = 0.0
                except:
                    cc = 0.0
                
                metrics = {
                    'CondID': cond_id,
                    'n_samples': len(y_true_cond),
                    'MSE': mean_squared_error(y_true_cond, y_pred_cond),
                    'MAE': mean_absolute_error(y_true_cond, y_pred_cond),
                    'R2': r2_score(y_true_cond, y_pred_cond),
                    'Pearson_r': pr,
                    'Pearson_p': pp,
                    'Spearman_r': sr,
                    'Spearman_p': sp,
                    'Chatterjee': cc
                }
                condition_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Could not calculate metrics for condition {cond_id}: {e}")
                continue
    
    return pd.DataFrame(condition_metrics)


# -----------------------
# Dataset Preparation
# -----------------------
def get_dense_X(adata):
    if hasattr(adata.X, 'toarray'):
        return adata.X.toarray().astype(np.float32)
    else:
        return adata.X.astype(np.float32)


# -----------------------
# Function to compute condition-wise means
# -----------------------
def compute_condition_means(adata):
    """
    计算每个condition的平均表达值
    返回: (condition_means, condition_info)
        condition_means: shape = (n_conditions, n_genes)
        condition_info: DataFrame with condition metadata
    """
    X = get_dense_X(adata).astype(np.float32)
    condition_ids = adata.obs['CondID'].values
    
    unique_conditions = np.unique(condition_ids)
    n_conditions = len(unique_conditions)
    n_genes = X.shape[1]
    
    condition_means = np.zeros((n_conditions, n_genes), dtype=np.float32)
    condition_info = []
    
    for i, cond_id in enumerate(unique_conditions):
        cond_mask = condition_ids == cond_id
        cond_cells = X[cond_mask]
        
        # Compute mean expression for this condition
        condition_means[i] = cond_cells.mean(axis=0)
        
        # Get condition metadata (assuming all cells in same condition have same metadata)
        cond_obs = adata.obs[cond_mask].iloc[0]
        condition_info.append({
            'CondID': cond_id,
            'cell_line': cond_obs['cell_line'],
            'drug': cond_obs['drug'],
            'dose': cond_obs['dose']
        })
    
    condition_info_df = pd.DataFrame(condition_info)
    
    del X
    gc.collect()
    
    return condition_means, condition_info_df


# -----------------------
# 🔥 Modified Main Pipeline Function for Expression-based Evaluation
# -----------------------
def run_sequential_memory_efficient_pipeline(
        adata_ctrl,
        adata_train,
        adata_test,
        n_estimators=300,
        max_memory_gb=4,
        target_samples=30000,
        max_features=3000,
        chunk_size=2000,
        precision='float32',
        output_file="sequential_efficient_metrics.csv",
        log_file="sequential_efficient_log.txt",
        mlp_hidden_layers=(1024, 512, 256),
        mlp_epochs=10,
        mlp_batch_size=20000,
        mlp_lr=1e-3,
        mlp_early_stop_patience=5,
        mlp_early_stop_min_delta=1e-4,
        dropout_rate=0.3,
        device='cuda'
):
    """
    Sequential execution pipeline with condition-wise mean processing
    Modified to use expression-based evaluation instead of delta-based
    """
    total_start = time.time()
    print_mem("Pipeline start")

    # -----------------------
    # 1. Control baselines (shared by both models)
    # -----------------------
    print("➤ Computing control baselines...")
    baseline_mean = {}       # baseline_mean[line] = mean vector of control cells
    X_ctrl = get_dense_X(adata_ctrl).astype(np.float32)
    lines_ctrl = adata_ctrl.obs['cell_line'].values

    for line in np.unique(lines_ctrl):
        X_line = X_ctrl[lines_ctrl == line]    # shape = (n_ctrl_cells, n_genes)
        baseline_mean[line] = X_line.mean(axis=0)  # compute mean per line

    del X_ctrl, lines_ctrl
    gc.collect()

    # -----------------------
    # 2. Compute condition-wise means for train/test
    # -----------------------
    print("➤ Computing condition-wise means for train data...")
    Y_train_means, train_condition_info = compute_condition_means(adata_train)
    print(f"Train: {Y_train_means.shape[0]} conditions, {Y_train_means.shape[1]} genes")

    print("➤ Computing condition-wise means for test data...")
    X_test_means, test_condition_info = compute_condition_means(adata_test)
    print(f"Test: {X_test_means.shape[0]} conditions, {X_test_means.shape[1]} genes")

    # 🔥 保存真实测试表达值用于评估
    Y_test_true_expr = X_test_means.copy()

    # -----------------------
    # 3. Compute delta for condition means
    # -----------------------
    def compute_delta_from_condition_means(condition_means, condition_info):
        """
        Compute Δ = (condition mean expression) - (control mean expression)
        """
        n_conditions = condition_means.shape[0]
        n_genes = condition_means.shape[1]
        delta = np.zeros((n_conditions, n_genes), dtype=np.float32)
        
        for i in range(n_conditions):
            cell_line = condition_info.iloc[i]['cell_line']
            ctrl_mean_vec = baseline_mean[cell_line]
            delta[i] = condition_means[i] - ctrl_mean_vec
        
        return delta

    print("➤ Computing deltas for condition means...")
    Y_train = compute_delta_from_condition_means(Y_train_means, train_condition_info)
    Y_test = compute_delta_from_condition_means(X_test_means, test_condition_info)
    
    del Y_train_means, X_test_means
    gc.collect()
    print_mem("After delta computation")

    # -----------------------
    # 4. Prepare encoders
    # -----------------------
    print("➤ Preparing encoders...")
    all_drugs = np.concatenate([train_condition_info['drug'].values, test_condition_info['drug'].values])
    all_cell_lines = np.concatenate([train_condition_info['cell_line'].values, test_condition_info['cell_line'].values])

    drug_encoder = LabelEncoder().fit(all_drugs)
    cell_encoder = LabelEncoder().fit(all_cell_lines)

    # One-hot encoders for MLP
    drug_onehot = OneHotEncoder(sparse_output=False, dtype=np.float32)
    cell_onehot = OneHotEncoder(sparse_output=False, dtype=np.float32)

    drug_onehot.fit(drug_encoder.transform(all_drugs).reshape(-1, 1))
    cell_onehot.fit(cell_encoder.transform(all_cell_lines).reshape(-1, 1))

    del all_drugs, all_cell_lines
    gc.collect()

    # -----------------------
    # 5. Prepare control-feature matrices for feature concatenation
    # -----------------------
    def prepare_control_features(condition_info):
        n_conditions = len(condition_info)
        n_genes = len(list(baseline_mean.values())[0])
        ctrl_features = np.zeros((n_conditions, n_genes), dtype=np.float32)
        
        for i in range(n_conditions):
            cell_line = condition_info.iloc[i]['cell_line']
            ctrl_features[i, :] = baseline_mean[cell_line]
        
        return ctrl_features

    ctrl_train = prepare_control_features(train_condition_info)
    ctrl_test = prepare_control_features(test_condition_info)

    # -----------------------
    # 6. MLP PROCESSING
    # -----------------------
    print("\n" + "=" * 50)
    print("PROCESSING MLP MODEL")
    print("=" * 50)

    def build_design_mlp(condition_info):
        """
        Build design matrix for MLP (one-hot encoding for drug and cell_line, plus dose).
        """
        drug_idx = drug_encoder.transform(condition_info['drug']).reshape(-1, 1).astype(int)
        cell_idx = cell_encoder.transform(condition_info['cell_line']).reshape(-1, 1).astype(int)
        dose = condition_info['dose'].values.astype(np.float32).reshape(-1, 1)

        drug_oh = drug_onehot.transform(drug_idx)
        cell_oh = cell_onehot.transform(cell_idx)

        X_cond = np.hstack([drug_oh, cell_oh, dose])
        return X_cond, drug_oh.shape[1], cell_oh.shape[1]

    print("➤ Building MLP design matrices...")
    X_train_cond_mlp, drug_oh_dim, cell_oh_dim = build_design_mlp(train_condition_info)
    X_test_cond_mlp, _, _ = build_design_mlp(test_condition_info)

    del drug_onehot, cell_onehot
    gc.collect()

    # Concatenate condition features with control-expression features
    X_train_mlp = np.hstack([X_train_cond_mlp, ctrl_train])
    X_test_mlp = np.hstack([X_test_cond_mlp, ctrl_test])

    del X_train_cond_mlp, X_test_cond_mlp
    gc.collect()

    print(f"MLP input shape: {X_train_mlp.shape}")
    print_mem("After MLP data preparation")

    # Define MLPModel
    class MLPRegressorFP32:
        def __init__(self, input_dim, output_dim, hidden_layers=(1024, 512, 256),
                     epochs=10, batch_size=20000, lr=1e-3,
                     early_stop_patience=5, early_stop_min_delta=1e-4,
                     dropout_rate=0.3, device='cuda'):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_layers = hidden_layers
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.early_stop_patience = early_stop_patience
            self.early_stop_min_delta = early_stop_min_delta
            self.dropout_rate = dropout_rate
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.loss_history = []

            # Build network
            layers = []
            prev_dim = input_dim

            # First layer
            first_h = hidden_layers[0]
            layers.append(nn.Linear(prev_dim, first_h))
            layers.append(nn.BatchNorm1d(first_h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = first_h

            # Hidden layers
            for h in hidden_layers[1:]:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.dropout_rate))
                prev_dim = h

            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))

            self.net = nn.Sequential(*layers).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            self.stats = {}

        def fit(self, X_train, Y_train):
            dataset = NumpyDataset(X_train, Y_train)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            best_loss = np.inf
            patience_counter = 0

            for epoch in tqdm(range(self.epochs), desc="MLP Epochs"):
                self.net.train()
                epoch_losses = []
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    self.optimizer.zero_grad()
                    preds = self.net(xb)
                    loss = self.criterion(preds, yb)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())

                avg_loss = np.mean(epoch_losses)
                self.loss_history.append(avg_loss)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

                if avg_loss + self.early_stop_min_delta < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        break

            self.stats['best_loss'] = best_loss

            # Plot loss curve - 🔥 修改文件名
            plt.figure()
            plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("MLP Training Loss")
            plt.grid(True)
            plt.savefig("mlp_loss_curve_random_expr_based_6_14_2025.pdf")
            plt.close()

        def predict(self, X):
            self.net.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X).float().to(self.device)
                preds = self.net(xb).cpu().numpy()
            return preds

        def get_stats(self):
            return self.stats

    # Train MLP
    print("➤ Training MLP...")
    mlp_start = time.time()
    mlp = MLPRegressorFP32(
        input_dim=X_train_mlp.shape[1],
        output_dim=Y_train.shape[1],
        hidden_layers=mlp_hidden_layers,
        epochs=mlp_epochs,
        batch_size=mlp_batch_size,
        lr=mlp_lr,
        early_stop_patience=mlp_early_stop_patience,
        early_stop_min_delta=mlp_early_stop_min_delta,
        dropout_rate=dropout_rate,
        device=device
    )
    mlp.fit(X_train_mlp, Y_train)
    mlp_train_time = time.time() - mlp_start

    # Predict with MLP
    print("➤ Predicting with MLP...")
    y_pred_mlp = mlp.predict(X_test_mlp)

    del X_train_mlp, X_test_mlp
    gc.collect()

    # 🔥 Convert delta predictions to absolute expressions for evaluation
    y_pred_mlp_expr = np.zeros_like(y_pred_mlp)
    for i in range(len(test_condition_info)):
        cell_line = test_condition_info.iloc[i]['cell_line']
        ctrl_mean_vec = baseline_mean[cell_line]
        y_pred_mlp_expr[i] = ctrl_mean_vec + y_pred_mlp[i]

    # Evaluate MLP - 🔥 使用表达值进行评估
    print("➤ Evaluating MLP...")
    metrics_mlp = evaluate_expression(Y_test_true_expr, y_pred_mlp_expr)
    
    # Per-condition evaluation for MLP
    mlp_condition_metrics = evaluate_per_condition_expression(Y_test_true_expr, y_pred_mlp_expr, test_condition_info['CondID'].values)

    print(f"MLP Results:")
    print(f"  R2: {metrics_mlp['R2']:.4f}")
    print(f"  Pearson r: {metrics_mlp['Pearson_r']:.4f}")
    print(f"  Spearman r: {metrics_mlp['Spearman_r']:.4f}")
    print(f"  Chatterjee: {metrics_mlp['Chatterjee']:.4f}")
    print_mem("After MLP")

    del mlp
    gc.collect()

    # -----------------------
    # 7. RF PROCESSING
    # -----------------------
    print("\n" + "=" * 50)
    print("PROCESSING RF MODEL")
    print("=" * 50)

    def build_design_rf(condition_info):
        """
        Build design matrix for RF (integer encoding).
        """
        drug = drug_encoder.transform(condition_info['drug']).astype(np.float32).reshape(-1, 1)
        cell = cell_encoder.transform(condition_info['cell_line']).astype(np.float32).reshape(-1, 1)
        dose = condition_info['dose'].values.astype(np.float32).reshape(-1, 1)
        return np.hstack([drug, cell, dose])

    print("➤ Building RF design matrices...")
    X_train_cond_rf = build_design_rf(train_condition_info)
    X_test_cond_rf = build_design_rf(test_condition_info)

    X_train_rf = np.hstack([X_train_cond_rf, ctrl_train])
    X_test_rf = np.hstack([X_test_cond_rf, ctrl_test])

    del X_train_cond_rf, X_test_cond_rf
    gc.collect()

    print(f"RF input shape: {X_train_rf.shape}")
    print_mem("After RF data preparation (before SelectKBest)")

    # SelectKBest
    cond_dim = 3
    X_train_gene_part = X_train_rf[:, cond_dim:]
    X_test_gene_part = X_test_rf[:, cond_dim:]

    selector = SelectKBest(score_func=f_regression, k=1500)
    selector.fit(X_train_gene_part, Y_train[:, 0])

    X_train_gene_selected = selector.transform(X_train_gene_part)
    X_test_gene_selected = selector.transform(X_test_gene_part)

    X_train_rf_selected = np.hstack([X_train_rf[:, :cond_dim], X_train_gene_selected])
    X_test_rf_selected = np.hstack([X_test_rf[:, :cond_dim], X_test_gene_selected])

    del X_train_rf, X_test_rf, X_train_gene_part, X_test_gene_part
    gc.collect()

    print(f"RF input shape after SelectKBest: {X_train_rf_selected.shape}")
    print_mem("After RF data preparation (after SelectKBest)")

    # Train RF
    print("➤ Training RF...")
    rf_start = time.time()
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        max_features="sqrt",
        n_jobs=50,
        random_state=42
    )
    rf.fit(X_train_rf_selected, Y_train)
    rf_train_time = time.time() - rf_start

    # Predict with RF
    print("➤ Predicting with RF...")
    y_pred_rf = rf.predict(X_test_rf_selected)

    del X_train_rf_selected, X_test_rf_selected
    gc.collect()

    # 🔥 Convert delta predictions to absolute expressions for evaluation
    y_pred_rf_expr = np.zeros_like(y_pred_rf)
    for i in range(len(test_condition_info)):
        cell_line = test_condition_info.iloc[i]['cell_line']
        ctrl_mean_vec = baseline_mean[cell_line]
        y_pred_rf_expr[i] = ctrl_mean_vec + y_pred_rf[i]

    # Evaluate RF - 🔥 使用表达值进行评估
    print("➤ Evaluating RF...")
    metrics_rf = evaluate_expression(Y_test_true_expr, y_pred_rf_expr)
    
    # Per-condition evaluation for RF
    rf_condition_metrics = evaluate_per_condition_expression(Y_test_true_expr, y_pred_rf_expr, test_condition_info['CondID'].values)

    print(f"RF Results:")
    print(f"  R2: {metrics_rf['R2']:.4f}")
    print(f"  Pearson r: {metrics_rf['Pearson_r']:.4f}")
    print(f"  Spearman r: {metrics_rf['Spearman_r']:.4f}")
    print(f"  Chatterjee: {metrics_rf['Chatterjee']:.4f}")
    print_mem("After RF")

    del rf
    gc.collect()

    # -----------------------
    # 7.6 NEW: Trivial Zero Baseline
    # -----------------------
    print("\n" + "=" * 50)
    print("TRIVIAL ZERO BASELINE: Using zeros as prediction")
    print("=" * 50)
    
    # 使用全零预测
    y_pred_trivial_zero = np.zeros_like(Y_test)
    
    # 🔥 Convert delta predictions to absolute expressions for evaluation
    y_pred_trivial_zero_expr = np.zeros_like(y_pred_trivial_zero)
    for i in range(len(test_condition_info)):
        cell_line = test_condition_info.iloc[i]['cell_line']
        ctrl_mean_vec = baseline_mean[cell_line]
        y_pred_trivial_zero_expr[i] = ctrl_mean_vec + y_pred_trivial_zero[i]
    
    # 评估 - 🔥 使用表达值进行评估
    print("➤ Evaluating Trivial Zero Baseline...")
    metrics_trivial_zero = evaluate_expression(Y_test_true_expr, y_pred_trivial_zero_expr)
    # 评估 - 🔥 使用表达值进行评估
    print("➤ Evaluating Trivial Zero Baseline...")
    metrics_trivial_zero = evaluate_expression(Y_test_true_expr, y_pred_trivial_zero_expr)
    trivial_zero_condition_metrics = evaluate_per_condition_expression(Y_test_true_expr, y_pred_trivial_zero_expr, test_condition_info['CondID'].values)
    
    print(f"Trivial Zero Baseline Results:")
    print(f"  R2: {metrics_trivial_zero['R2']:.4f}")
    print(f"  Pearson r: {metrics_trivial_zero['Pearson_r']:.4f}")
    print(f"  Spearman r: {metrics_trivial_zero['Spearman_r']:.4f}")
    print(f"  Chatterjee: {metrics_trivial_zero['Chatterjee']:.4f}")
    print_mem("After Trivial Zero Baseline")

    # -----------------------
    # 8. SAVE RESULTS
    # -----------------------
    # 🔥 新增：在删除变量之前，先保存预测数据
    prediction_data = {
        'y_pred_mlp': y_pred_mlp.copy(),
        'y_pred_rf': y_pred_rf.copy(),
        'y_pred_trivial_zero': y_pred_trivial_zero.copy(),
        'test_condition_info': test_condition_info.copy(),
        'baseline_mean': baseline_mean.copy()
    }
    del Y_train, Y_test, Y_test_true_expr, ctrl_train, ctrl_test, baseline_mean
    del drug_encoder, cell_encoder
    gc.collect()
    print_mem("After final cleanup")

    # Save overall metrics - 🔥 修改文件名
    metrics_df = pd.DataFrame({
        'Model': ['MLP', 'RF', 'TrivialZero'],
        'R2': [metrics_mlp['R2'], metrics_rf['R2'], metrics_trivial_zero['R2']],
        'Pearson_r': [metrics_mlp['Pearson_r'], metrics_rf['Pearson_r'], metrics_trivial_zero['Pearson_r']],
        'Spearman_r': [metrics_mlp['Spearman_r'], metrics_rf['Spearman_r'], metrics_trivial_zero['Spearman_r']],
        'Chatterjee': [metrics_mlp['Chatterjee'], metrics_rf['Chatterjee'], metrics_trivial_zero['Chatterjee']],
        'MSE': [metrics_mlp['MSE'], metrics_rf['MSE'], metrics_trivial_zero['MSE']],
        'MAE': [metrics_mlp['MAE'], metrics_rf['MAE'], metrics_trivial_zero['MAE']]
    })
    metrics_df.to_csv(output_file, index=False)

    # Save per-condition metrics - 🔥 修改文件名
    mlp_condition_metrics['Model'] = 'MLP'
    rf_condition_metrics['Model'] = 'RF'
    trivial_zero_condition_metrics['Model'] = 'TrivialZero'
    condition_metrics_combined = pd.concat([
        mlp_condition_metrics,
        rf_condition_metrics,
        trivial_zero_condition_metrics], ignore_index=True)
    condition_metrics_file = output_file.replace('.csv', '_per_condition.csv')
    condition_metrics_combined.to_csv(condition_metrics_file, index=False)

    # Log timing
    with open(log_file, 'w') as lf:
        lf.write(f"MLP train time: {mlp_train_time:.2f} s\n")
        lf.write(f"RF train time: {rf_train_time:.2f} s\n")
        lf.write(f"Total pipeline time: {time.time() - total_start:.2f} s\n")
        lf.write(f"Number of train conditions: {len(train_condition_info)}\n")
        lf.write(f"Number of test conditions: {len(test_condition_info)}\n")
        lf.write(f"Evaluation method: expression_based\n")

    return {
        'mlp': metrics_mlp,
        'rf': metrics_rf,
        'trivial_zero': metrics_trivial_zero,
        'mlp_condition_metrics': mlp_condition_metrics,
        'rf_condition_metrics': rf_condition_metrics,
        'trivial_zero_condition_metrics': trivial_zero_condition_metrics,
        'mlp_train_time': mlp_train_time,
        'rf_train_time': rf_train_time,
        'total_time': time.time() - total_start,
        'prediction_data': prediction_data
    }


# ===================
# Main execution
# ===================
if __name__ == "__main__":
    # Get configuration recommendations
    config = get_enhanced_config_recommendations(
        data_size_gb=500,
        available_memory_gb=4000
    )
    config['target_samples'] = 100000
    config['max_features'] = 1500
    config['chunk_size'] = 50000
    config['batch_size'] = 50000

    result = run_sequential_memory_efficient_pipeline(
        adata_ctrl=adata_ctrl_filter,
        adata_train=adata_train_filter,
        adata_test=adata_test_filter,
        n_estimators=150,

        # 使用配置建议
        target_samples=config['target_samples'],
        max_features=config['max_features'],
        chunk_size=config['chunk_size'],
        precision='float32',

        # 自定义参数
        max_memory_gb=4000,
        mlp_hidden_layers=config['mlp_hidden_layers'],
        mlp_epochs=config['mlp_epochs'],
        mlp_batch_size=config['batch_size'],
        mlp_lr=config['mlp_lr'],
        mlp_early_stop_patience=10,
        mlp_early_stop_min_delta=1e-5,
        dropout_rate=0.3,

        # 输出文件 - 🔥 修改文件名
        output_file="metrics_random_expr_based_full_6_14_2025.csv",
        log_file="random_expr_based_full_6_14_2025_log.txt",
        device='cuda:2'
    )

    # 🔥 新增：保存完整预测结果（简单版本）
    print("\n" + "="*40)
    print("保存random版本完整预测结果 (Expression-based)")
    print("="*40)
    
    adata_pred_dict = save_complete_baseline_predictions_simple(
        adata_test_filter,  # random版本使用原始随机split的测试数据
        result,
        "./baseline_random_expr_based_predictions"  # 🔥 修改输出目录名
    )
    
    if adata_pred_dict is not None:
        print("✅ random版本预测结果保存完成!")
        print("📁 保存位置: ./baseline_random_expr_based_predictions/")
    else:
        print("❌ 预测结果保存失败 - 请确保pipeline函数返回了prediction_data")

    print("\nPipeline finished (Expression-based Evaluation). Results:")
    print("MLP Metrics:", result['mlp'])
    print("RF Metrics: ", result['rf'])
    print("Trivial Zero Metrics:", result['trivial_zero'])