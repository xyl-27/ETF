import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config, get_model_config
from model import create_model
from utils import (
    engineer_features_39,
    engineer_features_158plus39,
    engineer_features_97,
    engineer_features_39plus97,
    engineer_features_158plus97,
    engineer_features_158plus39plus97,
    JQ_FACTORS,
)
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_code_column(df):
    """自动检测代码列名，股票用股票代码，ETF用代码"""
    for col in ["股票代码", "代码"]:
        if col in df.columns:
            return col
    raise KeyError(f"未找到代码列，可用列: {df.columns.tolist()}")


feature_cloums_map = {
    "39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
    "158+39": [
        "instrument",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌额",
        "换手率",
        "涨跌幅",
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
        "OPEN0",
        "HIGH0",
        "LOW0",
        "VWAP0",
        "ROC5",
        "ROC10",
        "ROC20",
        "ROC30",
        "ROC60",
        "MA5",
        "MA10",
        "MA20",
        "MA30",
        "MA60",
        "STD5",
        "STD10",
        "STD20",
        "STD30",
        "STD60",
        "BETA5",
        "BETA10",
        "BETA20",
        "BETA30",
        "BETA60",
        "RSQR5",
        "RSQR10",
        "RSQR20",
        "RSQR30",
        "RSQR60",
        "RESI5",
        "RESI10",
        "RESI20",
        "RESI30",
        "RESI60",
        "MAX5",
        "MAX10",
        "MAX20",
        "MAX30",
        "MAX60",
        "MIN5",
        "MIN10",
        "MIN20",
        "MIN30",
        "MIN60",
        "QTLU5",
        "QTLU10",
        "QTLU20",
        "QTLU30",
        "QTLU60",
        "QTLD5",
        "QTLD10",
        "QTLD20",
        "QTLD30",
        "QTLD60",
        "RANK5",
        "RANK10",
        "RANK20",
        "RANK30",
        "RANK60",
        "RSV5",
        "RSV10",
        "RSV20",
        "RSV30",
        "RSV60",
        "IMAX5",
        "IMAX10",
        "IMAX20",
        "IMAX30",
        "IMAX60",
        "IMIN5",
        "IMIN10",
        "IMIN20",
        "IMIN30",
        "IMIN60",
        "IMXD5",
        "IMXD10",
        "IMXD20",
        "IMXD30",
        "IMXD60",
        "CORR5",
        "CORR10",
        "CORR20",
        "CORR30",
        "CORR60",
        "CORD5",
        "CORD10",
        "CORD20",
        "CORD30",
        "CORD60",
        "CNTP5",
        "CNTP10",
        "CNTP20",
        "CNTP30",
        "CNTP60",
        "CNTN5",
        "CNTN10",
        "CNTN20",
        "CNTN30",
        "CNTN60",
        "CNTD5",
        "CNTD10",
        "CNTD20",
        "CNTD30",
        "CNTD60",
        "SUMP5",
        "SUMP10",
        "SUMP20",
        "SUMP30",
        "SUMP60",
        "SUMN5",
        "SUMN10",
        "SUMN20",
        "SUMN30",
        "SUMN60",
        "SUMD5",
        "SUMD10",
        "SUMD20",
        "SUMD30",
        "SUMD60",
        "VMA5",
        "VMA10",
        "VMA20",
        "VMA30",
        "VMA60",
        "VSTD5",
        "VSTD10",
        "VSTD20",
        "VSTD30",
        "VSTD60",
        "WVMA5",
        "WVMA10",
        "WVMA20",
        "WVMA30",
        "WVMA60",
        "VSUMP5",
        "VSUMP10",
        "VSUMP20",
        "VSUMP30",
        "VSUMP60",
        "VSUMN5",
        "VSUMN10",
        "VSUMN20",
        "VSUMN30",
        "VSUMN60",
        "VSUMD5",
        "VSUMD10",
        "VSUMD20",
        "VSUMD30",
        "VSUMD60",
        "sma_5",
        "sma_20",
        "ema_12",
        "ema_26",
        "rsi",
        "macd",
        "macd_signal",
        "volume_change",
        "obv",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "boll_mid",
        "boll_std",
        "atr_14",
        "ema_60",
        "volatility_10",
        "volatility_20",
        "return_1",
        "return_5",
        "return_10",
        "high_low_spread",
        "open_close_spread",
        "high_close_spread",
        "low_close_spread",
    ],
    "97": ["instrument"] + JQ_FACTORS,
    "39+97": ["instrument"] + JQ_FACTORS,
    "158+97": ["instrument"] + JQ_FACTORS,
    "158+39+97": ["instrument"]
    + JQ_FACTORS,  # 包含158+39的所有特征列（从engineer_features_158plus39plus97获取）
}
feature_engineer_func_map = {
    "39": engineer_features_39,
    "158+39": engineer_features_158plus39,
    "97": engineer_features_97,
    "39+97": engineer_features_39plus97,
    "158+97": engineer_features_158plus97,
    "158+39+97": engineer_features_158plus39plus97,
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed["open_t1"] = processed.groupby("股票代码")["开盘"].shift(-1)
    processed["open_t5"] = processed.groupby("股票代码")["开盘"].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed["open_t1"] > 1e-4]

    processed["label"] = (processed["open_t5"] - processed["open_t1"]) / (
        processed["open_t1"] + 1e-12
    )
    processed = processed.dropna(subset=["label"])

    processed.drop(columns=["open_t1", "open_t5"], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config["feature_num"] in feature_engineer_func_map, (
        f"Unsupported feature_num: {config['feature_num']}"
    )
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config["feature_num"]]
    feature_columns = feature_cloums_map[config["feature_num"]]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    groups = [group for _, group in df.groupby("股票代码", sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(
            tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc)
        )

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed["instrument"] = processed["股票代码"].map(stockid2idx)
    processed = processed.dropna(subset=["instrument"]).copy()
    processed["instrument"] = processed["instrument"].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(
            df, stockid2idx, desc="特征工程", drop_small_open=False
        )
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(
        df, stockid2idx, desc="验证集特征工程", drop_small_open=True
    )


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """

    def __init__(
        self,
        temperature=1.0,
        k=5,
        weight_factor=2.0,
        pairwise_weight=1,
        base_weight=1.0,
    ):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失 (KL散度 + Cross Entropy)"""

        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)

        # 加权 Cross Entropy（原实现未使用 weights）
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()

        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)

        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()

        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)

        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))

        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix

        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()

        return loss

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        _, top_indices = torch.topk(y_true, k, dim=1)

        # 2. 创建权重向量
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)

        # 组合两种损失
        total_loss = listwise + self.pairwise_weight * pairwise

        return total_loss


def calculate_ranking_metrics(y_pred, y_true, masks, k=5, hs300_returns=None):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)

    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []

    # ========== 新增指标 ==========
    excess_return_list = []  # 超额收益
    hit_rate_list = []  # 命中率
    proximity_score_list = []  # 接近度分数（新增）
    rank_ic_list = []  # 排序相关性
    precision_list = []  # 精确率 @k
    recall_list = []  # 召回率 @k
    mrr_list = []  # Mean Reciprocal Rank
    ndcg_list = []  # NDCG@k

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]  # This is the 5-day return

        N = valid_true.numel()

        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()

        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = (
            pred_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9
            else 0.0
        )
        ratio_random = (
            random_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9
            else 0.0
        )
        denominator = max_return_sum - random_return_sum

        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-6
            else 0.0
        )

        # ========== 计算新增指标 ==========

        # 4. 超额收益 (相对 HS300)
        if hs300_returns is not None:
            hs300_ret = hs300_returns[i].item() if i < len(hs300_returns) else 0.0
            excess_return = (pred_return_sum / k) - hs300_ret
        else:
            excess_return = pred_return_sum - random_return_sum
        excess_return_list.append(excess_return)

        # 5. 命中率 (Hit Rate): 选中的股票中有多少在真实Top-K中
        pred_set = set(pred_indices.cpu().numpy())
        true_set = set(true_indices.cpu().numpy())
        hit_rate = len(pred_set & true_set) / k
        hit_rate_list.append(hit_rate)

        # ========== 6. 接近度分数 (Proximity Score) ==========
        # 计算预测Top-K股票的真实排名百分位数
        pred_true_values = valid_true[pred_indices]
        percentiles = []
        for true_val in pred_true_values:
            # 收益比它大的股票比例
            worse_ratio = (valid_true <= true_val).sum().item() / N
            percentiles.append(worse_ratio)

        avg_percentile = np.mean(percentiles)

        # 计算随机基准（蒙特卡洛模拟）
        random_percentiles = []
        for _ in range(min(100, N)):  # 最多模拟100次
            random_pred = torch.randn_like(valid_true)
            _, random_indices = torch.topk(random_pred, k)
            random_vals = valid_true[random_indices]
            random_pct = [(valid_true <= v).sum().item() / N for v in random_vals]
            random_percentiles.append(np.mean(random_pct))
        random_benchmark = np.mean(random_percentiles) if random_percentiles else 0.27

        # 线性映射到[0,1]，使得随机水平≈0.5，完美预测≈1.0
        best_benchmark = 1.0
        worst_benchmark = 0.0

        if avg_percentile >= random_benchmark:
            # 高于随机水平：映射到 [0.5, 1.0]
            normalized = (
                0.5
                + (avg_percentile - random_benchmark)
                / (best_benchmark - random_benchmark)
                * 0.5
            )
        else:
            # 低于随机水平：映射到 [0.0, 0.5]
            normalized = (
                (avg_percentile - worst_benchmark)
                / (random_benchmark - worst_benchmark)
                * 0.5
            )

        proximity_score = max(0.0, min(1.0, normalized))
        proximity_score_list.append(proximity_score)

        # 7. 排序相关性 (Spearman Rank Correlation)
        try:
            from scipy.stats import spearmanr

            pred_np = valid_pred.detach().cpu().numpy()
            true_np = valid_true.detach().cpu().numpy()
            rank_corr, _ = spearmanr(pred_np, true_np)
            # 处理NaN情况
            if np.isnan(rank_corr):
                rank_corr = 0.0
        except:
            rank_corr = 0.0
        rank_ic_list.append(rank_corr)

        # 8. 精确率@k (Precision@k): 选中的股票中实际收益为正的比例
        positive_count = (pred_top_returns > 0).sum().item()
        precision = positive_count / k
        precision_list.append(precision)

        # 9. 召回率@k (Recall@k): 实际正收益股票中被选中的比例
        total_positive = (valid_true > 0).sum().item()
        if total_positive > 0:
            recall = positive_count / total_positive
        else:
            recall = 0.0
        recall_list.append(recall)

        # 10. MRR (Mean Reciprocal Rank): 第一个真实正收益股票的排名倒数
        # 按预测分数排序
        sorted_indices = torch.argsort(valid_pred, descending=True)
        mrr = 0.0
        for rank, idx in enumerate(sorted_indices.cpu().numpy(), 1):
            if valid_true[idx] > 0:  # 找到第一个正收益股票
                mrr = 1.0 / rank
                break
        mrr_list.append(mrr)

        # 11. NDCG@k (Normalized Discounted Cumulative Gain)
        # 计算DCG
        dcg = 0.0
        for rank, idx in enumerate(pred_indices.cpu().numpy(), 1):
            gain = valid_true[idx].item()
            dcg += gain / np.log2(rank + 1)

        # 计算IDCG (理想排序)
        ideal_indices = torch.argsort(valid_true, descending=True)[:k]
        idcg = 0.0
        for rank, idx in enumerate(ideal_indices.cpu().numpy(), 1):
            gain = valid_true[idx].item()
            idcg += gain / np.log2(rank + 1)

        ndcg = dcg / (idcg + 1e-12)
        ndcg_list.append(ndcg)

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    # 计算所有指标的平均值
    metrics = {
        # 原有指标
        "pred_return_sum": np.mean(pred_return_sum_list)
        if pred_return_sum_list
        else 0.0,
        "max_return_sum": np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        "random_return_sum": np.mean(random_return_sum_list)
        if random_return_sum_list
        else 0.0,
        "ratio_pred": np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        "ratio_random": np.mean(ratio_random_list) if ratio_random_list else 0.0,
        "final_score": np.mean(final_score_list) if final_score_list else 0.0,
        # 新增指标
        "excess_return": np.mean(excess_return_list)
        if excess_return_list
        else 0.0,  # 超额收益
        "hit_rate": np.mean(hit_rate_list) if hit_rate_list else 0.0,  # 命中率
        "proximity_score": np.mean(proximity_score_list)
        if proximity_score_list
        else 0.0,  # 接近度分数（新增）
        "rank_ic": np.mean(rank_ic_list) if rank_ic_list else 0.0,  # 排序IC
        "precision": np.mean(precision_list) if precision_list else 0.0,  # 精确率
        "recall": np.mean(recall_list) if recall_list else 0.0,  # 召回率
        "mrr": np.mean(mrr_list) if mrr_list else 0.0,  # 平均倒数排名
        "ndcg": np.mean(ndcg_list) if ndcg_list else 0.0,  # NDCG
        # 稳定性指标（标准差）
        "std_pred_return": np.std(pred_return_sum_list)
        if pred_return_sum_list
        else 0.0,
        "std_final_score": np.std(final_score_list) if final_score_list else 0.0,
    }

    return metrics


class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""

    def __init__(self, sequences, targets, relevance_scores, stock_indices, hs300_rets=None):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.hs300_rets = hs300_rets if hs300_rets else [0.0] * len(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequences": torch.FloatTensor(
                self.sequences[idx]
            ),
            "targets": torch.FloatTensor(self.targets[idx]),
            "relevance": torch.LongTensor(
                self.relevance_scores[idx]
            ),
            "stock_indices": torch.LongTensor(
                self.stock_indices[idx]
            ),
            "hs300_rets": torch.FloatTensor([self.hs300_rets[idx]]),
        }


def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item["sequences"] for item in batch]
    targets = [item["targets"] for item in batch]
    relevance = [item["relevance"] for item in batch]
    stock_indices = [item["stock_indices"] for item in batch]
    hs300_rets = [item["hs300_rets"] for item in batch]

    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)

    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, stock_idx, hs300_ret in zip(sequences, targets, relevance, stock_indices, hs300_rets):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        "sequences": torch.stack(
            padded_sequences
        ),  # [batch, max_stocks, seq_len, features]
        "targets": torch.stack(padded_targets),  # [batch, max_stocks]
        "relevance": torch.stack(padded_relevance),  # [batch, max_stocks]
        "stock_indices": torch.stack(padded_stock_indices),  # [batch, max_stocks]
        "masks": torch.stack(masks),  # [batch, max_stocks]
        "hs300_rets": torch.stack(hs300_rets),  # [batch, 1]
    }


# 排序训练函数
def train_ranking_model(
    model, dataloader, criterion, optimizer, device, epoch, writer, top_k=5, hs300_returns=None
):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training   Epoch {epoch + 1}"):
        sequences = batch["sequences"].to(
            device
        )  # [batch, max_stocks, seq_len, features]
        targets = batch["targets"].to(device)  # [batch, max_stocks] 真实涨跌幅
        relevance = batch["relevance"].to(
            device
        )  # [batch, max_stocks] 预处理的相关性得分
        masks = batch["masks"].to(device)  # [batch, max_stocks] 有效位置mask
        hs300_rets = batch["hs300_rets"].to(device)  # [batch, 1] HS300 收益率

        optimizer.zero_grad()

        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数

        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分

        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()

            if valid_indices.numel() == 0:
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]

            if len(valid_pred) > 1:
                # 直接使用预处理好的相关性得分，无需重新计算
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = (
                    batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
                )

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if not config.get("drop_clip", True):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                if writer:
                    writer.add_scalar(
                        "train/grad_norm",
                        grad_norm,
                        global_step=epoch * len(dataloader) + local_step,
                    )
            optimizer.step()

            total_loss += batch_loss.item()

            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(
                    masked_outputs, masked_targets, masks, k=top_k,
                    hs300_returns=hs300_rets.squeeze(1)
                )
                for metric_name, v in metrics.items():
                    if metric_name not in total_metrics:
                        total_metrics[metric_name] = 0
                    total_metrics[metric_name] += v

            local_step += 1
            if writer:
                writer.add_scalar(
                    "train/loss",
                    batch_loss.item(),
                    global_step=epoch * len(dataloader) + local_step,
                )
                for k, v in metrics.items():
                    writer.add_scalar(
                        f"train/{k}",
                        v,
                        global_step=epoch * len(dataloader) + local_step,
                    )

    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics


def evaluate_ranking_model(
    model, dataloader, criterion, device, writer, epoch, prefix="eval", top_k=5
):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            sequences = batch["sequences"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            hs300_rets = batch.get("hs300_rets", torch.zeros(sequences.size(0), 1)).to(device)

            # 模型预测
            outputs = model(sequences)

            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()

                if valid_indices.numel() == 0:
                    continue

                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)

                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(
                        len(valid_true), 0, -1, device=device, dtype=torch.float32
                    )
                    relevance_scores = relevance_scores.detach()

                    loss = criterion(
                        valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0)
                    )
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            # 计算评估指标
            metrics = calculate_ranking_metrics(
                masked_outputs, masked_targets, masks, k=top_k,
                hs300_returns=hs300_rets.squeeze(1)
            )
            for metric_name, v in metrics.items():
                if metric_name not in total_metrics:
                    total_metrics[metric_name] = 0
                total_metrics[metric_name] += v

            num_batches += 1

    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar(f"{prefix}/loss", avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f"{prefix}/{k}", v, global_step=epoch)

    return avg_loss, total_metrics


def predict_top_stocks(
    model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5
):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()

    # 获取最后一天的数据作为预测基础
    latest_date = data["日期"].max()

    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []

    for stock_code in data["股票代码"].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = (
            data[(data["股票代码"] == stock_code) & (data["日期"] <= latest_date)]
            .sort_values("日期")
            .tail(sequence_length)
        )

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])

    if len(day_sequences) == 0:
        return []

    # 转换为tensor
    sequences = (
        torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)
    )  # [1, num_stocks, seq_len, features]

    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]

        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]

        top_stocks = []
        for idx in top_indices:
            top_stocks.append(
                {
                    "stock_code": day_stock_codes[idx],
                    "predicted_score": scores[idx],
                    "rank": len(top_stocks) + 1,
                }
            )

    return top_stocks


def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append(
            {
                "排名": stock["rank"],
                "股票代码": stock["stock_code"],
                "预测分数": stock["predicted_score"],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length, val_months=2, val_folds=1,
                                   val_start_date=None, val_end_date=None):
    """按最后val_months或固定日期做验证集划分。

    Args:
        df: 数据DataFrame
        sequence_length: 序列长度
        val_months: 验证集月数（val_start_date为None时生效）
        val_folds: 验证集折数，1则不使用时间序列交叉验证
        val_start_date: 固定验证集开始日期（如"2026-01-01"），不为None时忽略val_months
        val_end_date: 固定验证集结束日期（如"2026-03-31"）
    """
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values(["日期", "股票代码"]).reset_index(drop=True)

    last_date = df["日期"].max()

    if val_start_date is not None:
        val_start = pd.to_datetime(val_start_date).normalize()
        val_end = pd.to_datetime(val_end_date).normalize() if val_end_date else last_date
    else:
        val_start = (last_date - pd.DateOffset(months=val_months)).normalize()
        val_end = last_date

    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df["日期"] < val_start].copy()
    val_df = df[(df["日期"] >= val_context_start) & (df["日期"] <= val_end)].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    if val_start_date is not None:
        print(f"验证集目标范围(固定日期): {val_start.date()} 到 {val_end.date()}")
    else:
        print(f"验证集目标范围(最后{val_months}个月): {val_start.date()} 到 {val_end.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    if val_folds > 1:
        print(f"[时间序列交叉验证] 验证集折数: {val_folds}, 每折长度: {val_months}个月")
        train_df.attrs["val_folds"] = val_folds
        train_df.attrs["val_months"] = val_months

    train_df["日期"] = train_df["日期"].dt.strftime("%Y-%m-%d")
    val_df["日期"] = val_df["日期"].dt.strftime("%Y-%m-%d")

    return train_df, val_df, val_start, val_end


# 主程序
def main():
    set_seed(config.get("seed", 42))
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 打印配置信息
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    # 临时加载数据获取维度信息
    data_file = os.path.join(config["data_path"], config.get("data_file", "train.csv"))
    temp_df = pd.read_csv(data_file)
    code_col = get_code_column(temp_df)
    config["code_col"] = code_col
    num_stocks = temp_df[code_col].nunique()
    feature_dim = len(
        [c for c in temp_df.columns if c not in [code_col, "日期", "label"]]
    )
    num_dates = temp_df["日期"].nunique()

    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"sequence_length: {config['sequence_length']}")
    print(f"feature_num:      {config['feature_num']}")
    print(f"device:           {device_str}")
    print(f"股票数量:         {num_stocks}")
    print(f"特征维度:         {feature_dim}")
    print(f"日期数量:         {num_dates}")
    print("=" * 60)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config["data_path"]
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = (
        SummaryWriter(log_dir=os.path.join(output_dir, "log")) if is_train else None
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. 数据加载
    data_file = os.path.join(data_path, config.get("data_file", "train.csv"))
    full_df = pd.read_csv(data_file)
    drops = [c for c in ["振幅", "涨跌额", "涨跌幅", "开盘", "收盘", "最高", "最低", "前收盘"] if c in full_df.columns]
    if drops:
        full_df = full_df.drop(columns=drops)
    full_df = full_df.rename(columns={
        "开盘_前复权": "开盘", "收盘_前复权": "收盘",
        "最高_前复权": "最高", "最低_前复权": "最低",
        "前收盘_前复权": "前收盘",
        "振幅_前复权": "振幅", "涨跌额_前复权": "涨跌额", "涨跌幅_前复权": "涨跌幅",
    })
    train_df, val_df, val_start, val_end = split_train_val_by_last_month(
        full_df,
        config["sequence_length"],
        config["val_months"],
        config.get("val_folds", 1),
        val_start_date=config.get("val_start_date"),
        val_end_date=config.get("val_end_date"),
    )

    # 获取所有股票ID，建立映射
    all_stock_ids = full_df["股票代码"].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 2. 特征工程与预处理
    train_data, features = preprocess_data(
        train_df, is_train=True, stockid2idx=stockid2idx
    )
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 保留原始 instrument（int 0-73）再缩放 — "instrument" 在 features 中
    train_instrument = train_data["instrument"].copy()
    val_instrument = val_data["instrument"].copy()
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    train_data["instrument"] = train_instrument
    val_data["instrument"] = val_instrument
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # 4. 创建排序数据集 - train / val / val_sliding 统一构造
    print("\n=== 创建排序数据集 ===")

    # 4.1 训练集: 使用全部训练数据
    print("\n[训练集]")
    train_sequences, train_targets, train_relevance, train_stock_indices, _ = (
        create_ranking_dataset_vectorized(
            train_data,
            features,
            config["sequence_length"],
            ranking_data_path=config.get("train_ranking_data_path"),
        )
    )
    print(f"训练集样本数: {len(train_sequences)}")

    # 4.2 验证集(按周): 只保留验证期内的日期
    print("\n[验证集-按周]")
    (
        val_sequences,
        val_targets,
        val_relevance,
        val_stock_indices,
        val_first_window_date,
        _,
        _,
    ) = create_ranking_dataset_vectorized(
        val_data,
        features,
        config["sequence_length"],
        ranking_data_path=config.get("val_ranking_data_path"),
        min_window_end_date=val_start.strftime("%Y-%m-%d"),
        verbose=True,
    )
    print(f"验证集样本数(按周): {len(val_sequences)}")

    # 记录按周验证的第一个窗口结束日期
    if val_first_window_date:
        val_first_sample_date = pd.to_datetime(val_first_window_date)
        print(f"验证集首个样本日期: {val_first_sample_date}")
    else:
        val_first_sample_date = val_start

    # 4.3 滑动验证集: 使用验证期内数据
    print("\n[验证集-滑动]")
    full_df_dates = pd.to_datetime(full_df["日期"])
    val_context_start = val_start - pd.tseries.offsets.BDay(
        config["sequence_length"] - 1
    )
    full_df["日期"] = full_df_dates
    val_sliding_df = full_df[
        (full_df["日期"] >= val_context_start) & (full_df["日期"] <= val_end)
    ]
    print(
        f"滑动验证取数范围: {val_context_start.strftime('%Y-%m-%d')} 到 {val_end.strftime('%Y-%m-%d')}"
    )
    print(
        f"滑动验证原始数据: {len(val_sliding_df)} 行, {val_sliding_df['日期'].nunique()} 唯一日期"
    )

    val_sliding_data, _ = preprocess_val_data(val_sliding_df, stockid2idx=stockid2idx)
    print(
        f"滑动验证预处理后: {len(val_sliding_data)} 行, {val_sliding_data['日期'].nunique()} 唯一日期"
    )
    val_sliding_data[features] = val_sliding_data[features].replace(
        [np.inf, -np.inf], np.nan
    )
    val_sliding_data = val_sliding_data.dropna(subset=features)
    val_sliding_instrument = val_sliding_data["instrument"].copy()
    val_sliding_data[features] = scaler.transform(val_sliding_data[features])
    val_sliding_data["instrument"] = val_sliding_instrument

    # 滑动验证使用val_first_sample_date作为min_window_end_date，与按周验证对齐
    min_date_for_sliding = val_first_sample_date.strftime("%Y-%m-%d")

    (
        val_sliding_sequences,
        val_sliding_targets,
        val_sliding_relevance,
        val_sliding_stock_indices,
        _,
        _,
        _,
    ) = create_ranking_dataset_vectorized(
        val_sliding_data,
        features,
        config["sequence_length"],
        ranking_data_path=None,
        min_window_end_date=min_date_for_sliding,
        require_natural_day_consecutive=False,
        verbose=True,
    )
    print(f"验证集样本数(滑动): {len(val_sliding_sequences)}")

    # 5. 创建 DataLoader
    train_dataset = RankingDataset(
        train_sequences, train_targets, train_relevance, train_stock_indices
    )
    val_dataset = RankingDataset(
        val_sequences, val_targets, val_relevance, val_stock_indices
    )
    val_sliding_dataset = RankingDataset(
        val_sliding_sequences,
        val_sliding_targets,
        val_sliding_relevance,
        val_sliding_stock_indices,
    )

    val_sliding_loader = DataLoader(
        val_sliding_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数(按周): {len(val_sequences)}")
    print(f"验证集样本数(滑动): {len(val_sliding_sequences)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # 6. 模型初始化
    model_type = config.get("model_type", "transformer")
    model_config = get_model_config(model_type)
    model_config.update(config)
    model = create_model(model_type, len(features), model_config, num_stocks)
    model.to(device)
    print(
        f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=config.get("top_k", 5),
        temperature=1.0,
        weight_factor=config["top5_weight"],
        pairwise_weight=config["pairwise_weight"],
        base_weight=config.get("base_weight", 1.0),
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=config["num_epochs"]
    )

    # 8. 排序模型训练
    if is_train:
        best_score = -float("inf")
        best_sliding_score = -float("inf")
        best_ndcg = -float("inf")
        best_epoch = -1
        best_ndcg_epoch = -1

        epoch_scores_file = os.path.join(output_dir, "epoch_scores.txt")
        with open(epoch_scores_file, "w") as f:
            f.write("epoch,weekly_score,sliding_score,train_loss,eval_loss\n")

        for epoch in range(config["num_epochs"]):
            print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")

            # 训练
            train_loss, train_metrics = train_ranking_model(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                writer,
                top_k=config.get("top_k", 5),
            )

            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            # 验证 (按周)
            eval_loss, eval_metrics = evaluate_ranking_model(
                model,
                val_loader,
                criterion,
                device,
                writer,
                epoch,
                top_k=config.get("top_k", 5),
            )

            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")

            # 滑动窗口验证 (更细粒度)
            if len(val_sliding_sequences) > 0:
                eval_sliding_loss, eval_sliding_metrics = evaluate_ranking_model(
                    model,
                    val_sliding_loader,
                    criterion,
                    device,
                    writer,
                    epoch,
                    prefix="sliding_",
                    top_k=config.get("top_k", 5),
                )

                print(f"Eval(Sliding) Loss: {eval_sliding_loss:.4f}")
                for k, v in eval_sliding_metrics.items():
                    print(f"Eval(Sliding) {k}: {v:.4f}")
            else:
                eval_sliding_loss = 0.0
                eval_sliding_metrics = {"final_score": 0.0}

            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar(
                    "train/learning_rate", scheduler.get_last_lr()[0], global_step=epoch
                )

            # 保存最佳模型（基于weekly final score）
            current_final_score = eval_metrics.get("final_score", 0.0)
            current_sliding_score = eval_sliding_metrics.get("final_score", 0.0)
            current_sliding_ndcg = eval_sliding_metrics.get("ndcg", 0.0)

            # 记录每个epoch的得分
            with open(epoch_scores_file, "a") as f:
                f.write(
                    f"{epoch + 1},{current_final_score:.6f},{current_sliding_score:.6f},{train_loss:.6f},{eval_loss:.6f}\n"
                )

            if current_final_score > best_score:
                best_score = current_final_score
                best_sliding_score = current_sliding_score
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_model.pth")
                )
                print(
                    f"保存最佳模型 - weekly: {best_score:.4f}, sliding: {current_sliding_score:.4f}"
                )

            if current_sliding_ndcg > best_ndcg:
                best_ndcg = current_sliding_ndcg
                best_ndcg_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_model_ndcg.pth")
                )
                print(
                    f"保存最佳NDCG模型 - epoch: {best_ndcg_epoch}, ndcg: {best_ndcg:.4f}"
                )

        if not os.path.exists(os.path.join(output_dir, "best_model_ndcg.pth")):
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_model_ndcg.pth")
            )

        print(
            f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 weekly final score: {best_score:.4f}, 最佳 ndcg: {best_ndcg:.4f} (epoch {best_ndcg_epoch})"
        )
        with open(os.path.join(output_dir, "final_score.txt"), "w") as f:
            f.write(
                f"Best epoch: {best_epoch}\nBest weekly_final_score: {best_score:.6f}\nBest sliding_final_score: {best_sliding_score:.6f}\nBest ndcg_epoch: {best_ndcg_epoch}\nBest sliding_ndcg: {best_ndcg:.6f}\n"
            )

        if writer:
            writer.close()

        return best_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config", help="Config module name"
    )
    args = parser.parse_args()

    if args.config != "config":
        import importlib

        config_module = importlib.import_module(args.config)
        config = config_module.config
        get_model_config = config_module.get_model_config

    # 多进程保护
    mp.set_start_method("spawn", force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
