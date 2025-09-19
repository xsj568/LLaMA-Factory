#!/usr/bin/env python3
"""
BERT排序微调脚本
用于提升query-document排序学习的NDCG@k指标

实现流程概览:
1) 数据准备: 将相同 query 下不同 rel 的文档构造成(正, 负)排序对。
2) 模型前向: 用 BERT 得到 query/doc 各自的[CLS]句向量, L2归一化后拼接输入排序头得到分数。
3) 损失函数: 
   - PairwiseLoss: 约束正样本分数 > 负样本分数, 采用hinge形式。
   - CombinedLoss: 将 MSE(分数贴近标注rel) 与 Pairwise(排序关系) 按权重加和。
4) 训练与评估: 标准训练循环, 以 NDCG@k 衡量排序质量。

继承设计说明:
- RankingDataset 继承自 torch.utils.data.Dataset:
  提供 __len__/__getitem__ 接口, 配合 DataLoader 进行小批量、shuffle、并行加载。
- BertRankingModel 继承自 torch.nn.Module:
  遵循PyTorch模块化范式, 统一参数管理/设备切换/权重保存, 容易与优化器/调度器集成。

数据格式: query, document, rel, weight
损失函数: MSE + Pairwise加权损失 或 仅Pairwise损失
模型: BERT (序列长度512)
优化目标: NDCG@k
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel, 
    BertTokenizer, 
    BertConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import ndcg_score
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import argparse
from pathlib import Path
import random
from collections import defaultdict
from datetime import datetime
import inspect
import sys
import traceback


# 设置日志格式，包含时间、行数、函数名
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger()


class RankingDataset(Dataset):
    """
    排序数据集类

    继承原因:
    - 继承 Dataset 以支持被 DataLoader 高效消费。
    - 内部可将原始 (query, document, rel, weight) 转换为 pairwise 训练所需的 (pos, neg) 对。
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 group_by_query: bool = True, max_pairs_per_query: int = 100,
                 group_field: str = 'key'):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            group_by_query: 是否按query分组
            max_pairs_per_query: 每个query的最大pair数量
        """
        logger.info(f"初始化RankingDataset: data_path={data_path}, max_length={max_length}")
        logger.info(f"分组设置: group_by_query={group_by_query}, max_pairs_per_query={max_pairs_per_query}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_by_query = group_by_query
        self.max_pairs_per_query = max_pairs_per_query
        self.group_field = group_field
        self.data = self._load_data(data_path)
        
        if self.group_by_query:
            logger.info("按query分组处理数据")
            self.query_groups = self._group_by_field()
            self.pairs = self._create_pairs()
        else:
            logger.info("创建简单排序对")
            self.pairs = self._create_simple_pairs()
        
        logger.info(f"RankingDataset初始化完成，共{len(self.pairs)}个排序对")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        logger.info(f"开始加载数据文件: {data_path}")
        data = []
        
        try:
            if data_path.endswith('.json'):
                logger.info("检测到JSON格式文件")
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_path.endswith('.jsonl'):
                logger.info("检测到JSONL格式文件")
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            logger.warning(f"第{line_num}行JSON解析失败: {e}")
                            continue
            else:
                logger.error(f"不支持的数据格式: {data_path}")
                raise ValueError("支持的数据格式: .json, .jsonl")
        except FileNotFoundError:
            logger.error(f"数据文件不存在: {data_path}")
            raise
        except Exception as e:
            logger.error(f"加载数据文件时发生错误: {e}")
            raise
            
        logger.info(f"数据加载完成: {len(data)} 条记录")
        return data
    
    def _group_by_field(self) -> Dict[str, List[Dict]]:
        """按指定字段(默认key, 回退query)分组数据"""
        logger.info(f"开始按{self.group_field}字段分组数据")
        groups = defaultdict(list)
        for item in self.data:
            key_value = item.get(self.group_field, item.get('query'))
            groups[key_value].append(item)
        logger.info(f"分组完成: {len(groups)} 个组，平均每组{len(self.data)/len(groups):.1f}条数据")
        return dict(groups)
    
    def _create_pairs(self) -> List[Dict]:
        """创建排序对"""
        logger.info("开始创建排序对")
        pairs = []
        total_queries = len(self.query_groups)
        
        for query_idx, (query, docs) in enumerate(self.query_groups.items()):
            # 按相关性分数排序
            docs.sort(key=lambda x: x['rel'], reverse=True)
            
            # 创建正负样本对
            query_pairs = 0
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    if docs[i]['rel'] > docs[j]['rel']:
                        pairs.append({
                            'query': query,
                            'pos_doc': docs[i]['document'],
                            'neg_doc': docs[j]['document'],
                            'pos_rel': docs[i]['rel'],
                            'neg_rel': docs[j]['rel'],
                            'pos_weight': docs[i]['weight'],
                            'neg_weight': docs[j]['weight']
                        })
                        query_pairs += 1
            
            # 每处理100个query记录一次进度
            if (query_idx + 1) % 100 == 0:
                logger.info(f"已处理{query_idx + 1}/{total_queries}个query，当前pairs数: {len(pairs)}")
            
            # 限制每个query的pair数量
            if len(pairs) > self.max_pairs_per_query * len(self.query_groups):
                # 随机采样
                random.shuffle(pairs)
                pairs = pairs[:self.max_pairs_per_query * len(self.query_groups)]
                logger.info(f"达到最大pairs限制，随机采样到{len(pairs)}个pairs")
        
        logger.info(f"排序对创建完成: {len(pairs)} 个pairs")
        return pairs
    
    def _create_simple_pairs(self) -> List[Dict]:
        """创建简单排序对（不按query分组）"""
        pairs = []
        
        # 随机配对
        for i in range(0, len(self.data) - 1, 2):
            if i + 1 < len(self.data):
                doc1, doc2 = self.data[i], self.data[i + 1]
                if doc1['rel'] != doc2['rel']:  # 确保有不同的相关性分数
                    if doc1['rel'] > doc2['rel']:
                        pairs.append({
                            'query': doc1['query'],
                            'pos_doc': doc1['document'],
                            'neg_doc': doc2['document'],
                            'pos_rel': doc1['rel'],
                            'neg_rel': doc2['rel'],
                            'pos_weight': doc1['weight'],
                            'neg_weight': doc2['weight']
                        })
                    else:
                        pairs.append({
                            'query': doc2['query'],
                            'pos_doc': doc2['document'],
                            'neg_doc': doc1['document'],
                            'pos_rel': doc2['rel'],
                            'neg_rel': doc1['rel'],
                            'pos_weight': doc2['weight'],
                            'neg_weight': doc1['weight']
                        })
        
        logger.info(f"创建简单排序对: {len(pairs)} 个pairs")
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            pair = self.pairs[idx]
            
            # 分别编码(兼容旧逻辑, 但训练将使用pair编码)
            query_encoding = self.tokenizer(
                pair['query'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            pos_doc_encoding = self.tokenizer(
                pair['pos_doc'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_doc_encoding = self.tokenizer(
                pair['neg_doc'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Pair编码: 将 query 与 doc 拼接为单条序列 [CLS] query [SEP] doc [SEP]
            pos_pair = self.tokenizer(
                pair['query'],
                pair['pos_doc'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_pair = self.tokenizer(
                pair['query'],
                pair['neg_doc'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            logger.error(f"处理第{idx}个排序对时发生错误: {e}")
            raise
        
        batch = {
            #'query_input_ids': query_encoding['input_ids'].squeeze(0),
            #'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            #'pos_doc_input_ids': pos_doc_encoding['input_ids'].squeeze(0),
            #'pos_doc_attention_mask': pos_doc_encoding['attention_mask'].squeeze(0),
            #'neg_doc_input_ids': neg_doc_encoding['input_ids'].squeeze(0),
            #'neg_doc_attention_mask': neg_doc_encoding['attention_mask'].squeeze(0),
            'pos_rel': torch.tensor(pair['pos_rel'], dtype=torch.float),
            'neg_rel': torch.tensor(pair['neg_rel'], dtype=torch.float),
            'pos_weight': torch.tensor(pair['pos_weight'], dtype=torch.float),
            'neg_weight': torch.tensor(pair['neg_weight'], dtype=torch.float)
        }
        # 添加pair输入
        batch['pos_pair_input_ids'] = pos_pair['input_ids'].squeeze(0)
        batch['pos_pair_attention_mask'] = pos_pair['attention_mask'].squeeze(0)
        batch['neg_pair_input_ids'] = neg_pair['input_ids'].squeeze(0)
        batch['neg_pair_attention_mask'] = neg_pair['attention_mask'].squeeze(0)
        # token_type_ids 可能不存在(如RoBERTa), 做兼容
        if 'token_type_ids' in pos_pair:
            batch['pos_pair_token_type_ids'] = pos_pair['token_type_ids'].squeeze(0)
            batch['neg_pair_token_type_ids'] = neg_pair['token_type_ids'].squeeze(0)
        else:
            zeros = torch.zeros_like(batch['pos_pair_input_ids'])
            batch['pos_pair_token_type_ids'] = zeros
            batch['neg_pair_token_type_ids'] = torch.zeros_like(batch['neg_pair_input_ids'])
        return batch


class BertRankingModel(nn.Module):
    """
    BERT排序模型

    继承原因与前向逻辑:
    - 继承 nn.Module 便于参数注册/设备迁移/保存加载。
    - 前向返回句向量; compute_ranking_score 通过拼接 query/doc 向量经排序头得到标量分数。
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', embedding_dim: int = 768):
        """
        初始化模型
        
        Args:
            model_name: BERT模型名称
            embedding_dim: 嵌入维度
        """
        logger.info(f"初始化BertRankingModel: model_name={model_name}, embedding_dim={embedding_dim}")
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        # 可选的投影层
        if embedding_dim != 768:
            logger.info(f"添加投影层: 768 -> {embedding_dim}")
            self.projection = nn.Linear(768, embedding_dim)
        else:
            logger.info("使用Identity投影层")
            self.projection = nn.Identity()
        
        # 排序头(向量拼接方式)
        self.ranking_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1)
        )
        # Pair方式的排序头(单向量)
        self.pair_head = nn.Linear(embedding_dim, 1)
        logger.info("BertRankingModel初始化完成")
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            
        Returns:
            文本向量表示
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        vector = self.projection(pooled_output)
        vector = F.normalize(vector, p=2, dim=1)
        return vector
    
    def compute_ranking_score(self, query_vector, doc_vector):
        """计算排序分数"""
        # 拼接query和document向量
        combined = torch.cat([query_vector, doc_vector], dim=1)
        score = self.ranking_head(combined)
        return score.squeeze(1)

    def compute_pair_score(self, input_ids, attention_mask, token_type_ids=None):
        """通过拼接后的单序列直接计算排序分数"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.pooler_output
        vec = self.projection(pooled)
        vec = F.normalize(vec, p=2, dim=1)
        score = self.pair_head(vec)
        return score.squeeze(1)

    def save_pretrained(self, save_directory: str):
        """
        保存模型到目录, 与Hugging Face风格兼容。

        - 保存内部BERT权重使用 HuggingFace 的 save_pretrained
        - 另存 projection / ranking_head / pair_head 的参数与最小配置
        """
        os.makedirs(save_directory, exist_ok=True)
        # 保存内部BERT
        self.bert.save_pretrained(save_directory)
        # 保存自定义层
        projection_path = os.path.join(save_directory, "projection.pt")
        ranking_head_path = os.path.join(save_directory, "ranking_head.pt")
        pair_head_path = os.path.join(save_directory, "pair_head.pt")
        if not isinstance(self.projection, nn.Identity):
            torch.save(self.projection.state_dict(), projection_path)
        torch.save(self.ranking_head.state_dict(), ranking_head_path)
        torch.save(self.pair_head.state_dict(), pair_head_path)
        # 保存最小配置
        cfg = {
            "embedding_dim": int(self.embedding_dim),
            "base_model": getattr(self.bert.config, "_name_or_path", "bert-base-uncased"),
            "has_projection": not isinstance(self.projection, nn.Identity)
        }
        with open(os.path.join(save_directory, "ranking_model_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        从目录加载模型, 与 save_pretrained 配套。
        """
        # 读取最小配置
        cfg_path = os.path.join(load_directory, "ranking_model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model", "bert-base-uncased")
            embedding_dim = int(cfg.get("embedding_dim", 768))
            has_projection = bool(cfg.get("has_projection", embedding_dim != 768))
        else:
            base_model = load_directory
            embedding_dim = 768
            has_projection = False

        model = cls(model_name=base_model, embedding_dim=embedding_dim)
        # 加载BERT权重
        model.bert = BertModel.from_pretrained(load_directory)
        # 加载自定义层
        projection_path = os.path.join(load_directory, "projection.pt")
        ranking_head_path = os.path.join(load_directory, "ranking_head.pt")
        pair_head_path = os.path.join(load_directory, "pair_head.pt")
        if has_projection and os.path.exists(projection_path) and not isinstance(model.projection, nn.Identity):
            state = torch.load(projection_path, map_location="cpu")
            model.projection.load_state_dict(state)
        if os.path.exists(ranking_head_path):
            state = torch.load(ranking_head_path, map_location="cpu")
            model.ranking_head.load_state_dict(state)
        if os.path.exists(pair_head_path):
            state = torch.load(pair_head_path, map_location="cpu")
            model.pair_head.load_state_dict(state)
        return model


class PairwiseLoss(nn.Module):
    """
    Pairwise损失函数

    设计动机:
    - 直接优化排序关系: 让正样本分数高于负样本分数至少 margin。
    - 比点式回归更贴合排序任务目标, 常用于学习排序(LTR)。
    """
    
    def __init__(self, margin: float = 1.0):
        """
        初始化Pairwise损失
        
        Args:
            margin: 边界值
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores, weights=None):
        """
        计算Pairwise损失
        
        Args:
            pos_scores: 正样本分数
            neg_scores: 负样本分数
            weights: 样本权重
            
        Returns:
            Pairwise损失
        """
        # 计算分数差
        score_diff = pos_scores - neg_scores
        
        # 计算hinge损失
        loss = torch.clamp(self.margin - score_diff, min=0.0)
        
        # 应用权重
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数: MSE + Pairwise

    设计动机:
    - MSE 使模型分数更贴近标注强度, Pairwise 强化相对次序, 二者取长互补。
    - 通过权重调节两者贡献, 适配不同数据/业务侧关注点。
    """
    
    def __init__(self, mse_weight: float = 0.5, pairwise_weight: float = 0.5, 
                 margin: float = 1.0):
        """
        初始化组合损失
        
        Args:
            mse_weight: MSE损失权重
            pairwise_weight: Pairwise损失权重
            margin: Pairwise损失边界值
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.pairwise_weight = pairwise_weight
        self.mse_loss = nn.MSELoss()
        self.pairwise_loss = PairwiseLoss(margin)
    
    def forward(self, pos_scores, neg_scores, pos_targets, neg_targets, 
                pos_weights=None, neg_weights=None):
        """
        计算组合损失
        
        Args:
            pos_scores: 正样本预测分数
            neg_scores: 负样本预测分数
            pos_targets: 正样本真实分数
            neg_targets: 负样本真实分数
            pos_weights: 正样本权重
            neg_weights: 负样本权重
            
        Returns:
            组合损失
        """
        # MSE损失
        mse_loss = self.mse_loss(pos_scores, pos_targets) + self.mse_loss(neg_scores, neg_targets)
        
        # Pairwise损失
        pairwise_loss = self.pairwise_loss(pos_scores, neg_scores, pos_weights)
        
        # 组合损失
        total_loss = self.mse_weight * mse_loss + self.pairwise_weight * pairwise_loss
        
        return total_loss, mse_loss, pairwise_loss


class RankingTrainer:
    """
    排序训练器

    实现流程:
    - train_epoch: 编码 -> 向量 -> 排序分数 -> 计算损失(组合或pairwise) -> 反向与优化
    - evaluate: 同上但不反向, 并统计 NDCG@k 作为排序质量指标
    """
    
    def __init__(self, model, tokenizer, device='cuda', loss_type='combined'):
        logger.info(f"初始化RankingTrainer: device={device}, loss_type={loss_type}")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.loss_type = loss_type
        
        if loss_type == 'combined':
            logger.info("使用组合损失函数(CombinedLoss)")
            self.criterion = CombinedLoss()
        elif loss_type == 'pairwise':
            logger.info("使用排序损失函数(PairwiseLoss)")
            self.criterion = PairwiseLoss()
        else:
            logger.error(f"不支持的损失函数类型: {loss_type}")
            raise ValueError("损失函数类型必须是 'combined' 或 'pairwise'")
        logger.info("RankingTrainer初始化完成")
    
    def evaluate(self, dataloader, k=10):
        """评估模型"""
        logger.info(f"开始评估模型，NDCG@k={k}")
        self.model.eval()
        total_loss = 0
        all_ndcg_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 移动数据到设备
                pos_pair_input_ids = batch['pos_pair_input_ids'].to(self.device)
                pos_pair_attention_mask = batch['pos_pair_attention_mask'].to(self.device)
                neg_pair_input_ids = batch['neg_pair_input_ids'].to(self.device)
                neg_pair_attention_mask = batch['neg_pair_attention_mask'].to(self.device)
                pos_pair_token_type_ids = batch['pos_pair_token_type_ids'].to(self.device)
                neg_pair_token_type_ids = batch['neg_pair_token_type_ids'].to(self.device)
                pos_rel = batch['pos_rel'].to(self.device)
                neg_rel = batch['neg_rel'].to(self.device)
                pos_weight = batch['pos_weight'].to(self.device)
                neg_weight = batch['neg_weight'].to(self.device)
                
                # 前向传播(拼接后的pair输入)
                pos_scores = self.model.compute_pair_score(
                    pos_pair_input_ids, pos_pair_attention_mask, pos_pair_token_type_ids
                )
                neg_scores = self.model.compute_pair_score(
                    neg_pair_input_ids, neg_pair_attention_mask, neg_pair_token_type_ids
                )
                
                # 计算损失
                if self.loss_type == 'combined':
                    loss, _, _ = self.criterion(
                        pos_scores, neg_scores, pos_rel, neg_rel, pos_weight, neg_weight
                    )
                else:
                    loss = self.criterion(pos_scores, neg_scores, pos_weight)
                
                total_loss += loss.item()
                
                # 计算NDCG
                ndcg = self.compute_ndcg_at_k(pos_scores, neg_scores, pos_rel, neg_rel, k)
                all_ndcg_scores.append(ndcg)
        
        avg_loss = total_loss / len(dataloader)
        avg_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0.0
        
        logger.info(f"评估完成: loss={avg_loss:.4f}, NDCG@{k}={avg_ndcg:.4f}")
        return avg_loss, avg_ndcg
    
    def compute_ndcg_at_k(self, pos_scores, neg_scores, pos_targets, neg_targets, k=10):
        """计算NDCG@k"""
        # 合并分数和目标
        all_scores = torch.cat([pos_scores, neg_scores], dim=0).cpu().numpy()
        all_targets = torch.cat([pos_targets, neg_targets], dim=0).cpu().numpy()
        
        # 确保有足够的样本
        if len(all_scores) < 2:
            return 0.0
        
        # 计算NDCG
        try:
            # 重塑为2D数组
            scores_2d = all_scores.reshape(1, -1)
            targets_2d = all_targets.reshape(1, -1)
            
            ndcg = ndcg_score(targets_2d, scores_2d, k=min(k, len(all_scores)))
            return ndcg
        except:
            return 0.0

def save_checkpoint(output_dir: str, step: int, model: BertRankingModel, tokenizer, train_loss: float, val_loss: float):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    metrics = {
        "step": step,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(ckpt_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return ckpt_dir


def list_checkpoints(output_dir: str):
    if not os.path.isdir(output_dir):
        return []
    items = []
    for name in os.listdir(output_dir):
        ckpt_path = os.path.join(output_dir, name)
        if os.path.isdir(ckpt_path) and name.startswith('checkpoint-step-'):
            metrics_path = os.path.join(ckpt_path, 'metrics.json')
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            items.append((ckpt_path, metrics))
    # 按步数排序
    def parse_step(p):
        try:
            return int(os.path.basename(p[0]).split('-')[-1])
        except Exception:
            return -1
    items.sort(key=parse_step)
    return items


def select_best_checkpoint(output_dir: str):
    """选择 metrics.json 中 val_loss 最小的 checkpoint。"""
    ckpts = list_checkpoints(output_dir)
    best = None
    best_val = float('inf')
    for path, metrics in ckpts:
        val_loss = metrics.get('val_loss', float('inf')) if isinstance(metrics, dict) else float('inf')
        if val_loss < best_val:
            best_val = val_loss
            best = (path, metrics)
    return best

def train_fn(args):
    """训练函数: 周期性保存checkpoint, 记录train/val损失"""
    logger.info("=" * 50)
    logger.info("开始训练流程")
    logger.info("=" * 50)
    
    # 直接使用显式提供的训练/验证集路径，不再进行数据切分
    train_path = args.train_path
    val_path = args.val_path
    if not train_path or not val_path:
        logger.error("请提供 --train_path 与 --val_path，已不再支持自动切分")
        return
    if not os.path.exists(train_path):
        logger.error(f"训练集不存在: {train_path}")
        return
    if not os.path.exists(val_path):
        logger.error(f"验证集不存在: {val_path}")
        return

    logger.info(f"训练参数: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    logger.info(f"模型: {args.model_name}, 设备: {args.device}, 损失类型: {args.loss_type}")
    logger.info(f"保存路径: {args.save_path}")

    # 初始化tokenizer和模型
    logger.info("初始化tokenizer和模型...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertRankingModel(args.model_name)

    # 创建数据集/加载器
    logger.info("创建数据集...")
    train_ds = RankingDataset(train_path, tokenizer, args.max_length, True, args.max_pairs_per_query, args.group_field)
    val_ds = RankingDataset(val_path, tokenizer, args.max_length, True, args.max_pairs_per_query, args.group_field)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"数据加载器创建完成: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    # 训练器与优化器
    trainer = RankingTrainer(model, tokenizer, args.device, args.loss_type)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    logger.info(f"优化器和调度器初始化完成: total_steps={total_steps}, warmup_steps={args.warmup_steps}")

    # 训练循环, 按步保存
    logger.info("开始训练...")
    os.makedirs(args.save_path, exist_ok=True)
    global_step = 0
    best_eval = float('inf')
    for epoch in range(args.num_epochs):
        logger.info(f"开始 Epoch {epoch + 1}/{args.num_epochs}")
        # 逐batch训练以便拿到步数
        trainer.model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_pair = 0.0
        for batch in tqdm(train_loader, desc=f"Training E{epoch+1}"):
            # 手动执行与 train_epoch 一致的逻辑
            pos_pair_input_ids = batch['pos_pair_input_ids'].to(trainer.device)
            pos_pair_attention_mask = batch['pos_pair_attention_mask'].to(trainer.device)
            neg_pair_input_ids = batch['neg_pair_input_ids'].to(trainer.device)
            neg_pair_attention_mask = batch['neg_pair_attention_mask'].to(trainer.device)
            pos_pair_token_type_ids = batch['pos_pair_token_type_ids'].to(trainer.device)
            neg_pair_token_type_ids = batch['neg_pair_token_type_ids'].to(trainer.device)
            pos_rel = batch['pos_rel'].to(trainer.device)
            neg_rel = batch['neg_rel'].to(trainer.device)
            pos_weight = batch['pos_weight'].to(trainer.device)
            neg_weight = batch['neg_weight'].to(trainer.device)

            pos_scores = trainer.model.compute_pair_score(pos_pair_input_ids, pos_pair_attention_mask, pos_pair_token_type_ids)
            neg_scores = trainer.model.compute_pair_score(neg_pair_input_ids, neg_pair_attention_mask, neg_pair_token_type_ids)

            if trainer.loss_type == 'combined':
                loss, mse_loss, pairwise_loss = trainer.criterion(pos_scores, neg_scores, pos_rel, neg_rel, pos_weight, neg_weight)
                epoch_mse += mse_loss.item()
                epoch_pair += pairwise_loss.item()
            else:
                loss = trainer.criterion(pos_scores, neg_scores, pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

            # 按间隔保存checkpoint并在验证集评估
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # train损失取当前batch或滑动窗口; 这里记录epoch均值的近似
                train_loss_avg = epoch_loss / max(1, global_step)
                val_loss, _ = trainer.evaluate(val_loader)
                ckpt_dir = save_checkpoint(args.save_path, global_step, trainer.model, trainer.tokenizer, train_loss_avg, val_loss)
                logger.info(f"保存checkpoint: {ckpt_dir} (train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f})")
                if val_loss < best_eval:
                    best_eval = val_loss

        # 每个epoch结束也做一次完整评估
        val_loss, val_ndcg = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch+1} 结束: train_loss={epoch_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}, NDCG@10={val_ndcg:.4f}")
        # 保存一个epoch末尾的checkpoint
        ckpt_dir = save_checkpoint(args.save_path, global_step, trainer.model, trainer.tokenizer, epoch_loss/len(train_loader), val_loss)
        logger.info(f"保存epoch末尾checkpoint: {ckpt_dir}")
    
    logger.info("=" * 50)
    logger.info("训练流程完成")
    logger.info("=" * 50)


def evaluate_fn(args):
    """评估函数: 遍历/选择checkpoint进行评估, 输出最佳。"""
    logger.info("=" * 50)
    logger.info("开始评估流程")
    logger.info("=" * 50)
    
    # 选择checkpoint
    ckpt = select_best_checkpoint(args.save_path) if args.select_best else None
    target_ckpts = []
    if args.checkpoint_dir:
        logger.info(f"评估指定checkpoint: {args.checkpoint_dir}")
        target_ckpts = [(args.checkpoint_dir, {})]
    elif ckpt is not None:
        logger.info(f"评估最佳checkpoint: {ckpt[0]}")
        target_ckpts = [ckpt]
    else:
        logger.info("评估所有checkpoint")
        target_ckpts = list_checkpoints(args.save_path)
    
    if not target_ckpts:
        logger.error("未找到可评估的checkpoint")
        return
    
    logger.info(f"找到{len(target_ckpts)}个checkpoint待评估")

    # 数据：直接使用显式提供的评估集路径
    if not args.eval_path:
        logger.error("请提供 --eval_path，已不再支持自动切分生成评估集")
        return
    test_path = args.eval_path
    if not os.path.exists(test_path):
        logger.error(f"评估集不存在: {test_path}")
        return

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    test_ds = RankingDataset(test_path, tokenizer, args.max_length, True, args.max_pairs_per_query, args.group_field)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = []
    for path, _ in target_ckpts:
        model = BertRankingModel.from_pretrained(path).to(args.device)
        trainer = RankingTrainer(model, tokenizer, args.device, args.loss_type)
        loss, ndcg = trainer.evaluate(test_loader)
        results.append({"checkpoint": path, "loss": float(loss), "ndcg@10": float(ndcg)})
        logger.info(f"评估 {path}: loss={loss:.4f}, NDCG@10={ndcg:.4f}")

    # 选择最佳
    best = sorted(results, key=lambda x: (x['loss'], -x['ndcg@10']))[0]
    logger.info(f"评估结果汇总: 共{len(results)}个模型")
    for result in results:
        logger.info(f"  {result['checkpoint']}: loss={result['loss']:.4f}, NDCG@10={result['ndcg@10']:.4f}")
    
    summary_path = os.path.join(args.save_path, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({"all": results, "best": best}, f, ensure_ascii=False, indent=2)
    logger.info(f"评估摘要已保存到: {summary_path}")
    logger.info(f"最佳模型: {best['checkpoint']} (loss={best['loss']:.4f}, NDCG@10={best['ndcg@10']:.4f})")
    
    logger.info("=" * 50)
    logger.info("评估流程完成")
    logger.info("=" * 50)


def deploy_fn(args):
    """部署为FastAPI服务。"""
    logger.info("=" * 50)
    logger.info("开始部署流程")
    logger.info("=" * 50)
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        logger.info("FastAPI依赖检查通过")
    except Exception as e:
        logger.error(f"FastAPI 依赖未安装: {e}")
        return

    device = args.device
    logger.info(f"加载模型: {args.model_dir}")
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertRankingModel.from_pretrained(args.model_dir).to(device)
    model.eval()
    logger.info("模型加载完成，开始创建API服务")

    app = FastAPI(title="BERT Ranking Service", version="1.0.0")

    class RankRequest(BaseModel):
        query: str
        document: str

    class BatchRankRequest(BaseModel):
        queries: List[str]
        documents: List[str]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/score")
    def score(req: RankRequest):
        with torch.no_grad():
            pair = tokenizer(req.query, req.document, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
            token_type_ids = pair.get('token_type_ids', torch.zeros_like(pair['input_ids']))
            s = model.compute_pair_score(pair['input_ids'].to(device), pair['attention_mask'].to(device), token_type_ids.to(device)).item()
            return {"score": float(s)}

    @app.post("/score_batch")
    def score_batch(req: BatchRankRequest):
        if not req.queries or not req.documents or len(req.queries) != len(req.documents):
            return {"scores": []}
        with torch.no_grad():
            pair = tokenizer(req.queries, req.documents, max_length=args.max_length, padding=True, truncation=True, return_tensors='pt')
            token_type_ids = pair.get('token_type_ids', torch.zeros_like(pair['input_ids']))
            s = model.compute_pair_score(pair['input_ids'].to(device), pair['attention_mask'].to(device), token_type_ids.to(device))
            return {"scores": s.cpu().tolist()}

    logger.info(f"启动API服务: {args.host}:{args.port}")
    logger.info("=" * 50)
    logger.info("部署流程完成，服务已启动")
    logger.info("=" * 50)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description='BERT排序微调脚本')
    parser.add_argument('--data_path', type=str, default='rank_train.json',
                       help='训练数据路径')
    parser.add_argument('--splits_dir', type=str, default='./data_splits',
                       help='切分后的数据输出目录')
    parser.add_argument('--train_path', type=str, default='data/rank_train.json',
                       help='可选: 直接指定训练集路径')
    parser.add_argument('--val_path', type=str, default='data/rank_test.json',
                       help='可选: 直接指定验证集路径')
    parser.add_argument('--eval_path', type=str, default='data/rank_valid.json',
                       help='可选: 直接指定评估集路径(评估函数用)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='BERT模型名称')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='预热步数')
    parser.add_argument('--save_path', type=str, default='./output/ranking_model',
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--loss_type', type=str, default='combined', choices=['combined', 'pairwise'],
                       help='损失函数类型: combined (MSE+Pairwise) 或 pairwise')
    parser.add_argument('--mse_weight', type=float, default=0.5,
                       help='MSE损失权重 (仅当loss_type=combined时有效)')
    parser.add_argument('--pairwise_weight', type=float, default=0.5,
                       help='Pairwise损失权重 (仅当loss_type=combined时有效)')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Pairwise损失边界值')
    parser.add_argument('--group_by_query', action='store_true',
                       help='按query分组创建pairs')
    parser.add_argument('--max_pairs_per_query', type=int, default=100,
                       help='每个query的最大pair数量')
    parser.add_argument('--group_field', type=str, default='key',
                       help='分组字段(默认key, 若缺省回退query)')
    parser.add_argument('--save_steps', type=int, default=2,
                       help='每多少步保存一次checkpoint, 0为不按步保存')
    # 不再支持示例数据创建，统一从本地JSON加载
    # 模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'deploy'],
                       help='运行模式: 训练/评估/部署')
    # 评估相关
    parser.add_argument('--checkpoint_dir', type=str, default='',
                       help='指定单个checkpoint目录评估, 为空则遍历全部')
    parser.add_argument('--select_best', action='store_true',
                       help='自动选择val_loss最小的checkpoint评估')
    # 部署相关
    parser.add_argument('--model_dir', type=str, default='',
                       help='部署时加载的模型目录, 为空则选best checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务Host')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务端口')
    
    args = parser.parse_args()
    logger.info("BERT排序微调脚本启动")
    logger.info(f"运行模式: {args.mode}")
    
    # 统一从本地JSON加载数据

    # 分派模式
    if args.mode == 'train':
        train_fn(args)
    elif args.mode == 'evaluate':
        evaluate_fn(args)
    elif args.mode == 'deploy':
        # 若未指定model_dir则选最佳
        if not args.model_dir:
            logger.info("未指定模型目录，自动选择最佳checkpoint")
            best = select_best_checkpoint(args.save_path)
            if best is None:
                logger.error("未找到最佳checkpoint用于部署")
                return
            args.model_dir = best[0]
            logger.info(f"选择最佳checkpoint: {args.model_dir}")
        deploy_fn(args)


if __name__ == "__main__":
    '''
    训练（按 key 切分并按步保存 checkpoint）
    python ranking_finetune.py --mode train --data_path rank_train.json --group_field key --splits_dir splits --save_path output/ranking --save_steps 100
    评估（自动选择最优 checkpoint）
    python ranking_finetune.py --mode evaluate --data_path rank_train.json --group_field key --splits_dir splits --save_path output/ranking --select_best
    部署（使用最佳 checkpoint）
    python ranking_finetune.py --mode deploy --save_path output/ranking --host 0.0.0.0 --port 8000
    '''
    try:
        main()
    except Exception as e:
        logger.error(f"排序微调模型错误: {traceback.format_exc()}")
