#!/usr/bin/env python3
"""
BERT向量化微调脚本
用于提升query-document相关性学习的topk召回率

实现流程概览:
1) 数据加载与整理: 读取每条 (query, document, rel, weight) 记录, 使用tokenizer编码。
2) 模型前向: 使用BERT获取[CLS]向量, 经过投影并L2归一化得到文本向量。
3) 相似度计算: 计算 query 向量与 document 向量的余弦相似度作为预测分数。
4) 损失计算: 使用带权重的MSE, 将预测分数拟合到标注相关性 rel (按 weight 加权)。
5) 训练与评估: 标准单机训练循环, 线性warmup+衰减, 以 NDCG@k 粗略衡量召回排序质量。

继承设计说明:
- VectorDataset 继承自 torch.utils.data.Dataset:
  提供 __len__/__getitem__ 接口, 以便与 DataLoader 无缝对接实现批处理和并行加载。
- BertVectorModel 继承自 torch.nn.Module:
  遵循PyTorch模块化设计, 便于参数注册、迁移到GPU、保存/加载权重, 与优化器对接训练。

数据格式: query, document, rel, weight
损失函数: 带权重的均方差损失
模型: BERT (序列长度512)
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
from collections import defaultdict
from datetime import datetime
import inspect
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


class VectorDataset(Dataset):
    """
    向量化数据集类

    继承原因:
    - 继承 Dataset 以实现与 DataLoader 的解耦, 让小批次组装、shuffle、并行加载更高效。
    - 数据样本是独立的(query, document, rel, weight), __getitem__ 返回模型前向所需的张量。
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
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
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                logger.error(f"不支持的数据格式: {data_path}")
                raise ValueError("支持的数据格式: .json, .jsonl")
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            raise
            
        logger.info(f"数据加载完成: {len(data)} 条记录")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取query和document
        query = item['query']
        document = item['document']
        relevance = float(item['rel'])  # 相关性分数
        weight = float(item['weight'])  # 权重
        
        # 对query和document分别编码
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        doc_encoding = self.tokenizer(
            document,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'doc_input_ids': doc_encoding['input_ids'].squeeze(0),
            'doc_attention_mask': doc_encoding['attention_mask'].squeeze(0),
            'relevance': torch.tensor(relevance, dtype=torch.float),
            'weight': torch.tensor(weight, dtype=torch.float),
            # 保留原始query文本, 便于按query分组评估NDCG
            'query_text': query
        }

class BertVectorModel(nn.Module):
    """
    BERT向量化模型

    继承原因:
    - 继承 nn.Module 以便:
      1) 自动注册可学习参数, 方便 optimizer 获取参数。
      2) 轻松切换 device (CPU/GPU), 使用 .to(device)。
      3) save_pretrained/load 以及与上层训练框架的对接更加规范。

    前向逻辑:
    - 输入编码后的 input_ids 与 attention_mask
    - 取 BERT 的 pooler_output ([CLS]) 作为句向量
    - 可选线性投影到目标维度并做 L2 归一化以便用于余弦相似度。
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', embedding_dim: int = 768):
        """
        初始化模型
        
        Args:
            model_name: BERT模型名称
            embedding_dim: 嵌入维度
        """
        logger.info(f"初始化BertVectorModel: model_name={model_name}, embedding_dim={embedding_dim}")
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        # 可选的投影层，用于调整嵌入维度
        if embedding_dim != 768:
            logger.info(f"添加投影层: 768 -> {embedding_dim}")
            self.projection = nn.Linear(768, embedding_dim)
        else:
            logger.info("使用Identity投影层")
            self.projection = nn.Identity()
        logger.info("BertVectorModel初始化完成")
    
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
        
        # 使用[CLS] token的表示作为句子向量
        pooled_output = outputs.pooler_output
        
        # 投影到目标维度
        vector = self.projection(pooled_output)
        
        # L2归一化
        vector = F.normalize(vector, p=2, dim=1)
        
        return vector

    def save_pretrained(self, save_directory: str):
        """
        保存模型到目录, 与Hugging Face风格兼容。

        - 保存内部的BERT权重使用 HuggingFace 的 save_pretrained
        - 另存 projection 层参数与最小配置, 方便 from_pretrained 还原
        """
        os.makedirs(save_directory, exist_ok=True)
        # 保存内部BERT
        self.bert.save_pretrained(save_directory)
        # 保存投影层(若为Identity则跳过)
        projection_path = os.path.join(save_directory, "projection.pt")
        if not isinstance(self.projection, nn.Identity):
            torch.save(self.projection.state_dict(), projection_path)
        # 保存最小配置
        cfg = {
            "embedding_dim": int(self.embedding_dim),
            "base_model": getattr(self.bert.config, "_name_or_path", "bert-base-uncased"),
            "has_projection": not isinstance(self.projection, nn.Identity)
        }
        with open(os.path.join(save_directory, "vector_model_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        从目录加载模型, 与 save_pretrained 配套。
        """
        # 读取最小配置
        cfg_path = os.path.join(load_directory, "vector_model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model", "bert-base-uncased")
            embedding_dim = int(cfg.get("embedding_dim", 768))
            has_projection = bool(cfg.get("has_projection", embedding_dim != 768))
        else:
            # 回退: 若无配置文件, 使用目录本身作为base_model, 假设768维
            base_model = load_directory
            embedding_dim = 768
            has_projection = False

        model = cls(model_name=base_model, embedding_dim=embedding_dim)
        # 加载BERT权重
        model.bert = BertModel.from_pretrained(load_directory)
        # 加载投影层
        projection_path = os.path.join(load_directory, "projection.pt")
        if has_projection and os.path.exists(projection_path) and not isinstance(model.projection, nn.Identity):
            state = torch.load(projection_path, map_location="cpu")
            model.projection.load_state_dict(state)
        return model


class WeightedMSELoss(nn.Module):
    """
    带权重的均方差损失函数

    设计动机:
    - 不同样本重要性不同时, 通过 weight 对样本的 MSE 进行加权, 让训练更关注关键样本。
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predicted_scores, target_scores, weights):
        """
        计算带权重的MSE损失
        
        Args:
            predicted_scores: 预测的相关性分数
            target_scores: 真实的相关性分数
            weights: 样本权重
            
        Returns:
            加权MSE损失
        """
        mse_losses = self.mse_loss(predicted_scores, target_scores)
        weighted_losses = mse_losses * weights
        return weighted_losses.mean()


class VectorTrainer:
    """
    向量化训练器

    实现流程:
    - train_epoch: 前向 -> 相似度 -> 加权MSE -> 反向 -> 更新参数 -> (可选)学习率调度
    - evaluate: 前向 -> 相似度 -> 加权MSE -> 聚合并计算 NDCG@k 作为效果参考
    - compute_similarity_score: 使用余弦相似度衡量query-doc贴合程度
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        logger.info(f"初始化VectorTrainer: device={device}")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = WeightedMSELoss()
        logger.info("VectorTrainer初始化完成")
        
    def compute_similarity_score(self, query_vector, doc_vector):
        """计算query和document的相似度分数"""
        # 使用余弦相似度
        similarity = torch.cosine_similarity(query_vector, doc_vector, dim=1)
        return similarity
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """训练一个epoch"""
        logger.info("开始训练一个epoch")
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            batch_count += 1
            # 移动数据到设备
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            doc_input_ids = batch['doc_input_ids'].to(self.device)
            doc_attention_mask = batch['doc_attention_mask'].to(self.device)
            relevance = batch['relevance'].to(self.device)
            weight = batch['weight'].to(self.device)
            
            # 前向传播
            query_vector = self.model(query_input_ids, query_attention_mask)
            doc_vector = self.model(doc_input_ids, doc_attention_mask)
            
            # 计算相似度分数
            predicted_scores = self.compute_similarity_score(query_vector, doc_vector)
            
            # 计算损失
            loss = self.criterion(predicted_scores, relevance, weight)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            # 每100个batch记录一次
            if batch_count % 100 == 0:
                logger.info(f"训练进度: batch {batch_count}, 当前损失: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch训练完成，平均损失: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, dataloader, k=10):
        """评估模型: 返回(平均加权MSE损失, 按query分组的NDCG@k均值)"""
        logger.info(f"开始评估模型，NDCG@k={k}")
        self.model.eval()
        total_loss = 0
        # 按query分组收集
        query_to_scores = defaultdict(list)
        query_to_targets = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 移动数据到设备
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                doc_input_ids = batch['doc_input_ids'].to(self.device)
                doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                relevance = batch['relevance'].to(self.device)
                weight = batch['weight'].to(self.device)
                # 如果数据集保留了原始query文本, 用于分组计算NDCG
                query_texts = batch.get('query_text', None)
                
                # 前向传播
                query_vector = self.model(query_input_ids, query_attention_mask)
                doc_vector = self.model(doc_input_ids, doc_attention_mask)
                
                # 计算相似度分数
                predicted_scores = self.compute_similarity_score(query_vector, doc_vector)
                
                # 计算损失
                loss = self.criterion(predicted_scores, relevance, weight)
                total_loss += loss.item()
                
                # 收集NDCG分组数据(仅当提供query_text)
                if query_texts is not None:
                    preds_np = predicted_scores.detach().cpu().numpy()
                    tars_np = relevance.detach().cpu().numpy()
                    for i, q in enumerate(query_texts):
                        query_to_scores[q].append(float(preds_np[i]))
                        query_to_targets[q].append(float(tars_np[i]))
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"评估损失计算完成: {avg_loss:.4f}")
        
        # 计算NDCG@k (对含>=2文档的query求均值)
        ndcg_mean = 0.0
        if len(query_to_scores) > 0:
            logger.info(f"开始计算NDCG@k，共{len(query_to_scores)}个query")
            ndcgs = []
            for q, scores in query_to_scores.items():
                targets = query_to_targets[q]
                if len(scores) < 2:
                    continue
                scores_arr = np.asarray(scores, dtype=np.float32).reshape(1, -1)
                targets_arr = np.asarray(targets, dtype=np.float32).reshape(1, -1)
                try:
                    ndcgs.append(ndcg_score(targets_arr, scores_arr, k=min(k, scores_arr.shape[1])))
                except Exception as e:
                    logger.warning(f"计算NDCG时出错: {e}")
                    continue
            ndcg_mean = float(np.mean(ndcgs)) if ndcgs else 0.0
            logger.info(f"NDCG@k计算完成: {ndcg_mean:.4f} (基于{len(ndcgs)}个有效query)")
        else:
            logger.warning("没有找到query文本，无法计算NDCG")
        
        logger.info(f"评估完成: loss={avg_loss:.4f}, NDCG@{k}={ndcg_mean:.4f}")
        return avg_loss, ndcg_mean
    


## 删除了示例数据创建逻辑，统一从本地JSON加载


def save_checkpoint(output_dir: str, step: int, model: 'BertVectorModel', tokenizer, train_loss: float, val_loss: float):
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
    def parse_step(p):
        try:
            return int(os.path.basename(p[0]).split('-')[-1])
        except Exception:
            return -1
    items.sort(key=parse_step)
    return items


def select_best_checkpoint(output_dir: str):
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
    logger.info("=" * 50)
    logger.info("开始训练流程")
    logger.info("=" * 50)
    
    # 直接使用显式提供的数据集
    train_path = args.train_path
    val_path = args.val_path
    if not (train_path and val_path):
        logger.error("请提供 --train_path 与 --val_path")
        return
    if not os.path.exists(train_path):
        logger.error(f"训练集不存在: {train_path}")
        return
    if not os.path.exists(val_path):
        logger.error(f"验证集不存在: {val_path}")
        return

    logger.info(f"训练参数: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    logger.info(f"模型: {args.model_name}, 设备: {args.device}")
    logger.info(f"保存路径: {args.save_path}")

    logger.info("初始化tokenizer和模型...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertVectorModel(args.model_name)

    logger.info("创建数据集...")
    train_ds = VectorDataset(train_path, tokenizer, args.max_length)
    val_ds = VectorDataset(val_path, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"数据加载器创建完成: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    trainer = VectorTrainer(model, tokenizer, args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    logger.info(f"优化器和调度器初始化完成: total_steps={total_steps}, warmup_steps={args.warmup_steps}")

    os.makedirs(args.save_path, exist_ok=True)
    logger.info("开始训练...")
    global_step = 0
    best_val = float('inf')
    for epoch in range(args.num_epochs):
        logger.info(f"开始 Epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training E{epoch+1}"):
            query_input_ids = batch['query_input_ids'].to(trainer.device)
            query_attention_mask = batch['query_attention_mask'].to(trainer.device)
            doc_input_ids = batch['doc_input_ids'].to(trainer.device)
            doc_attention_mask = batch['doc_attention_mask'].to(trainer.device)
            relevance = batch['relevance'].to(trainer.device)
            weight = batch['weight'].to(trainer.device)

            query_vector = model(query_input_ids, query_attention_mask)
            doc_vector = model(doc_input_ids, doc_attention_mask)
            predicted_scores = trainer.compute_similarity_score(query_vector, doc_vector)
            loss = trainer.criterion(predicted_scores, relevance, weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += loss.item()
            global_step += 1

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                train_loss_avg = running_loss / max(1, global_step)
                val_loss, _ = trainer.evaluate(val_loader)
                ckpt_dir = save_checkpoint(args.save_path, global_step, model, tokenizer, train_loss_avg, val_loss)
                logger.info(f"保存checkpoint: {ckpt_dir} (train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    logger.info(f"新的最佳验证损失: {best_val:.4f}")

        # epoch end
        val_loss, ndcg = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch+1} 结束: train_loss={running_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}, NDCG@10={ndcg:.4f}")
        ckpt_dir = save_checkpoint(args.save_path, global_step, model, tokenizer, running_loss/len(train_loader), val_loss)
        logger.info(f"保存epoch末尾checkpoint: {ckpt_dir}")
    
    logger.info("=" * 50)
    logger.info("训练流程完成")
    logger.info("=" * 50)


def evaluate_fn(args):
    logger.info("=" * 50)
    logger.info("开始评估流程")
    logger.info("=" * 50)
    
    # 待评估checkpoint集合
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

    # 测试集
    if not args.test_path or not os.path.exists(args.test_path):
        logger.error("请提供有效的 --test_path")
        return

    logger.info(f"加载测试集: {args.test_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    test_ds = VectorDataset(args.test_path, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"测试集加载完成: {len(test_ds)}条数据, {len(test_loader)}个batch")

    results = []
    for i, (path, _) in enumerate(target_ckpts, 1):
        logger.info(f"评估checkpoint {i}/{len(target_ckpts)}: {path}")
        model = BertVectorModel.from_pretrained(path).to(args.device)
        trainer = VectorTrainer(model, tokenizer, args.device)
        loss, ndcg = trainer.evaluate(test_loader)
        results.append({"checkpoint": path, "loss": float(loss), "ndcg@10": float(ndcg)})
        logger.info(f"评估完成: loss={loss:.4f}, NDCG@10={ndcg:.4f}")

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
    model_dir = args.model_dir
    if not model_dir:
        logger.info("未指定模型目录，自动选择最佳checkpoint")
        best = select_best_checkpoint(args.save_path)
        if best is None:
            logger.error("未找到最佳checkpoint用于部署")
            return
        model_dir = best[0]
        logger.info(f"选择最佳checkpoint: {model_dir}")

    logger.info(f"加载模型: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertVectorModel.from_pretrained(model_dir).to(device)
    model.eval()
    logger.info("模型加载完成，开始创建API服务")

    app = FastAPI(title="BERT Vector Service", version="1.0.0")

    class EmbedRequest(BaseModel):
        text: str

    class EmbedBatchRequest(BaseModel):
        texts: List[str]

    class ScoreRequest(BaseModel):
        query: str
        document: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/embed")
    def embed(req: EmbedRequest):
        with torch.no_grad():
            enc = tokenizer(req.text, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
            vec = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            return {"vector": vec.squeeze(0).cpu().tolist()}

    @app.post("/embed_batch")
    def embed_batch(req: EmbedBatchRequest):
        if not req.texts:
            return {"vectors": []}
        with torch.no_grad():
            enc = tokenizer(req.texts, max_length=args.max_length, padding=True, truncation=True, return_tensors='pt')
            vecs = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            return {"vectors": vecs.cpu().tolist()}

    @app.post("/score")
    def score(req: ScoreRequest):
        with torch.no_grad():
            q = tokenizer(req.query, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
            d = tokenizer(req.document, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')
            qv = model(q['input_ids'].to(device), q['attention_mask'].to(device))
            dv = model(d['input_ids'].to(device), d['attention_mask'].to(device))
            sim = torch.cosine_similarity(qv, dv, dim=1).item()
            return {"score": float(sim)}

    logger.info(f"启动API服务: {args.host}:{args.port}")
    logger.info("=" * 50)
    logger.info("部署流程完成，服务已启动")
    logger.info("=" * 50)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    logger.info("BERT向量化微调脚本启动")
    parser = argparse.ArgumentParser(description='BERT向量化微调脚本')
    parser.add_argument('--train_path', type=str, default='data/vec_train.json',
                       help='训练集路径')
    parser.add_argument('--val_path', type=str, default='data/vec_valid.json',
                       help='验证集路径')
    parser.add_argument('--test_path', type=str, default='data/vec_test.json',
                       help='测试集路径')
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
    parser.add_argument('--save_path', type=str, default='./output/vector_model',
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    # 不再支持示例数据创建，统一从本地JSON加载
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'deploy'],
                       help='运行模式: 训练/评估/部署')
    # 训练与保存
    parser.add_argument('--save_steps', type=int, default=2,
                       help='每多少步保存一次checkpoint, 0为不按步保存')
    # 评估相关
    parser.add_argument('--checkpoint_dir', type=str, default='',
                       help='指定单个checkpoint目录评估, 为空则遍历全部')
    parser.add_argument('--select_best', action='store_true',
                       help='自动选择val_loss最小的checkpoint评估')
    # 部署相关
    parser.add_argument('--model_dir', type=str, default='',
                       help='部署时加载的模型目录, 为空则选best checkpoint')
    # 服务相关
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务Host')
    parser.add_argument('--port', type=int, default=8000,
                       help='服务端口')
    
    args = parser.parse_args()
    logger.info(f"运行模式: {args.mode}")
    
    # 分派模式
    if args.mode == 'train':
        train_fn(args)
    elif args.mode == 'evaluate':
        evaluate_fn(args)
    elif args.mode == 'deploy':
        deploy_fn(args)
    else:
        logger.error(f"未知的运行模式: {args.mode}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"向量微调模型错误: {traceback.format_exc()}")
