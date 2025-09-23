import os
import sys
import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import ndcg_score
from datetime import datetime

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ranking_finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RankingFinetune")

class RankingDataset(Dataset):
    """排序模型的数据集类"""
    def __init__(self, data_path, tokenizer, max_length=512, max_pairs_per_query=100):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            max_pairs_per_query: 每个查询的最大成对数量
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs_per_query = max_pairs_per_query
        
        # 加载数据
        logger.info(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"数据加载完成，共 {len(self.data)} 条样本")
        
        # 按key分组
        self.query_groups = self._group_by_key()
        # 创建成对的训练样本
        self.pairs = self._create_pairs()
        
        logger.info(f"创建训练对完成，共 {len(self.pairs)} 对")
    
    def _group_by_key(self):
        """按key分组"""
        query_groups = {}
        for sample in self.data:
            key = sample["key"]
            if key not in query_groups:
                query_groups[key] = []
            query_groups[key].append(sample)
        return query_groups
    
    def _create_pairs(self):
        """创建成对的训练样本"""
        pairs = []
        for key, samples in self.query_groups.items():
            # 按相关度降序排序
            samples_sorted = sorted(samples, key=lambda x: x["rel"], reverse=True)
            
            # 为每个样本创建与低相关度样本的配对
            num_pairs = 0
            for i in range(len(samples_sorted)):
                for j in range(i + 1, len(samples_sorted)):
                    if num_pairs >= self.max_pairs_per_query:
                        break
                    
                    # 确保i的相关性大于j
                    if samples_sorted[i]["rel"] > samples_sorted[j]["rel"]:
                        # 同时保存key和query信息
                        pairs.append({
                            "key": key,
                            "query": samples_sorted[i]["query"],  # 使用i的query
                            "doc_pos": samples_sorted[i]["document"],
                            "doc_neg": samples_sorted[j]["document"],
                            "rel_pos": samples_sorted[i]["rel"],
                            "rel_neg": samples_sorted[j]["rel"],
                            "weight_pos": samples_sorted[i]["weight"],
                            "weight_neg": samples_sorted[j]["weight"]
                        })
                        num_pairs += 1
                if num_pairs >= self.max_pairs_per_query:
                    break
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """获取单个样本，返回query和文档拼接后的表示"""
        pair = self.pairs[idx]
        
        # 拼接query和正文档
        query_doc_pos = f"{pair['query']} [SEP] {pair['doc_pos']}"
        # 拼接query和负文档
        query_doc_neg = f"{pair['query']} [SEP] {pair['doc_neg']}"
        
        # 处理拼接后的query+正文档
        query_doc_pos_encoding = self.tokenizer(
            query_doc_pos,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # 处理拼接后的query+负文档
        query_doc_neg_encoding = self.tokenizer(
            query_doc_neg,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "query_doc_pos_input_ids": query_doc_pos_encoding["input_ids"].squeeze(0),
            "query_doc_pos_attention_mask": query_doc_pos_encoding["attention_mask"].squeeze(0),
            "query_doc_neg_input_ids": query_doc_neg_encoding["input_ids"].squeeze(0),
            "query_doc_neg_attention_mask": query_doc_neg_encoding["attention_mask"].squeeze(0),
            "rel_pos": torch.tensor(pair["rel_pos"], dtype=torch.float32),
            "rel_neg": torch.tensor(pair["rel_neg"], dtype=torch.float32),
            "weight_pos": torch.tensor(pair["weight_pos"], dtype=torch.float32),
            "weight_neg": torch.tensor(pair["weight_neg"], dtype=torch.float32)
        }

class BertRankingModel(nn.Module):
    """BERT排序模型"""
    def __init__(self, model_name):
        """
        初始化模型
        Args:
            model_name: BERT模型名称
        """
        super(BertRankingModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 添加一个线性层用于评分
        self.rank_head = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        # 获取拼接后的query+文档的表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 计算排序分数
        score = self.rank_head(cls_embedding)
        
        return score.squeeze(-1)
    
    def compute_pair_score(self, query_doc_pos_input_ids, query_doc_pos_attention_mask, query_doc_neg_input_ids, query_doc_neg_attention_mask):
        """计算成对分数"""
        # 获取query+正文档的分数
        score_pos = self.forward(query_doc_pos_input_ids, query_doc_pos_attention_mask)
        
        # 获取query+负文档的分数
        score_neg = self.forward(query_doc_neg_input_ids, query_doc_neg_attention_mask)
        
        return score_pos, score_neg
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        os.makedirs(save_directory, exist_ok=True)
        # 保存BERT模型
        self.bert.save_pretrained(save_directory)
        # 保存自定义层的权重
        torch.save({
            'rank_head.weight': self.rank_head.weight,
            'rank_head.bias': self.rank_head.bias
        }, os.path.join(save_directory, 'custom_layers.bin'))
    
    @classmethod
    def from_pretrained(cls, save_directory):
        """加载模型"""
        # 初始化模型
        model = cls(save_directory)
        # 加载自定义层的权重
        custom_layers_path = os.path.join(save_directory, 'custom_layers.bin')
        if os.path.exists(custom_layers_path):
            custom_layers = torch.load(custom_layers_path)
            model.rank_head.weight.data = custom_layers['rank_head.weight']
            model.rank_head.bias.data = custom_layers['rank_head.bias']
        return model

class PairwiseLoss(nn.Module):
    """成对损失函数"""
    def __init__(self, margin=1.0):
        super(PairwiseLoss, self).__init__()
        self.margin = margin
    
    def forward(self, score_pos, score_neg, weight_pos, weight_neg):
        """计算成对损失"""
        # 计算基本的hinge损失
        loss = nn.functional.relu(self.margin - (score_pos - score_neg))
        
        # 应用权重
        weights = (weight_pos + weight_neg) / 2  # 使用平均权重
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, margin=1.0, mse_weight=0.5, pairwise_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.pairwise_loss = PairwiseLoss(margin)
        self.mse_weight = mse_weight
        self.pairwise_weight = pairwise_weight
    
    def forward(self, score_pos, score_neg, rel_pos, rel_neg, weight_pos, weight_neg):
        """计算组合损失"""
        # 计算MSE损失
        mse_pos = self.mse_loss(score_pos, rel_pos) * weight_pos
        mse_neg = self.mse_loss(score_neg, rel_neg) * weight_neg
        mse_total = (mse_pos.mean() + mse_neg.mean()) / 2
        
        # 计算成对损失
        pairwise_total = self.pairwise_loss(score_pos, score_neg, weight_pos, weight_neg)
        
        # 组合两种损失
        total_loss = self.mse_weight * mse_total + self.pairwise_weight * pairwise_total
        
        return total_loss

class RankingTrainer:
    """排序模型训练器"""
    def __init__(self, model, tokenizer, device, loss_fn):
        """
        初始化训练器
        Args:
            model: 模型
            tokenizer: 分词器
            device: 设备
            loss_fn: 损失函数
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.loss_fn = loss_fn
    
    def train_step(self, batch, optimizer, scheduler):
        """执行单个训练步骤"""
        self.model.train()  # 确保模型处于训练模式
        
        # 移动数据到设备
        query_doc_pos_input_ids = batch["query_doc_pos_input_ids"].to(self.device)
        query_doc_pos_attention_mask = batch["query_doc_pos_attention_mask"].to(self.device)
        query_doc_neg_input_ids = batch["query_doc_neg_input_ids"].to(self.device)
        query_doc_neg_attention_mask = batch["query_doc_neg_attention_mask"].to(self.device)
        rel_pos = batch["rel_pos"].to(self.device)
        rel_neg = batch["rel_neg"].to(self.device)
        weight_pos = batch["weight_pos"].to(self.device)
        weight_neg = batch["weight_neg"].to(self.device)
        
        # 前向传播
        score_pos, score_neg = self.model.compute_pair_score(
            query_doc_pos_input_ids, query_doc_pos_attention_mask,
            query_doc_neg_input_ids, query_doc_neg_attention_mask
        )
        
        # 计算损失
        if isinstance(self.loss_fn, CombinedLoss):
            loss = self.loss_fn(score_pos, score_neg, rel_pos, rel_neg, weight_pos, weight_neg)
        else:
            loss = self.loss_fn(score_pos, score_neg, weight_pos, weight_neg)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()  # 设置训练模式
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # 注意：train_step内部也会设置训练模式，但这里先设置确保一致性
            loss = self.train_step(batch, optimizer, scheduler)
            total_loss += loss
        
        return total_loss / len(dataloader)
    
    def save_checkpoint_with_eval(self, output_dir, step, model, tokenizer, train_loss, val_dataloader):
        """保存检查点并评估模型"""
        # 评估当前模型
        val_loss, val_ndcg = self.evaluate(val_dataloader)
        
        # 保存检查点
        checkpoint_dir = save_checkpoint(
            output_dir, 
            step, 
            model, 
            tokenizer, 
            train_loss, 
            val_loss
        )
        
        logger.info(f"保存检查点 (step {step}): {checkpoint_dir}")
        logger.info(f"当前评估结果: val_loss={val_loss:.4f}, NDCG@10={val_ndcg:.4f}")
        
        return checkpoint_dir, val_loss, val_ndcg
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_weights = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 移动数据到设备
                query_doc_pos_input_ids = batch["query_doc_pos_input_ids"].to(self.device)
                query_doc_pos_attention_mask = batch["query_doc_pos_attention_mask"].to(self.device)
                query_doc_neg_input_ids = batch["query_doc_neg_input_ids"].to(self.device)
                query_doc_neg_attention_mask = batch["query_doc_neg_attention_mask"].to(self.device)
                rel_pos = batch["rel_pos"].to(self.device)
                rel_neg = batch["rel_neg"].to(self.device)
                weight_pos = batch["weight_pos"].to(self.device)
                weight_neg = batch["weight_neg"].to(self.device)
                
                # 前向传播
                score_pos, score_neg = self.model.compute_pair_score(
                    query_doc_pos_input_ids, query_doc_pos_attention_mask,
                    query_doc_neg_input_ids, query_doc_neg_attention_mask
                )
                
                # 计算损失
                if isinstance(self.loss_fn, CombinedLoss):
                    loss = self.loss_fn(score_pos, score_neg, rel_pos, rel_neg, weight_pos, weight_neg)
                else:
                    loss = self.loss_fn(score_pos, score_neg, weight_pos, weight_neg)
                
                total_loss += loss.item()
                
                # 收集预测结果和真实值
                all_predictions.extend(torch.cat([score_pos, score_neg]).cpu().numpy())
                all_targets.extend(torch.cat([rel_pos, rel_neg]).cpu().numpy())
                all_weights.extend(torch.cat([weight_pos, weight_neg]).cpu().numpy())
        
        # 计算NDCG@10
        ndcg = self.compute_ndcg_at_k(all_predictions, all_targets, k=10)
        
        return total_loss / len(dataloader), ndcg
    
    def compute_ndcg_at_k(self, predictions, targets, k=10):
        """计算NDCG@k"""
        # 重塑为2D数组
        predictions_2d = np.array(predictions).reshape(1, -1)
        targets_2d = np.array(targets).reshape(1, -1)
        
        try:
            return ndcg_score(targets_2d, predictions_2d, k=min(k, len(predictions)))
        except:
            return 0.0

def save_checkpoint(output_dir, step, model, tokenizer, train_loss, val_loss):
    """保存模型检查点"""
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


def list_checkpoints(output_dir):
    """列出所有检查点"""
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


def select_best_checkpoint(output_dir):
    """选择验证损失最小的检查点"""
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
    """训练函数"""
    logger.info("=" * 50)
    logger.info("开始训练流程")
    logger.info("=" * 50)
    
    # 检查数据路径
    if not os.path.exists(args.train_path):
        logger.error(f"训练集不存在: {args.train_path}")
        return
    if not os.path.exists(args.val_path):
        logger.error(f"验证集不存在: {args.val_path}")
        return
    
    # 记录训练参数
    logger.info(f"训练参数: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    logger.info(f"模型: {args.model_name}, 设备: {args.device}")
    logger.info(f"损失类型: {args.loss_type}, 保存路径: {args.save_path}")
    logger.info(f"模型保存策略: {'按固定步数保存' if args.save_steps > 0 else '按epoch保存'} {f'({args.save_steps}步)' if args.save_steps > 0 else ''}")
    
    # 初始化模型组件
    logger.info("初始化模型组件...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertRankingModel(args.model_name)
    
    # 创建数据集和数据加载器
    logger.info("创建数据集...")
    train_dataset = RankingDataset(args.train_path, tokenizer, args.max_length, args.max_pairs_per_query)
    val_dataset = RankingDataset(args.val_path, tokenizer, args.max_length, args.max_pairs_per_query)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"数据加载器创建完成: train_batches={len(train_dataloader)}, val_batches={len(val_dataloader)}")
    
    # 初始化损失函数、优化器和调度器
    if args.loss_type == 'combined':
        loss_fn = CombinedLoss(margin=args.margin, mse_weight=args.mse_weight, pairwise_weight=args.pairwise_weight)
    else:
        loss_fn = PairwiseLoss(margin=args.margin)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 初始化训练器
    trainer = RankingTrainer(model, tokenizer, args.device, loss_fn)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 训练循环变量
    global_step = 0
    best_val_loss = float('inf')
    
    # 开始训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"开始 Epoch {epoch + 1}/{args.num_epochs}")
        
        # 判断是否按步数保存
        if args.save_steps > 0:
            # 按步数保存模式
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                # 执行训练步骤
                loss = trainer.train_step(batch, optimizer, scheduler)
                global_step += 1
                
                # 检查是否需要保存模型
                if global_step % args.save_steps == 0:
                    checkpoint_dir, val_loss, val_ndcg = trainer.save_checkpoint_with_eval(
                        args.save_path, global_step, model, tokenizer, loss, val_dataloader
                    )
                    
                    # 更新最佳验证损失
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"找到新的最佳模型: val_loss={best_val_loss:.4f}")
        else:
            # 按epoch保存模式
            # 训练一个epoch
            train_loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
            
            # 验证并保存
            global_step += len(train_dataloader)  # 每个epoch的步数
            checkpoint_dir, val_loss, val_ndcg = trainer.save_checkpoint_with_eval(
                args.save_path, global_step, model, tokenizer, train_loss, val_dataloader
            )
            
            logger.info(f"Epoch {epoch + 1} 完成: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, NDCG@10={val_ndcg:.4f}")
            
            # 更新最佳验证损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"找到新的最佳模型: val_loss={best_val_loss:.4f}")
    
    # 训练完成后，再次评估并保存最终模型
    final_checkpoint_dir, final_val_loss, final_val_ndcg = trainer.save_checkpoint_with_eval(
        args.save_path, global_step, model, tokenizer, 0.0, val_dataloader
    )
    logger.info(f"保存最终检查点: {final_checkpoint_dir}")
    logger.info(f"最终评估结果: val_loss={final_val_loss:.4f}, NDCG@10={final_val_ndcg:.4f}")
    
    logger.info("=" * 50)
    logger.info("训练流程完成")
    logger.info("=" * 50)

def evaluate_fn(args):
    """评估函数"""
    logger.info("=" * 50)
    logger.info("开始评估流程")
    logger.info("=" * 50)
    
    # 检查评估数据路径
    if not os.path.exists(args.eval_path):
        logger.error(f"评估集不存在: {args.eval_path}")
        return
    
    # 选择检查点
    if args.checkpoint_dir:
        # 使用指定的检查点
        checkpoint_path = args.checkpoint_dir
        logger.info(f"评估指定检查点: {checkpoint_path}")
    elif args.select_best:
        # 选择最佳检查点
        best_ckpt = select_best_checkpoint(args.save_path)
        if best_ckpt is None:
            logger.error(f"在 {args.save_path} 中未找到检查点")
            return
        checkpoint_path = best_ckpt[0]
        logger.info(f"评估最佳检查点: {checkpoint_path}")
    else:
        # 评估所有检查点
        checkpoints = list_checkpoints(args.save_path)
        if not checkpoints:
            logger.error(f"在 {args.save_path} 中未找到检查点")
            return
        
        logger.info(f"找到 {len(checkpoints)} 个检查点")
        
        # 初始化tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        
        # 创建评估数据集和数据加载器
        eval_dataset = RankingDataset(args.eval_path, tokenizer, args.max_length, args.max_pairs_per_query)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 评估所有检查点
        results = []
        for checkpoint_path, _ in checkpoints:
            try:
                # 加载模型
                model = BertRankingModel.from_pretrained(checkpoint_path)
                
                # 初始化损失函数
                if args.loss_type == 'combined':
                    loss_fn = CombinedLoss(margin=args.margin, mse_weight=args.mse_weight, pairwise_weight=args.pairwise_weight)
                else:
                    loss_fn = PairwiseLoss(margin=args.margin)
                
                # 初始化训练器
                trainer = RankingTrainer(model, tokenizer, args.device, loss_fn)
                
                # 评估
                loss, ndcg = trainer.evaluate(eval_dataloader)
                
                results.append({
                    "checkpoint": checkpoint_path,
                    "loss": float(loss),
                    "ndcg@10": float(ndcg)
                })
                
                logger.info(f"评估 {checkpoint_path}: loss={loss:.4f}, NDCG@10={ndcg:.4f}")
            except Exception as e:
                logger.error(f"评估 {checkpoint_path} 时出错: {str(e)}")
        
        # 保存结果
        if results:
            results.sort(key=lambda x: (x["loss"], -x["ndcg@10"]))
            
            summary_path = os.path.join(args.save_path, "evaluation_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "all": results,
                    "best": results[0]
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存到: {summary_path}")
            logger.info(f"最佳模型: {results[0]['checkpoint']} (loss={results[0]['loss']:.4f}, NDCG@10={results[0]['ndcg@10']:.4f})")
        
        logger.info("=" * 50)
        logger.info("评估流程完成")
        logger.info("=" * 50)
        return
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # 加载模型
    model = BertRankingModel.from_pretrained(checkpoint_path)
    
    # 初始化损失函数
    if args.loss_type == 'combined':
        loss_fn = CombinedLoss(margin=args.margin, mse_weight=args.mse_weight, pairwise_weight=args.pairwise_weight)
    else:
        loss_fn = PairwiseLoss(margin=args.margin)
    
    # 创建评估数据集和数据加载器
    eval_dataset = RankingDataset(args.eval_path, tokenizer, args.max_length, args.max_pairs_per_query)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化训练器
    trainer = RankingTrainer(model, tokenizer, args.device, loss_fn)
    
    # 评估
    loss, ndcg = trainer.evaluate(eval_dataloader)
    
    logger.info(f"评估结果: loss={loss:.4f}, NDCG@10={ndcg:.4f}")
    
    logger.info("=" * 50)
    logger.info("评估流程完成")
    logger.info("=" * 50)

def deploy_fn(args):
    """部署模型为FastAPI服务"""
    logger.info("=" * 50)
    logger.info("开始部署流程")
    logger.info("=" * 50)
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        logger.info("FastAPI依赖检查通过")
    except Exception as e:
        logger.error(f"FastAPI依赖未安装: {e}")
        return
    
    # 加载模型
    if not args.model_dir:
        # 选择最佳检查点
        best_ckpt = select_best_checkpoint(args.save_path)
        if best_ckpt is None:
            logger.error(f"在 {args.save_path} 中未找到检查点")
            return
        args.model_dir = best_ckpt[0]
        logger.info(f"选择最佳检查点: {args.model_dir}")
    
    logger.info(f"加载模型: {args.model_dir}")
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertRankingModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    
    # 创建FastAPI应用
    app = FastAPI(title="BERT Ranking Service", version="1.0.0")
    
    # 定义请求模型
    class ScoreRequest(BaseModel):
        query: str
        document: str
    
    class BatchScoreRequest(BaseModel):
        query: str
        documents: list[str]
    
    # 健康检查端点
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    # 评分端点
    @app.post("/score")
    def score(req: ScoreRequest):
        with torch.no_grad():
            # 拼接query和文档
            query_doc = f"{req.query} [SEP] {req.document}"
            encoding = tokenizer(query_doc, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
            
            score = model(
                encoding['input_ids'], 
                encoding['attention_mask']
            ).item()
            return {"score": score}
    
    # 批量评分端点
    @app.post("/score_batch")
    def score_batch(req: BatchScoreRequest):
        with torch.no_grad():
            # 处理多个文档
            scores = []
            for doc in req.documents:
                # 拼接query和文档
                query_doc = f"{req.query} [SEP] {doc}"
                encoding = tokenizer(query_doc, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
                
                score = model(
                    encoding['input_ids'], 
                    encoding['attention_mask']
                ).item()
                scores.append(score)
            
            # 排序结果
            ranked_results = sorted(
                zip(req.documents, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                "results": [
                    {"document": doc, "score": score}
                    for doc, score in ranked_results
                ]
            }
    
    logger.info(f"启动API服务: {args.host}:{args.port}")
    logger.info("=" * 50)
    logger.info("部署流程完成，服务已启动")
    logger.info("=" * 50)
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT排序模型微调脚本')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, default='data/rank_train.json', help='训练数据路径')
    parser.add_argument('--val_path', type=str, default='data/rank_valid.json', help='验证数据路径')
    parser.add_argument('--eval_path', type=str, default='data/rank_test.json', help='评估数据路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='BERT模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--max_pairs_per_query', type=int, default=100, help='每个查询的最大成对数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--save_path', type=str, default='./output/ranking_model', help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--save_steps', type=int, default=2, help='保存模型的步数间隔，0表示每个epoch结束时保存')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='pairwise', choices=['pairwise', 'combined'], help='损失函数类型')
    parser.add_argument('--margin', type=float, default=1.0, help='边界值')
    parser.add_argument('--mse_weight', type=float, default=0.5, help='MSE损失的权重')
    parser.add_argument('--pairwise_weight', type=float, default=0.5, help='成对损失的权重')
    
    # 评估参数
    parser.add_argument('--checkpoint_dir', type=str, default='', help='指定检查点目录')
    parser.add_argument('--select_best', action='store_true', help='选择最佳检查点')
    
    # 部署参数
    parser.add_argument('--model_dir', type=str, default='', help='部署模型目录')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    parser.add_argument('--port', type=int, default=8002, help='服务端口')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'deploy'], help='运行模式')
    
    args = parser.parse_args()
    
    logger.info("BERT排序模型微调脚本启动")
    logger.info(f"运行模式: {args.mode}")
    
    # 根据模式执行不同的函数
    if args.mode == 'train':
        train_fn(args)
    elif args.mode == 'evaluate':
        evaluate_fn(args)
    elif args.mode == 'deploy':
        deploy_fn(args)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"排序模型微调错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)