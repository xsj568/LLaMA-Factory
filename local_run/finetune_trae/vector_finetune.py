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
        logging.FileHandler("vector_finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VectorFinetune")

class VectorDataset(Dataset):
    """向量化模型的数据集类"""
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        logger.info(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"数据加载完成，共 {len(self.data)} 条样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data[idx]
        
        # 处理查询
        query_encoding = self.tokenizer(
            sample["query"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # 处理文档
        doc_encoding = self.tokenizer(
            sample["document"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # 处理可选的key字段
        key = sample.get("key", f"sample_{idx}")
        
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "doc_input_ids": doc_encoding["input_ids"].squeeze(0),
            "doc_attention_mask": doc_encoding["attention_mask"].squeeze(0),
            "rel": torch.tensor(sample["rel"], dtype=torch.float32),
            "weight": torch.tensor(sample["weight"], dtype=torch.float32),
            "key": key
        }

class BertVectorModel(nn.Module):
    """BERT向量化模型"""
    def __init__(self, model_name):
        """
        初始化模型
        Args:
            model_name: BERT模型名称
        """
        super(BertVectorModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 添加一个线性层用于向量投影
        self.projection = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 应用线性投影和层归一化
        projected = self.layer_norm(self.projection(cls_output))
        # L2归一化
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        os.makedirs(save_directory, exist_ok=True)
        # 保存BERT模型
        self.bert.save_pretrained(save_directory)
        # 保存自定义层的权重
        torch.save({
            'projection.weight': self.projection.weight,
            'projection.bias': self.projection.bias,
            'layer_norm.weight': self.layer_norm.weight,
            'layer_norm.bias': self.layer_norm.bias
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
            model.projection.weight.data = custom_layers['projection.weight']
            model.projection.bias.data = custom_layers['projection.bias']
            model.layer_norm.weight.data = custom_layers['layer_norm.weight']
            model.layer_norm.bias.data = custom_layers['layer_norm.bias']
        return model

class WeightedMSELoss(nn.Module):
    """带权重的均方差损失函数"""
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, outputs, targets, weights):
        """计算带权重的均方差损失"""
        # 计算原始MSE损失
        mse = self.mse_loss(outputs, targets)
        # 应用权重
        weighted_mse = mse * weights
        # 返回平均损失
        return weighted_mse.mean()

class VectorTrainer:
    """向量化模型训练器"""
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
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # 移动数据到设备
            query_input_ids = batch["query_input_ids"].to(self.device)
            query_attention_mask = batch["query_attention_mask"].to(self.device)
            doc_input_ids = batch["doc_input_ids"].to(self.device)
            doc_attention_mask = batch["doc_attention_mask"].to(self.device)
            rel = batch["rel"].to(self.device)
            weight = batch["weight"].to(self.device)
            
            # 前向传播
            query_vectors = self.model(query_input_ids, query_attention_mask)
            doc_vectors = self.model(doc_input_ids, doc_attention_mask)
            
            # 计算余弦相似度
            cosine_similarity = torch.sum(query_vectors * doc_vectors, dim=1)
            
            # 计算损失
            loss = self.loss_fn(cosine_similarity, rel, weight)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
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
                query_input_ids = batch["query_input_ids"].to(self.device)
                query_attention_mask = batch["query_attention_mask"].to(self.device)
                doc_input_ids = batch["doc_input_ids"].to(self.device)
                doc_attention_mask = batch["doc_attention_mask"].to(self.device)
                rel = batch["rel"].to(self.device)
                weight = batch["weight"].to(self.device)
                
                # 前向传播
                query_vectors = self.model(query_input_ids, query_attention_mask)
                doc_vectors = self.model(doc_input_ids, doc_attention_mask)
                
                # 计算余弦相似度
                cosine_similarity = torch.sum(query_vectors * doc_vectors, dim=1)
                
                # 计算损失
                loss = self.loss_fn(cosine_similarity, rel, weight)
                total_loss += loss.item()
                
                # 收集预测结果和真实值
                all_predictions.extend(cosine_similarity.cpu().numpy())
                all_targets.extend(rel.cpu().numpy())
                all_weights.extend(weight.cpu().numpy())
        
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
    
    logger.info(f"训练参数: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    logger.info(f"模型: {args.model_name}, 设备: {args.device}")
    logger.info(f"保存路径: {args.save_path}")
    
    # 初始化tokenizer和模型
    logger.info("初始化tokenizer和模型...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertVectorModel(args.model_name)
    
    # 创建数据集和数据加载器
    logger.info("创建数据集...")
    train_dataset = VectorDataset(args.train_path, tokenizer, args.max_length)
    val_dataset = VectorDataset(args.val_path, tokenizer, args.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"数据加载器创建完成: train_batches={len(train_dataloader)}, val_batches={len(val_dataloader)}")
    
    # 初始化损失函数、优化器和调度器
    loss_fn = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 初始化训练器
    trainer = VectorTrainer(model, tokenizer, args.device, loss_fn)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"开始 Epoch {epoch + 1}/{args.num_epochs}")
        
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_dataloader, optimizer, scheduler)
        
        # 验证
        val_loss, val_ndcg = trainer.evaluate(val_dataloader)
        
        logger.info(f"Epoch {epoch + 1} 完成: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, NDCG@10={val_ndcg:.4f}")
        
        # 保存检查点
        global_step += len(train_dataloader)
        checkpoint_dir = save_checkpoint(
            args.save_path, 
            global_step, 
            model, 
            tokenizer, 
            train_loss, 
            val_loss
        )
        
        logger.info(f"保存检查点: {checkpoint_dir}")
        
        # 更新最佳验证损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"找到新的最佳模型: val_loss={best_val_loss:.4f}")
    
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
        eval_dataset = VectorDataset(args.eval_path, tokenizer, args.max_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 评估所有检查点
        results = []
        for checkpoint_path, _ in checkpoints:
            try:
                # 加载模型
                model = BertVectorModel.from_pretrained(checkpoint_path)
                
                # 初始化训练器
                loss_fn = WeightedMSELoss()
                trainer = VectorTrainer(model, tokenizer, args.device, loss_fn)
                
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
    model = BertVectorModel.from_pretrained(checkpoint_path)
    
    # 创建评估数据集和数据加载器
    eval_dataset = VectorDataset(args.eval_path, tokenizer, args.max_length)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化训练器
    loss_fn = WeightedMSELoss()
    trainer = VectorTrainer(model, tokenizer, args.device, loss_fn)
    
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
    model = BertVectorModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    
    # 创建FastAPI应用
    app = FastAPI(title="BERT Vector Service", version="1.0.0")
    
    # 定义请求模型
    class EmbedRequest(BaseModel):
        text: str
    
    class BatchEmbedRequest(BaseModel):
        texts: list[str]
    
    class ScoreRequest(BaseModel):
        query: str
        document: str
    
    # 健康检查端点
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    # 嵌入端点
    @app.post("/embed")
    def embed(req: EmbedRequest):
        with torch.no_grad():
            encoding = tokenizer(req.text, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
            vector = model(encoding['input_ids'], encoding['attention_mask']).squeeze().cpu().tolist()
            return {"embedding": vector}
    
    # 批量嵌入端点
    @app.post("/embed_batch")
    def embed_batch(req: BatchEmbedRequest):
        with torch.no_grad():
            encodings = tokenizer(req.texts, max_length=args.max_length, padding=True, truncation=True, return_tensors='pt').to(args.device)
            vectors = model(encodings['input_ids'], encodings['attention_mask']).cpu().tolist()
            return {"embeddings": vectors}
    
    # 评分端点
    @app.post("/score")
    def score(req: ScoreRequest):
        with torch.no_grad():
            query_encoding = tokenizer(req.query, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
            doc_encoding = tokenizer(req.document, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt').to(args.device)
            
            query_vector = model(query_encoding['input_ids'], query_encoding['attention_mask'])
            doc_vector = model(doc_encoding['input_ids'], doc_encoding['attention_mask'])
            
            score = torch.sum(query_vector * doc_vector).item()
            return {"score": score}
    
    logger.info(f"启动API服务: {args.host}:{args.port}")
    logger.info("=" * 50)
    logger.info("部署流程完成，服务已启动")
    logger.info("=" * 50)
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT向量化模型微调脚本')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, default='data/vec_train.json', help='训练数据路径')
    parser.add_argument('--val_path', type=str, default='data/vec_valid.json', help='验证数据路径')
    parser.add_argument('--eval_path', type=str, default='data/vec_test.json', help='评估数据路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='BERT模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--save_path', type=str, default='./output/vector_model', help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    
    # 评估参数
    parser.add_argument('--checkpoint_dir', type=str, default='', help='指定检查点目录')
    parser.add_argument('--select_best', action='store_true', help='选择最佳检查点')
    
    # 部署参数
    parser.add_argument('--model_dir', type=str, default='', help='部署模型目录')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    parser.add_argument('--port', type=int, default=8001, help='服务端口')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'deploy'], help='运行模式')
    
    args = parser.parse_args()
    
    logger.info("BERT向量化模型微调脚本启动")
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
        logger.error(f"向量化模型微调错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)