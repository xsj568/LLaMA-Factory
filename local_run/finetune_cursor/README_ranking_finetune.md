# BERT排序微调脚本使用说明

## 概述

`ranking_finetune.py` 是一个基于BERT的排序微调脚本，专门用于提升query-document排序学习的NDCG@k指标。脚本提供统一的三段式流程：训练、评估、部署（FastAPI）。支持两种损失函数：MSE+Pairwise加权损失和纯Pairwise损失，通过排序对学习来优化文档排序效果。

## 功能特性

- **模型**: 基于BERT的排序模型，包含专门的排序头
- **序列长度**: 支持最大512个token
- **数据格式**: query, document, rel, weight
- **损失函数**: 
  - `combined`: MSE + Pairwise加权损失
  - `pairwise`: 纯Pairwise损失
- **优化目标**: NDCG@k指标
- **排序对生成**: 支持按query分组和随机配对两种模式

## 数据格式

训练数据需要是JSON或JSONL格式，每条记录包含以下字段：

```json
{
    "query": "查询文本",
    "document": "文档文本", 
    "rel": 0.9,
    "weight": 1.0
}
```

字段说明：
- `query`: 查询文本
- `document`: 文档文本
- `rel`: 相关性分数 (0-1之间)
- `weight`: 样本权重 (用于损失函数加权)

## 模型架构

### BertRankingModel
- 基于预训练BERT模型
- 使用[CLS] token作为句子向量
- 包含专门的排序头：`Linear(embedding_dim*2) -> ReLU -> Dropout -> Linear(1)`
- 通过拼接query和document向量计算排序分数

### 损失函数

#### 1. PairwiseLoss
```python
loss = max(0, margin - (pos_score - neg_score))
```
- 基于hinge损失的pairwise排序损失
- 确保正样本分数高于负样本分数至少margin

#### 2. CombinedLoss
```python
total_loss = mse_weight * mse_loss + pairwise_weight * pairwise_loss
```
- MSE损失：直接优化预测分数与真实分数的差异
- Pairwise损失：优化排序关系
- 支持权重调整两种损失的贡献

## 排序对生成策略

### 1. 按Query分组 (推荐)
- 将相同query的文档分组
- 按相关性分数排序
- 创建所有可能的正负样本对
- 适合有多个相关文档的query

### 2. 随机配对
- 随机选择文档对
- 根据相关性分数确定正负样本
- 适合数据量较小的情况

## 安装依赖

```bash
pip install torch transformers scikit-learn tqdm
```

## 使用方法

### 1. 训练（组合损失）

```bash
python ranking_finetune.py \
    --mode train \
    --train_path local_run/finetune_cursor/data/rank_train.json \
    --val_path local_run/finetune_cursor/data/rank_test.json \
    --loss_type combined \
    --mse_weight 0.3 \
    --pairwise_weight 0.7 \
    --group_field key \
    --num_epochs 5 \
    --save_steps 200 \
    --save_path ./output/ranking_model
```

### 2. 训练（纯Pairwise损失）

```bash
python ranking_finetune.py \
    --mode train \
    --train_path local_run/finetune_cursor/data/rank_train.json \
    --val_path local_run/finetune_cursor/data/rank_test.json \
    --loss_type pairwise \
    --margin 1.0 \
    --group_field key \
    --num_epochs 5 \
    --save_steps 200 \
    --save_path ./output/ranking_model
```

### 3. 评估

评估所有 checkpoints 并自动选择最优（val_loss 最小）：

```bash
python ranking_finetune.py \
  --mode evaluate \
  --eval_path local_run/finetune_cursor/data/rank_valid.json \
  --save_path ./output/ranking_model \
  --select_best
```

也可指定单个 checkpoint 目录：

```bash
python ranking_finetune.py --mode evaluate --eval_path local_run/finetune_cursor/data/rank_valid.json --checkpoint_dir ./output/ranking_model/checkpoint-step-2000
```

### 4. 部署

未指定 `--model_dir` 时自动选择最优 checkpoint：

```bash
python ranking_finetune.py --mode deploy --save_path ./output/ranking_model --host 0.0.0.0 --port 8000
```

### 5. 参数说明（核心）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_path` | data/rank_train.json | 训练集路径 |
| `--val_path` | data/rank_test.json | 验证集路径 |
| `--eval_path` | data/rank_valid.json | 评估集路径 |
| `--model_name` | bert-base-uncased | BERT模型名称 |
| `--max_length` | 512 | 最大序列长度 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--num_epochs` | 3 | 训练轮数 |
| `--loss_type` | combined | 损失函数类型 |
| `--mse_weight` | 0.5 | MSE损失权重 |
| `--pairwise_weight` | 0.5 | Pairwise损失权重 |
| `--margin` | 1.0 | Pairwise损失边界值 |
| `--max_pairs_per_query` | 100 | 每个query的最大pair数量 |
| `--group_field` | key | 分组字段（默认 key，缺失回退 query） |
| `--save_steps` | 200 | 每多少步保存一次 checkpoint |
| `--save_path` | ./output/ranking_model | 模型保存根目录 |

## 评估指标

### NDCG@k (归一化折损累积增益)
- 评估排序质量的标准指标
- 考虑排序位置的重要性
- 值越高表示排序效果越好
- 默认计算NDCG@10

### 训练监控
- 总损失
- MSE损失 (当使用combined时)
- Pairwise损失 (当使用combined时)
- NDCG@k指标

## 输出文件

训练完成后，模型将保存到指定路径，包含：
- `config.json`: 模型配置
- `pytorch_model.bin`: 模型权重
- `tokenizer.json`: 分词器配置
- `tokenizer_config.json`: 分词器参数

## 使用训练好的模型

```python
from transformers import BertTokenizer
from ranking_finetune import BertRankingModel

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('./output/ranking_model')
model = BertRankingModel.from_pretrained('./output/ranking_model')

# 编码文本
query = "What is machine learning?"
document = "Machine learning is a subset of AI..."

# 获取向量表示
query_vector = model.encode(query)
doc_vector = model.encode(document)

# 计算排序分数
ranking_score = model.compute_ranking_score(query_vector, doc_vector)
```

## 训练策略建议

### 1. 损失函数选择
- **combined**: 适合需要同时优化绝对分数和排序关系的场景
- **pairwise**: 适合只关注排序关系的场景

### 2. 权重调整
- MSE权重高：更关注分数预测准确性
- Pairwise权重高：更关注排序关系

### 3. 数据准备
- 确保每个query有多个不同相关性的文档
- 相关性分数标注要准确
- 使用`--group_by_query`获得更好的排序对

### 4. 超参数调优
- 学习率：从2e-5开始，根据收敛情况调整
- margin：影响pairwise损失的敏感度
- batch_size：根据GPU内存调整

## 注意事项

1. **数据质量**: 相关性分数标注质量直接影响模型性能
2. **排序对数量**: 过多的排序对可能导致训练时间过长
3. **内存使用**: 排序对会显著增加内存使用量
4. **收敛性**: Pairwise损失可能需要更多epoch才能收敛
5. **评估频率**: 建议每个epoch都进行评估以监控NDCG变化

## 扩展功能

可以根据需要扩展以下功能：
- 支持更多预训练模型 (RoBERTa, DeBERTa等)
- 实现Listwise损失函数
- 添加更多评估指标 (MRR, MAP等)
- 支持多GPU训练
- 实现负采样策略
- 添加早停机制
- 支持动态margin调整

## 故障排除

1. **CUDA内存不足**: 减小batch_size或max_pairs_per_query
2. **NDCG不提升**: 检查数据质量，调整损失权重
3. **训练不收敛**: 降低学习率，增加训练轮数
4. **排序对过少**: 使用--group_by_query或增加数据量
5. **模型加载失败**: 确保transformers版本兼容
