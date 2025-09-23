# BERT向量化微调脚本使用说明

## 概述

`vector_finetune.py` 是一个基于BERT的向量化微调脚本，用于提升query-document相关性学习的topk召回率。脚本提供统一的三段式流程：训练、评估、部署（FastAPI）。实现了带权重的均方差损失函数，支持从本地JSON直接加载三分数据。

## 功能特性

- **模型**: 基于BERT的向量化模型
- **序列长度**: 支持最大512个token
- **数据格式**: query, document, rel, weight
- **损失函数**: 带权重的均方差损失
- **评估指标**: NDCG@k (用于评估topk召回率)
- **优化器**: AdamW + 线性学习率调度

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

## 安装依赖

```bash
pip install torch transformers scikit-learn tqdm
```

## 使用方法

### 1. 训练

```bash
python vector_finetune.py \
    --mode train \
    --train_path local_run/finetune_cursor/data/vec_train.json \
    --val_path local_run/finetune_cursor/data/vec_valid.json \
    --model_name bert-base-uncased \
    --max_length 512 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --save_steps 200 \
    --save_path ./output/vector_model
```

### 2. 评估

评估所有 checkpoints 并自动选择最优（val_loss 最小）：

```bash
python vector_finetune.py \
  --mode evaluate \
  --test_path local_run/finetune_cursor/data/vec_test.json \
  --save_path ./output/vector_model \
  --select_best
```

也可指定单个 checkpoint 目录：

```bash
python vector_finetune.py --mode evaluate --test_path local_run/finetune_cursor/data/vec_test.json --checkpoint_dir ./output/vector_model/checkpoint-step-2000
```

### 3. 部署

未指定 `--model_dir` 时自动选择最优 checkpoint：

```bash
python vector_finetune.py --mode deploy --save_path ./output/vector_model --host 0.0.0.0 --port 8000
```

### 4. 参数说明（核心）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_path` | data/vec_train.json | 训练集路径 |
| `--val_path` | data/vec_valid.json | 验证集路径 |
| `--test_path` | data/vec_test.json | 测试集路径 |
| `--model_name` | bert-base-uncased | BERT模型名称 |
| `--max_length` | 512 | 最大序列长度 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--num_epochs` | 3 | 训练轮数 |
| `--warmup_steps` | 100 | 预热步数 |
| `--save_steps` | 200 | 每多少步保存一次 checkpoint |
| `--save_path` | ./output/vector_model | 模型保存路径 |
| `--device` | cuda/cpu | 训练设备 |

## 模型架构

### BertVectorModel
- 基于预训练BERT模型
- 使用[CLS] token作为句子向量
- 支持可选的投影层调整嵌入维度
- L2归一化输出向量

### WeightedMSELoss
- 带权重的均方差损失函数
- 支持样本级别的权重调整
- 适用于不平衡数据集

## 评估指标

- **NDCG@k**: 归一化折损累积增益，用于评估排序质量
- **训练/验证损失**: 监控模型收敛情况

## 输出文件

训练完成后，模型将保存到指定路径，包含：
- `config.json`: 模型配置
- `pytorch_model.bin`: 模型权重
- `tokenizer.json`: 分词器配置
- `tokenizer_config.json`: 分词器参数

## 使用训练好的模型

```python
from transformers import BertTokenizer
from vector_finetune import BertVectorModel

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('./output/vector_model')
model = BertVectorModel.from_pretrained('./output/vector_model')

# 编码文本
query = "What is machine learning?"
document = "Machine learning is a subset of AI..."

# 获取向量表示
query_vector = model.encode(query)
doc_vector = model.encode(document)

# 计算相似度
similarity = torch.cosine_similarity(query_vector, doc_vector)
```

## 注意事项

1. **数据质量**: 确保相关性分数(rel)的标注质量，这对模型性能至关重要
2. **权重设置**: 合理设置样本权重，可以平衡不同重要性的样本
3. **序列长度**: 根据实际数据调整max_length参数
4. **批次大小**: 根据GPU内存调整batch_size
5. **学习率**: 建议从2e-5开始，根据训练效果调整

## 扩展功能

可以根据需要扩展以下功能：
- 支持更多预训练模型 (RoBERTa, DeBERTa等)
- 添加更多评估指标 (MRR, MAP等)
- 实现负采样策略
- 支持多GPU训练
- 添加早停机制

## 故障排除

1. **CUDA内存不足**: 减小batch_size或max_length
2. **数据格式错误**: 检查JSON格式和字段名称
3. **模型加载失败**: 确保transformers版本兼容
4. **训练不收敛**: 调整学习率或增加训练轮数
