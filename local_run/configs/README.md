# LLaMA-Factory 配置文件说明

## 📁 配置文件结构

```
configs/
├── README.md                    # 本说明文件
├── runner_config.yaml          # 统一运行配置文件
├── llama3.2_lora_sft_local.yaml # Llama-3.2-1B LoRA 微调配置
├── qwen2.5_lora_sft_local.yaml  # Qwen-2.5 LoRA 微调配置
└── LLaMA_Factory_API.postman_collection.json # API 测试集合
```

## 🚀 快速开始

### 1. 训练模型
```bash
# 使用默认配置 (Llama-3.2-1B LoRA 微调)
python llamafactory_local_runner.py

# 或指定配置文件
python llamafactory_local_runner.py train --config configs/llama3.2_lora_sft_local.yaml
```

### 2. 聊天测试
```bash
# 修改 runner_config.yaml 中的 command 为 chat
python llamafactory_local_runner.py
```

### 3. Web UI
```bash
# 修改 runner_config.yaml 中的 command 为 webui
python llamafactory_local_runner.py
```

## ⚙️ 配置文件详解

### runner_config.yaml
统一运行配置文件，控制所有运行场景：

```yaml
# 当前激活配置
command: train  # 命令类型
config: llama3.2_lora_sft_local.yaml  # 配置文件
args:  # 全局参数
  cache_dir: hf_cache
  dataset_dir: ../data
```

**支持的命令类型：**
- `train`: 训练模型
- `chat`: CLI 聊天
- `webchat`: Web 聊天
- `api`: API 服务
- `webui`: Web UI
- `export`: 模型导出
- `eval`: 模型评估

### llama3.2_lora_sft_local.yaml
Llama-3.2-1B-Instruct 模型的 LoRA 微调配置：

**主要配置项：**
- **模型配置**: 模型路径、信任远程代码
- **微调方法**: LoRA 参数 (rank=8, alpha=16, dropout=0.1)
- **数据集配置**: 数据集、模板、截断长度
- **训练参数**: 批次大小、学习率、训练轮数
- **输出配置**: 保存路径、日志间隔

**关键参数说明：**
```yaml
lora_rank: 8          # LoRA 秩，控制参数量 (推荐: 8-16)
lora_alpha: 16        # LoRA 缩放参数，通常为 rank 的 2 倍
lora_target: q_proj,v_proj  # LoRA 作用模块
learning_rate: 1.0e-4 # 学习率 (LoRA 推荐: 1e-4 到 5e-4)
max_samples: 1000     # 最大样本数 (测试时可设为 10)
```

## 🔧 配置修改指南

### 1. 切换模型
修改 `runner_config.yaml` 中的 `config` 字段：
```yaml
config: llama3.2_lora_sft_local.yaml  # Llama 模型
# config: qwen2.5_lora_sft_local.yaml  # Qwen 模型
```

### 2. 调整训练参数
修改对应的 YAML 配置文件：
```yaml
# 快速测试
max_samples: 10
num_train_epochs: 1.0
save_steps: 2

# 正式训练
max_samples: 1000
num_train_epochs: 3.0
save_steps: 500
```

### 3. 调整 LoRA 参数
```yaml
# 轻量级配置 (参数量少)
lora_rank: 4
lora_alpha: 8
lora_target: q_proj,v_proj

# 标准配置 (推荐)
lora_rank: 8
lora_alpha: 16
lora_target: q_proj,v_proj

# 完整配置 (参数量多)
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj
```

## 📊 性能优化建议

### 1. 内存优化
```yaml
per_device_train_batch_size: 1  # 减小批次大小
gradient_accumulation_steps: 8  # 增加梯度累积
bf16: true  # 使用 bfloat16 精度
```

### 2. 训练速度优化
```yaml
dataloader_num_workers: 4  # 增加数据加载进程 (Linux/Mac)
preprocessing_num_workers: 4  # 增加预处理进程
```

### 3. 模型质量优化
```yaml
learning_rate: 5.0e-4  # 适当提高学习率
num_train_epochs: 5.0  # 增加训练轮数
warmup_ratio: 0.1  # 学习率预热
```

## 🐛 常见问题

### 1. 内存不足
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `bf16: true`

### 2. 训练速度慢
- 增加 `dataloader_num_workers`
- 减少 `max_samples` 进行测试
- 使用更小的 `lora_rank`

### 3. 模型质量差
- 增加 `num_train_epochs`
- 调整 `learning_rate`
- 使用更多训练数据

## 📝 配置模板

### 快速测试配置
```yaml
max_samples: 10
num_train_epochs: 1.0
save_steps: 2
logging_steps: 1
```

### 生产环境配置
```yaml
max_samples: 10000
num_train_epochs: 5.0
save_steps: 1000
logging_steps: 100
```

### 高性能配置
```yaml
lora_rank: 16
lora_alpha: 32
learning_rate: 5.0e-4
per_device_train_batch_size: 2
```
