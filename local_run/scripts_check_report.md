# LLaMA-Factory 本地运行脚本检查报告

## 📋 脚本概览

### 1. 主要脚本文件

| 脚本文件 | 功能 | 状态 | 路径配置 |
|---------|------|------|----------|
| `llamafactory_local_runner.py` | 统一运行器（训练/推理/API/WebUI） | ✅ 正常 | 已优化 |
| `fastapi_service.py` | FastAPI 服务部署 | ✅ 正常 | 已优化 |

### 2. 配置文件

| 配置文件 | 用途 | 路径配置 |
|---------|------|----------|
| `configs/runner_config.yaml` | 统一运行配置 | ✅ 正常 |
| `configs/llama3.2_lora_sft_local.yaml` | LoRA 微调配置 | ✅ 正常 |
| `configs/qwen2.5_lora_sft_local.yaml` | Qwen 微调配置 | ✅ 正常 |

## 🔧 功能模块检查

### 1. 预训练 (Pretraining)
- **状态**: ❌ 未配置
- **说明**: 当前配置主要针对微调，预训练需要单独配置
- **建议**: 如需预训练，需要创建专门的预训练配置文件

### 2. 微调 (Fine-tuning)
- **状态**: ✅ 已配置
- **配置文件**: `configs/llama3.2_lora_sft_local.yaml`
- **路径配置**:
  ```yaml
  model_name_or_path: models/Llama-3.2-1B-Instruct
  output_dir: saves/llama3.2-1b-lora-sft
  ```
- **支持方法**: LoRA 微调
- **启动命令**: `python llamafactory_local_runner.py`

### 3. 模型合并 (Model Merging)
- **状态**: ✅ 已配置
- **配置位置**: `configs/runner_config.yaml` (第118-131行)
- **路径配置**:
  ```yaml
  model_path: models/Llama-3.2-1B-Instruct
  adapter_path: saves/llama3-1b/lora/sft
  export_dir: output/merged_model
  ```
- **启动方式**: 修改 `runner_config.yaml` 中的 command 为 `export`

### 4. 部署 (Deployment)
- **状态**: ✅ 已配置
- **部署方式**:
  1. **FastAPI 服务**: `python fastapi_service.py`
  2. **Web UI**: 通过 `llamafactory_local_runner.py` 启动
  3. **API 服务**: 通过 `llamafactory_local_runner.py` 启动

## 📁 路径配置检查

### 1. 模型路径
```python
# fastapi_service.py 中的路径配置
model_path = LOCAL_RUN_DIR / "models" / "Llama-3.2-1B-Instruct"
adapter_path = LOCAL_RUN_DIR / "saves" / "llama3.2-1b-lora-sft"
```

### 2. 配置文件路径
```python
# llamafactory_local_runner.py 中的路径配置
LOCAL_RUN_DIR = THIS_FILE.parent  # local_run 目录
PROJECT_ROOT = LOCAL_RUN_DIR.parent  # LLaMA-Factory 根目录
```

### 3. 输出路径
- **训练输出**: `saves/llama3.2-1b-lora-sft/`
- **日志文件**: `logs/`
- **缓存目录**: `hf_cache/`

## 🚀 启动命令汇总

### 1. 训练 (LoRA 微调)
```bash
conda activate llamafactory
python llamafactory_local_runner.py
```

### 2. 聊天 (CLI)
```bash
# 修改 runner_config.yaml 中的 command 为 chat
python llamafactory_local_runner.py
```

### 3. Web UI
```bash
# 修改 runner_config.yaml 中的 command 为 webui
python llamafactory_local_runner.py
```

### 4. API 服务
```bash
# 方法1: 使用 FastAPI 服务
python fastapi_service.py

# 方法2: 使用原生 API
# 修改 runner_config.yaml 中的 command 为 api
python llamafactory_local_runner.py
```

### 5. 模型合并
```bash
# 修改 runner_config.yaml 中的 command 为 export
python llamafactory_local_runner.py
```

## ⚠️ 注意事项

### 1. 路径一致性
- ✅ 所有脚本的路径配置已统一
- ✅ 模型路径: `models/Llama-3.2-1B-Instruct`
- ✅ 适配器路径: `saves/llama3.2-1b-lora-sft`

### 2. 环境要求
- Python 3.10+
- conda 环境: `llamafactory`
- 依赖包版本已修复

### 3. 配置文件
- 默认使用 `configs/runner_config.yaml`
- 可通过修改配置文件切换不同功能
- 支持零参数启动

## 🔍 建议改进

### 1. 添加预训练配置
- 创建预训练专用的 YAML 配置文件
- 配置预训练数据集和参数

### 2. 添加评估脚本
- 创建模型评估配置文件
- 支持多种评估指标

### 3. 添加批量处理脚本
- 支持批量训练多个模型
- 支持批量评估和测试

### 4. 添加监控脚本
- 训练过程监控
- 资源使用监控
