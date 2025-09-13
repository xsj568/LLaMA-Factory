# LLaMA-Factory 本地运行环境

## 📁 目录结构

```
local_run/
├── README.md                    # 本说明文件
├── fastapi_service.py           # FastAPI 服务主程序
├── llamafactory_local_runner.py # 本地训练运行器
├── configs/                     # 配置文件目录
│   ├── llama3.2_lora_sft_local.yaml
│   ├── qwen2.5_lora_sft_local.yaml
│   ├── runner_config.yaml
│   └── LLaMA_Factory_API.postman_collection.json  # API 测试集合
├── docs/                        # 文档目录
│   ├── API_服务部署说明.md
│   └── LLaMA-Factory源代码深度分析.md
├── logs/                        # 日志文件目录
│   ├── fastapi_service.log
│   └── llamafactory_runner.log
├── models/                      # 模型文件目录
│   └── Llama-3.2-1B-Instruct/
├── saves/                       # 训练保存目录
│   └── llama3.2-1b-lora-sft/
└── hf_cache/                    # HuggingFace 缓存目录
```

## 🚀 快速开始

### 1. 启动 FastAPI 服务

```bash
# 手动启动
conda activate llamafactory
python fastapi_service.py
```

### 2. 服务地址

- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **模型信息**: http://localhost:8000/model/info

### 3. 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 简单聊天测试
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

## 📋 配置说明

### 模型路径配置
- **基础模型**: `models/Llama-3.2-1B-Instruct/`
- **LoRA 适配器**: `saves/llama3.2-1b-lora-sft/`

### 训练配置
- **配置文件**: `configs/llama3.2_lora_sft_local.yaml`
- **输出目录**: `saves/llama3.2-1b-lora-sft/`

## 🔧 环境要求

- Python 3.10+
- conda 环境: `llamafactory`
- 依赖包版本:
  - transformers: 4.49.0-4.55.0
  - datasets: 2.16.0-3.6.0
  - peft: 0.14.0-0.15.2
  - trl: 0.8.6-0.9.6
  - fastapi: 0.116.1
  - uvicorn: 0.35.0

## 📝 注意事项

1. 确保模型文件已正确放置在 `models/` 目录下
2. 训练后的 LoRA 适配器会保存在 `saves/` 目录下
3. 日志文件会保存在 `logs/` 目录下
4. 配置文件统一放在 `configs/` 目录下
