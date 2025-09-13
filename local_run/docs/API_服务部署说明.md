# LLaMA-Factory API 服务部署说明

## 概述

本项目提供了基于 FastAPI 的模型推理服务，支持 Llama-3.2-1B-Instruct 模型的 LoRA 微调版本。

## 文件说明

- `fastapi_service.py` - FastAPI 服务主文件（包含环境检查和启动功能）
- `LLaMA_Factory_API.postman_collection.json` - Postman 测试集合
- `test_api.sh` - Linux/Mac 测试脚本
- `API_服务部署说明.md` - 本说明文件

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic
```

### 2. 启动服务

```bash
# 启动服务（包含环境检查）
python fastapi_service.py
```

### 3. 访问服务

- **服务地址**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## API 接口说明

### 基础接口

#### 1. 根路径
- **URL**: `GET /`
- **说明**: 获取服务基本信息

**curl 请求示例**:
```bash
curl -X GET "http://localhost:8000/"
```

**响应示例**:
```json
{
  "message": "LLaMA-Factory API 服务",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### 2. 健康检查
- **URL**: `GET /health`
- **说明**: 检查服务状态和模型加载情况

**curl 请求示例**:
```bash
curl -X GET "http://localhost:8000/health"
```

**响应示例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Llama-3.2-1B-Instruct"
}
```

#### 3. 模型信息
- **URL**: `GET /model/info`
- **说明**: 获取模型详细信息

**curl 请求示例**:
```bash
curl -X GET "http://localhost:8000/model/info"
```

**响应示例**:
```json
{
  "model_name": "Llama-3.2-1B-Instruct",
  "model_path": "D:/python_projects/LLaMA-Factory/local_run/models/Llama-3.2-1B-Instruct",
  "adapter_path": "D:/python_projects/LLaMA-Factory/local_run/llama3.2-1b-lora-sft",
  "template": "llama3",
  "status": "loaded"
}
```

### 聊天接口

#### 1. 聊天对话
- **URL**: `POST /chat`
- **说明**: 支持多轮对话的聊天接口，支持系统提示词
- **注意**: 系统提示词通过 `messages` 中的 `system` 角色传递，会被自动分离并传递给模型的 `system` 参数

**curl 请求示例**:
```bash
# 简单聊天
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9,
    "stream": false
  }'

# 带系统提示词的聊天
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "你是一个专业的编程助手，擅长Python和机器学习。请用简洁明了的方式回答问题。"
      },
      {
        "role": "user",
        "content": "如何实现一个简单的神经网络？"
      }
    ],
    "temperature": 0.3,
    "max_tokens": 300,
    "top_p": 0.9,
    "stream": false
  }'
```

**多轮对话 curl 示例**:
```bash
# 带系统提示词的多轮对话
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "你是一个友好的AI助手，请保持对话的连贯性，并记住之前的对话内容。"
      },
      {
        "role": "user",
        "content": "我想学习Python编程"
      },
      {
        "role": "assistant",
        "content": "很好！Python是一门非常适合初学者的编程语言。你想从哪个方面开始学习呢？比如基础语法、数据结构、还是某个特定的应用领域？"
      },
      {
        "role": "user",
        "content": "我想先了解基本语法"
      }
    ],
    "temperature": 0.5,
    "max_tokens": 300,
    "top_p": 0.9,
    "stream": false
  }'

# 编程相关的多轮对话
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "你是一个专业的编程助手，擅长Python和机器学习。"
      },
      {
        "role": "user",
        "content": "请帮我写一个Python函数来计算斐波那契数列"
      },
      {
        "role": "assistant",
        "content": "好的，我来为您写一个计算斐波那契数列的Python函数：\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\n这个函数使用递归的方式计算斐波那契数列。"
      },
      {
        "role": "user",
        "content": "能优化一下这个函数吗？递归版本效率比较低"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9,
    "stream": false
  }'
```

**响应示例**:
```json
{
  "response": "当然可以！递归版本确实效率较低，特别是对于较大的数字。我来为您提供一个更高效的迭代版本：\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n\n这个版本的时间复杂度是 O(n)，空间复杂度是 O(1)，比递归版本效率高很多。",
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 120,
    "total_tokens": 205
  },
  "model": "Llama-3.2-1B-Instruct"
}
```

### 文本生成接口

#### 1. 文本生成
- **URL**: `POST /generate`
- **说明**: 基于提示词生成文本

**代码生成 curl 示例**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请写一个Python函数来计算斐波那契数列",
    "temperature": 0.3,
    "max_tokens": 600,
    "top_p": 0.8,
    "stream": false
  }'
```

**文本摘要 curl 示例**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请为以下文本写一个摘要：\n\n人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。",
    "temperature": 0.5,
    "max_tokens": 200,
    "top_p": 0.9,
    "stream": false
  }'
```

**翻译任务 curl 示例**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请将以下英文翻译成中文：\n\nArtificial intelligence is transforming the way we work and live. From smart assistants to autonomous vehicles, AI technologies are becoming increasingly integrated into our daily lives.",
    "temperature": 0.3,
    "max_tokens": 300,
    "top_p": 0.8,
    "stream": false
  }'
```

**响应示例**:
```json
{
  "generated_text": "人工智能正在改变我们的工作和生活方式。从智能助手到自动驾驶汽车，AI技术正日益融入我们的日常生活。",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 25,
    "total_tokens": 70
  },
  "model": "Llama-3.2-1B-Instruct"
}
```

### 批量处理接口

#### 1. 批量生成
- **URL**: `POST /batch`
- **说明**: 批量处理多个提示词

**批量问答 curl 示例**:
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "什么是机器学习？",
      "Python有哪些优势？",
      "如何学习编程？",
      "人工智能的发展趋势是什么？"
    ],
    "temperature": 0.7,
    "max_tokens": 300,
    "top_p": 0.9
  }'
```

**批量代码生成 curl 示例**:
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "写一个Python函数计算两个数的最大公约数",
      "写一个Python函数判断一个数是否为质数",
      "写一个Python函数实现冒泡排序"
    ],
    "temperature": 0.3,
    "max_tokens": 400,
    "top_p": 0.8
  }'
```

**响应示例**:
```json
{
  "results": [
    "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。",
    "Python的优势包括：语法简洁易读、丰富的库生态系统、跨平台兼容、强大的社区支持、适合数据科学和AI开发。",
    "学习编程可以从基础语法开始，多练习写代码，阅读优秀代码，参与开源项目，持续学习和实践。",
    "人工智能的发展趋势包括：更强大的大语言模型、多模态AI、边缘计算、AI伦理和可解释性、自动化程度提高。"
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 180,
    "total_tokens": 225
  },
  "model": "Llama-3.2-1B-Instruct"
}
```

## Postman 测试

### 导入集合

1. 打开 Postman
2. 点击 "Import" 按钮
3. 选择 `LLaMA_Factory_API.postman_collection.json` 文件
4. 导入成功后会看到 "LLaMA Factory API 服务" 集合

### 设置环境变量

1. 在 Postman 中创建新环境
2. 添加变量 `base_url`，值为 `http://localhost:8000`

### 测试接口

集合包含以下测试用例：

#### 基础接口
- 根路径
- 健康检查
- 模型信息

#### 聊天接口
- 简单对话
- 多轮对话
- 创意写作

#### 文本生成接口
- 代码生成
- 文本摘要
- 翻译任务

#### 批量处理接口
- 批量问答
- 批量代码生成

## 参数说明

### 生成参数

- **temperature** (0.0-2.0): 控制生成的随机性，值越高越随机
- **max_tokens** (1-4096): 最大生成 token 数
- **top_p** (0.0-1.0): 核采样参数，控制词汇选择的多样性
- **stream** (bool): 是否使用流式输出

### 推荐参数设置

#### 代码生成
```json
{
  "temperature": 0.3,
  "max_tokens": 600,
  "top_p": 0.8
}
```

#### 创意写作
```json
{
  "temperature": 0.9,
  "max_tokens": 800,
  "top_p": 0.95
}
```

#### 问答任务
```json
{
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9
}
```

## 错误处理

### 常见错误码

- **503**: 模型未加载
- **500**: 生成失败
- **422**: 请求参数错误

### 错误响应示例

```json
{
  "detail": "模型未加载"
}
```

## 性能优化

### 1. 模型加载优化
- 确保模型文件在 SSD 上
- 使用足够的内存（建议 8GB+）

### 2. 并发处理
- 当前版本为单线程处理
- 如需高并发，建议使用多实例部署

### 3. 缓存策略
- 可以添加 Redis 缓存常用回复
- 实现请求去重机制

## 部署建议

### 开发环境
```bash
python fastapi_service.py
```

### 生产环境
```bash
# 使用 gunicorn 部署
pip install gunicorn
gunicorn fastapi_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker 部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "fastapi_service.py"]
```

## 监控和日志

### 日志文件
- 服务日志: `fastapi_service.log`
- 训练日志: `llamafactory_runner.log`

### 监控指标
- 请求响应时间
- 内存使用情况
- GPU 使用情况（如果使用）

## 故障排除

### 1. 模型加载失败
- 检查模型文件路径是否正确
- 确保有足够的内存
- 检查模型文件是否完整

### 2. 服务启动失败
- 检查端口是否被占用
- 检查依赖是否安装完整
- 查看错误日志

### 3. 生成质量不佳
- 调整 temperature 参数
- 增加 max_tokens
- 检查输入提示词质量

## curl 测试示例

### 快速测试脚本

创建一个测试脚本 `test_api.sh`：

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "=== 测试 LLaMA-Factory API 服务 ==="
echo

# 1. 健康检查
echo "1. 健康检查..."
curl -s -X GET "$BASE_URL/health" | jq .
echo

# 2. 模型信息
echo "2. 模型信息..."
curl -s -X GET "$BASE_URL/model/info" | jq .
echo

# 3. 简单聊天
echo "3. 简单聊天..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
    "temperature": 0.7,
    "max_tokens": 200
  }' | jq .
echo

# 4. 代码生成
echo "4. 代码生成..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一个Python函数计算阶乘",
    "temperature": 0.3,
    "max_tokens": 300
  }' | jq .
echo

# 5. 批量处理
echo "5. 批量处理..."
curl -s -X POST "$BASE_URL/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["什么是AI？", "Python有什么特点？"],
    "temperature": 0.7,
    "max_tokens": 150
  }' | jq .
echo

echo "=== 测试完成 ==="
```

### 常用 curl 命令

#### 检查服务状态
```bash
# 检查服务是否运行
curl -s http://localhost:8000/health

# 获取服务信息
curl -s http://localhost:8000/

# 获取模型信息
curl -s http://localhost:8000/model/info
```

#### 聊天测试
```bash
# 简单问候
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"temperature":0.7,"max_tokens":100}'

# 带系统提示词的技术问答
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"你是一个专业的AI助手"},{"role":"user","content":"解释一下什么是机器学习"}],"temperature":0.5,"max_tokens":300}'

# 多轮对话
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"你是一个友好的助手"},{"role":"user","content":"我想学习编程"},{"role":"assistant","content":"很好！你想学什么语言？"},{"role":"user","content":"Python"}],"temperature":0.7,"max_tokens":200}'

# 创意写作
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"写一个关于未来的短故事"}],"temperature":0.9,"max_tokens":500}'
```

#### 代码生成测试
```bash
# 算法实现
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"写一个Python函数实现二分查找","temperature":0.3,"max_tokens":400}'

# 数据处理
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"写一个Python函数读取CSV文件并计算平均值","temperature":0.3,"max_tokens":500}'
```

#### 批量处理测试
```bash
# 批量问答
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"prompts":["什么是深度学习？","什么是神经网络？","什么是强化学习？"],"temperature":0.7,"max_tokens":200}'

# 批量代码生成
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"prompts":["写一个排序函数","写一个搜索函数","写一个过滤函数"],"temperature":0.3,"max_tokens":300}'
```

### 参数调优测试

#### 不同温度参数对比
```bash
# 低温度（更确定）
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"解释什么是Python","temperature":0.1,"max_tokens":200}'

# 中等温度（平衡）
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"解释什么是Python","temperature":0.7,"max_tokens":200}'

# 高温度（更随机）
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"解释什么是Python","temperature":1.2,"max_tokens":200}'
```

#### 不同 token 长度测试
```bash
# 短回答
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"什么是AI？","temperature":0.7,"max_tokens":50}'

# 中等回答
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"什么是AI？","temperature":0.7,"max_tokens":200}'

# 长回答
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"什么是AI？","temperature":0.7,"max_tokens":500}'
```

## 联系支持

如有问题，请查看：
1. 服务日志文件
2. LLaMA-Factory 官方文档
3. 项目 GitHub 仓库
