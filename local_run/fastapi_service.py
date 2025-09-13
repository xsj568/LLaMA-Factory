#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-Factory FastAPI 服务部署
==============================

提供基于 FastAPI 的模型推理服务，支持：
- 文本生成
- 对话聊天
- 批量处理
- 健康检查
- 启动前环境检查

使用方法：
python fastapi_service.py
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# 添加项目路径
THIS_FILE = Path(__file__).resolve()
LOCAL_RUN_DIR = THIS_FILE.parent  # 当前目录就是 local_run
PROJECT_ROOT = LOCAL_RUN_DIR.parent  # 上级目录是 LLaMA-Factory
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# FastAPI 相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# LLaMA-Factory 导入
from llamafactory.chat.chat_model import ChatModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOCAL_RUN_DIR / 'fastapi_service.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
chat_model = None
model_loaded = False

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model_files():
    """检查模型文件是否存在"""
    model_path = LOCAL_RUN_DIR / "models" / "Llama-3.2-1B-Instruct"
    adapter_path = LOCAL_RUN_DIR / "saves" / "llama3.2-1b-lora-sft"
    
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("请确保模型文件已正确放置在 models/ 目录下")
        return False
    
    if not adapter_path.exists():
        print(f"⚠️  警告: LoRA 适配器不存在: {adapter_path}")
        print("将使用基础模型进行推理")
    else:
        print(f"✅ LoRA 适配器存在: {adapter_path}")
    
    return True

def show_startup_info():
    """显示启动信息"""
    print("=" * 60)
    print("🚀 LLaMA-Factory FastAPI 服务")
    print("=" * 60)
    print("📋 服务信息:")
    print(f"   • 服务地址: http://localhost:8000")
    print(f"   • API 文档: http://localhost:8000/docs")
    print(f"   • 健康检查: http://localhost:8000/health")
    print(f"   • 模型信息: http://localhost:8000/model/info")
    print()
    print("🔧 可用接口:")
    print("   • POST /chat      - 聊天对话")
    print("   • POST /generate  - 文本生成")
    print("   • POST /batch     - 批量处理")
    print()
    print("📝 测试命令:")
    print("   • 健康检查: curl http://localhost:8000/health")
    print("   • 简单聊天: curl -X POST http://localhost:8000/chat \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'")
    print()
    print("⏹️  按 Ctrl+C 停止服务")
    print("=" * 60)

# Pydantic 模型定义
class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Dict[str, str]] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成token数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    stream: bool = Field(default=False, description="是否流式输出")

class TextGenerationRequest(BaseModel):
    """文本生成请求模型"""
    prompt: str = Field(..., description="输入提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成token数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    stream: bool = Field(default=False, description="是否流式输出")

class BatchRequest(BaseModel):
    """批量处理请求模型"""
    prompts: List[str] = Field(..., description="提示词列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成token数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="模型回复")
    usage: Dict[str, int] = Field(..., description="token使用统计")
    model: str = Field(..., description="模型名称")

class TextGenerationResponse(BaseModel):
    """文本生成响应模型"""
    generated_text: str = Field(..., description="生成的文本")
    usage: Dict[str, int] = Field(..., description="token使用统计")
    model: str = Field(..., description="模型名称")

class BatchResponse(BaseModel):
    """批量处理响应模型"""
    results: List[str] = Field(..., description="生成结果列表")
    usage: Dict[str, int] = Field(..., description="token使用统计")
    model: str = Field(..., description="模型名称")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    model_name: Optional[str] = Field(None, description="模型名称")

# 创建 FastAPI 应用
app = FastAPI(
    title="LLaMA-Factory API 服务",
    description="基于 LLaMA-Factory 的模型推理 API 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """加载模型"""
    global chat_model, model_loaded
    
    try:
        # 检查模型路径
        model_path = LOCAL_RUN_DIR / "models" / "Llama-3.2-1B-Instruct"
        adapter_path = LOCAL_RUN_DIR / "saves" / "llama3.2-1b-lora-sft"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 构建模型参数
        model_args = {
            "model_name_or_path": str(model_path),
            "template": "llama3",
            "trust_remote_code": True,
        }
        
        # 如果存在适配器，则加载
        if adapter_path.exists():
            model_args["adapter_name_or_path"] = str(adapter_path)
            logger.info(f"加载 LoRA 适配器: {adapter_path}")
        
        # 创建聊天模型
        chat_model = ChatModel(args=model_args)
        model_loaded = True
        
        logger.info(f"模型加载成功: {model_path}")
        if adapter_path.exists():
            logger.info(f"LoRA 适配器加载成功: {adapter_path}")
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        model_loaded = False
        raise e

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动 FastAPI 服务...")
    try:
        load_model()
        logger.info("FastAPI 服务启动成功！")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "LLaMA-Factory API 服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name="Llama-3.2-1B-Instruct" if model_loaded else None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 设置生成参数
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        # 处理消息，分离系统提示词和对话消息
        system_message = None
        chat_messages = []
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)
        
        # 生成回复
        if request.stream:
            # 流式生成
            response_text = ""
            for new_text in chat_model.stream_chat(chat_messages, system=system_message, **generation_kwargs):
                response_text += new_text
            # 流式生成时使用简化计算
            input_tokens = sum(len(str(msg["content"]).split()) for msg in chat_messages)
            if system_message:
                input_tokens += len(str(system_message).split())
            output_tokens = len(response_text.split())
        else:
            # 非流式生成
            response_list = chat_model.chat(chat_messages, system=system_message, **generation_kwargs)
            response_obj = response_list[0]
            response_text = response_obj.response_text
            # 使用 Response 对象中的准确 token 计数
            input_tokens = response_obj.prompt_length
            output_tokens = response_obj.response_length
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            model="Llama-3.2-1B-Instruct"
        )
        
    except Exception as e:
        logger.error(f"聊天生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """文本生成接口"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 将提示词转换为消息格式
        messages = [{"role": "user", "content": request.prompt}]
        
        # 设置生成参数
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        # 生成文本
        if request.stream:
            # 流式生成
            generated_text = ""
            for new_text in chat_model.stream_chat(messages, **generation_kwargs):
                generated_text += new_text
            # 流式生成时使用简化计算
            input_tokens = len(request.prompt.split())
            output_tokens = len(generated_text.split())
        else:
            # 非流式生成
            response_list = chat_model.chat(messages, **generation_kwargs)
            response_obj = response_list[0]
            generated_text = response_obj.response_text
            # 使用 Response 对象中的准确 token 计数
            input_tokens = response_obj.prompt_length
            output_tokens = response_obj.response_length
        
        return TextGenerationResponse(
            generated_text=generated_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            model="Llama-3.2-1B-Instruct"
        )
        
    except Exception as e:
        logger.error(f"文本生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/batch", response_model=BatchResponse)
async def batch_generate(request: BatchRequest):
    """批量生成接口"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 设置生成参数
        generation_kwargs = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        
        # 批量处理
        for prompt in request.prompts:
            messages = [{"role": "user", "content": prompt}]
            response_list = chat_model.chat(messages, **generation_kwargs)
            response_obj = response_list[0]
            generated_text = response_obj.response_text
            results.append(generated_text)
            
            # 累计 token 使用量（使用 Response 对象中的准确计数）
            total_input_tokens += response_obj.prompt_length
            total_output_tokens += response_obj.response_length
        
        return BatchResponse(
            results=results,
            usage={
                "prompt_tokens": total_input_tokens,
                "completion_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            },
            model="Llama-3.2-1B-Instruct"
        )
        
    except Exception as e:
        logger.error(f"批量生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")

@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_name": "Llama-3.2-1B-Instruct",
        "model_path": str(LOCAL_RUN_DIR / "models" / "Llama-3.2-1B-Instruct"),
        "adapter_path": str(LOCAL_RUN_DIR / "saves" / "llama3.2-1b-lora-sft"),
        "template": "llama3",
        "status": "loaded"
    }

def main():
    """主函数：启动前检查并启动服务"""
    print("🔍 正在检查环境...")
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 环境检查失败，请安装缺失的依赖后重试")
        sys.exit(1)
    
    # 检查模型文件
    if not check_model_files():
        print("\n❌ 模型文件检查失败，请确保模型文件存在")
        sys.exit(1)
    
    print("✅ 环境检查通过")
    print()
    
    # 显示启动信息
    show_startup_info()
    
    try:
        # 启动服务
        print("🚀 正在启动服务...")
        uvicorn.run(
            "fastapi_service:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
