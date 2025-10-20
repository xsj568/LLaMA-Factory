#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 模型部署服务
=================

基于 LLaMA-Factory 的简洁 Gemma 模型部署服务，支持：
- 从魔搭社区自动下载模型
- 服务器和本地不同配置
- vLLM 推理引擎
- 默认模板配置

使用方法：
python gemma_service.py
"""

import os
import sys
import asyncio
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目路径
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# FastAPI 相关导入
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# LLaMA-Factory 导入
from llamafactory.chat.chat_model import ChatModel
from llamafactory.extras.constants import EngineName

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
chat_model = None
model_loaded = False
config: dict[str, Any] | None = None
current_config_path: str | None = None

def load_config() -> dict[str, Any]:
    """加载配置文件（通过 GEMMA_CONFIG 指定路径，未设置则默认 config.yaml）。"""
    global config, current_config_path
    cfg_env = os.getenv("GEMMA_CONFIG")
    config_path = Path(cfg_env) if cfg_env else (PROJECT_ROOT / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    current_config_path = str(config_path)
    # 期望的结构: 根下含有 environment 和 common
    if "environment" not in config or "common" not in config:
        raise ValueError("配置文件缺少必要的 'environment' 或 'common' 节。")
    logger.info(f"配置文件加载成功: {current_config_path}")
    return config

def get_environment_config() -> Dict[str, Any]:
    """获取当前配置的 environment 段。"""
    if config is None:
        load_config()
    return config["environment"]

# Pydantic 模型定义
class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Dict[str, str]] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成token数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    stream: bool = Field(default=False, description="是否流式输出")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="模型回复")
    usage: Dict[str, int] = Field(..., description="token使用统计")
    model: str = Field(..., description="模型名称")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    deployment_type: str = Field(..., description="部署类型")
    model_info: Dict[str, Any] = Field(..., description="模型信息")

# 创建 FastAPI 应用
app = FastAPI(
    title="Gemma 模型部署服务",
    description="基于 LLaMA-Factory 的 Gemma 模型推理 API 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model_args() -> Dict[str, Any]:
    """获取模型参数配置"""
    env_config = get_environment_config()
    
    # 基础模型参数
    model_args = {
        "model_name_or_path": env_config["model_name"],
        "template": env_config["template"],
    }
    
    # 根据推理引擎设置不同的参数
    if env_config["inference_engine"] == "vllm":
        model_args["infer_backend"] = EngineName.VLLM
        # 添加 vLLM 特定参数
        if "vllm_config" in env_config:
            model_args.update(env_config["vllm_config"])
        logger.info(f"使用 vLLM 推理引擎")
    else:
        model_args["infer_backend"] = EngineName.HF
        # 添加 HuggingFace 特定参数
        if "hf_config" in env_config:
            model_args.update(env_config["hf_config"])
        logger.info(f"使用 HuggingFace 推理引擎")
    
    # 添加通用模型参数
    model_args.update(config["common"]["model_args"])
    
    logger.info(f"模型参数配置完成: {env_config['model_name']}")
    return model_args

def load_model() -> bool:
    """加载模型"""
    global chat_model, model_loaded
    
    try:
        logger.info("开始加载模型...")
        
        # 获取模型参数
        model_args = get_model_args()
        model_name = model_args["model_name_or_path"]
        
        logger.info(f"模型名称: {model_name}")
        logger.info(f"推理引擎: {model_args['infer_backend']}")
        logger.info(f"模板: {model_args['template']}")
        
        # 创建聊天模型
        chat_model = ChatModel(args=model_args)
        model_loaded = True
        
        logger.info("模型加载成功！")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        model_loaded = False
        return False

def get_model_info() -> Dict[str, Any]:
    """获取模型信息"""
    env_config = get_environment_config()
    
    return {
        "model_name": env_config["model_name"],
        "description": env_config["description"],
        "template": env_config["template"],
        "inference_engine": env_config["inference_engine"],
        "config_path": current_config_path or "",
        "status": "loaded" if model_loaded else "not_loaded"
    }

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动 Gemma 模型部署服务...")
    
    try:
        # 加载模型
        success = load_model()
        
        if success:
            logger.info("服务启动成功！")
        else:
            logger.error("模型加载失败，服务启动失败")
            raise RuntimeError("模型加载失败")
            
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "Gemma 模型部署服务",
        "version": "1.0.0",
        "config_path": current_config_path or "",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    model_info = get_model_info()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        deployment_type="config",
        model_info=model_info
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
            model=get_model_info()["model_name"]
        )
        
    except Exception as e:
        logger.error(f"聊天生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    return get_model_info()

def show_startup_info():
    """显示启动信息"""
    env_config = get_environment_config()
    service_config = env_config["service_config"]
    
    logger.info("=" * 60)
    logger.info("Gemma 模型部署服务")
    logger.info("=" * 60)
    logger.info("服务信息:")
    logger.info(f"   服务地址: http://{service_config['host']}:{service_config['port']}")
    logger.info(f"   API 文档: http://{service_config['host']}:{service_config['port']}/docs")
    logger.info(f"   健康检查: http://{service_config['host']}:{service_config['port']}/health")
    logger.info(f"   模型信息: http://{service_config['host']}:{service_config['port']}/model/info")
    logger.info("")
    logger.info("可用接口:")
    logger.info("   POST /chat      - 聊天对话")
    logger.info("")
    logger.info("测试命令:")
    logger.info(f"   健康检查: curl http://{service_config['host']}:{service_config['port']}/health")
    logger.info(f"   简单聊天: curl -X POST http://{service_config['host']}:{service_config['port']}/chat \\")
    logger.info("     -H 'Content-Type: application/json' \\")
    logger.info("     -d '{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'")
    logger.info("")
    logger.info("按 Ctrl+C 停止服务")
    logger.info("=" * 60)

def main():
    """主函数"""
    logger.info("正在检查环境...")
    
    # 加载配置
    try:
        load_config()
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        sys.exit(1)
    
    # 检查依赖
    required_packages = config["common"]["required_packages"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少以下依赖包: {', '.join(missing_packages)}")
        logger.error("请运行以下命令安装:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    logger.info("环境检查通过")
    logger.info("")
    
    # 显示启动信息
    show_startup_info()
    
    # 显示部署配置
    env_config = get_environment_config()
    logger.info(f"部署配置: {env_config['description']}")
    logger.info(f"推理引擎: {env_config['inference_engine']}")
    logger.info(f"模板: {env_config['template']}")
    logger.info("")
    
    try:
        # 启动服务
        logger.info("正在启动服务...")
        service_config = env_config["service_config"]
        uvicorn.run(
            "gemma_service:app",
            host=service_config["host"],
            port=service_config["port"],
            reload=False,
            log_level=service_config["log_level"]
        )
    except KeyboardInterrupt:
        logger.info("\n\n服务已停止")
    except Exception as e:
        logger.error(f"\n服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
