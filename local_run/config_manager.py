#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理服务配置管理器
=================

统一管理推理服务配置，包括模型、推理引擎、生成参数等
避免配置分散在多个文件中
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

class InferenceConfigManager:
    """推理服务配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器"""
        if config_path is None:
            # 默认配置文件路径
            self.config_path = Path(__file__).parent / "configs" / "inference_config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get_current_model(self) -> str:
        """获取当前使用的模型名称"""
        return self._config.get("current_model", "qwen3_0_6b")
    
    def get_model_info(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """获取模型信息"""
        if model_key is None:
            # 使用当前模型
            model_key = self.get_current_model()
        
        # 获取指定模型信息
        supported_models = self._config.get("supported_models", {})
        if model_key not in supported_models:
            raise ValueError(f"不支持的模型: {model_key}")
        
        model_info = supported_models[model_key]
        base_info = {
            "model_name": model_info.get("name", "Unknown"),
            "template": model_info.get("template", "unknown"),
            "base_path": model_info.get("base_path", ""),
            "merged_path": model_info.get("merged_path", ""),
            "description": model_info.get("description", "")
        }
        
        # 从模型文件动态读取配置
        try:
            model_path = Path(model_info.get("base_path", ""))
            if model_path.exists():
                # 读取模型配置
                config_path = model_path / "config.json"
                tokenizer_config_path = model_path / "tokenizer_config.json"
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        model_config = json.load(f)
                    base_info["context_length"] = model_config.get("max_position_embeddings", 0)
                
                if tokenizer_config_path.exists():
                    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                        tokenizer_config = json.load(f)
                    
                    # 提取特殊token
                    special_tokens = []
                    if "additional_special_tokens" in tokenizer_config:
                        special_tokens.extend(tokenizer_config["additional_special_tokens"])
                    if "bos_token" in tokenizer_config and tokenizer_config["bos_token"]:
                        special_tokens.append(tokenizer_config["bos_token"])
                    if "eos_token" in tokenizer_config and tokenizer_config["eos_token"]:
                        special_tokens.append(tokenizer_config["eos_token"])
                    
                    base_info["special_tokens"] = special_tokens
                    base_info["model_max_length"] = tokenizer_config.get("model_max_length", 0)
        except Exception as e:
            # 如果读取失败，使用默认值
            base_info["context_length"] = 0
            base_info["special_tokens"] = []
            base_info["model_max_length"] = 0
        
        return base_info
    
    def get_inference_engine_config(self) -> Dict[str, Any]:
        """获取推理引擎配置"""
        return self._config.get("inference_engines", {})
    
    def get_generation_defaults(self) -> Dict[str, Any]:
        """获取生成参数默认值"""
        return self._config.get("generation_defaults", {})
    
    def get_service_config(self) -> Dict[str, Any]:
        """获取服务配置"""
        return self._config.get("service_config", {})
    
    def get_model_args(self, use_merged: bool = True, use_vllm: bool = True) -> Dict[str, Any]:
        """获取模型参数配置"""
        model_info = self.get_model_info()
        engine_config = self.get_inference_engine_config()
        
        # 基础模型参数
        model_args = {
            "template": model_info["template"],
            "trust_remote_code": True,
        }
        
        # 选择模型路径
        if use_merged:
            model_args["model_name_or_path"] = model_info["merged_path"]
        else:
            model_args["model_name_or_path"] = model_info["base_path"]
        
        # 推理引擎配置
        if use_vllm:
            try:
                import vllm
                model_args["infer_backend"] = "vllm"
                
                # 添加vllm特定配置
                vllm_config = engine_config.get("vllm_config", {})
                model_args.update({
                    "vllm_maxlen": vllm_config.get("maxlen", 4096),
                    "vllm_gpu_util": vllm_config.get("gpu_util", 0.7),
                    "vllm_enforce_eager": vllm_config.get("enforce_eager", False),
                    "vllm_max_lora_rank": vllm_config.get("max_lora_rank", 32),
                })
            except ImportError:
                model_args["infer_backend"] = engine_config.get("fallback", "huggingface")
        else:
            model_args["infer_backend"] = engine_config.get("fallback", "huggingface")
        
        return model_args
    
    def get_supported_models(self) -> Dict[str, Any]:
        """获取所有支持的模型列表"""
        return self._config.get("supported_models", {})
    
    def switch_model(self, model_key: str):
        """切换当前使用的模型"""
        supported_models = self.get_supported_models()
        if model_key not in supported_models:
            raise ValueError(f"不支持的模型: {model_key}")
        
        # 更新当前模型配置
        self._config["current_model"] = model_key
        
        # 保存配置
        self._save_config()
    
    def _save_config(self):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

# 全局配置管理器实例
config_manager = InferenceConfigManager()
