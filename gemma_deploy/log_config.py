#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志配置模块
================

提供统一的日志配置，支持：
- 基于文件大小的日志轮转
- 统一的日志格式
- 多级别日志记录
- 自动清理旧日志文件

使用方法：
from log_config import setup_logging, get_logger

# 设置日志
setup_logging()

# 获取日志记录器
logger = get_logger(__name__)
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import yaml


class UnifiedLogConfig:
    """统一日志配置类"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化日志配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.log_dir = Path(__file__).parent / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # 默认配置
        self.default_config = {
            'log_level': 'INFO',
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'backup_count': 10,
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'console_output': True,
            'file_output': True,
            'unified_log_file': 'unified.log',  # 统一日志文件
            'cleanup_old_logs': True,
            'max_log_age_days': 30
        }
        
        # 加载配置
        self.config = self._load_config()
        
        # 设置日志
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载日志配置"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    # 合并默认配置
                    merged_config = self.default_config.copy()
                    merged_config.update(config.get('logging', {}))
                    return merged_config
            except Exception as e:
                print(f"警告: 无法加载日志配置文件 {self.config_path}: {e}")
                print("使用默认配置")
        
        return self.default_config.copy()
    
    def _setup_logging(self):
        """设置日志配置"""
        # 清除现有的处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置根日志级别
        root_logger.setLevel(getattr(logging, self.config['log_level'].upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.config['log_format'],
            datefmt=self.config['date_format']
        )
        
        # 控制台处理器
        if self.config['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, self.config['log_level'].upper()))
            root_logger.addHandler(console_handler)
        
        # 统一日志文件处理器（带轮转）
        if self.config['file_output']:
            unified_log_file = self.log_dir / self.config['unified_log_file']
            unified_handler = logging.handlers.RotatingFileHandler(
                unified_log_file,
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
            unified_handler.setFormatter(formatter)
            unified_handler.setLevel(logging.DEBUG)  # 记录所有级别
            root_logger.addHandler(unified_handler)
        
        # 清理旧日志文件
        if self.config['cleanup_old_logs']:
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """清理旧的日志文件"""
        try:
            max_age_days = self.config['max_log_age_days']
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            for log_file in self.log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"已删除旧日志文件: {log_file}")
        except Exception as e:
            print(f"清理旧日志文件时出错: {e}")
    
    def get_request_logger(self) -> logging.Logger:
        """获取请求日志记录器（现在也使用统一日志文件）"""
        request_logger = logging.getLogger('request_logger')
        request_logger.setLevel(logging.INFO)
        # 请求日志现在也会写入统一日志文件，不需要单独的处理器
        return request_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        return logging.getLogger(name)
    
    def log_startup_info(self, service_name: str, config_info: Dict[str, Any]):
        """记录服务启动信息"""
        logger = self.get_logger(service_name)
        logger.info("=" * 60)
        logger.info(f"🚀 启动 {service_name}")
        logger.info(f"📁 日志目录: {self.log_dir}")
        logger.info(f"📄 统一日志文件: {self.config['unified_log_file']}")
        logger.info(f"📏 最大文件大小: {self.config['max_file_size'] / 1024 / 1024:.1f}MB")
        logger.info(f"📚 备份文件数量: {self.config['backup_count']}")
        logger.info(f"🧹 自动清理: {self.config['cleanup_old_logs']}")
        if self.config['cleanup_old_logs']:
            logger.info(f"🗑️ 最大保留天数: {self.config['max_log_age_days']}")
        logger.info("=" * 60)
        
        # 记录配置信息
        for key, value in config_info.items():
            logger.info(f"⚙️ {key}: {value}")
    
    def log_shutdown_info(self, service_name: str):
        """记录服务关闭信息"""
        logger = self.get_logger(service_name)
        logger.info("=" * 60)
        logger.info(f"🛑 关闭 {service_name}")
        logger.info("=" * 60)


# 全局日志配置实例
_log_config: Optional[UnifiedLogConfig] = None


def setup_logging(config_path: Optional[Path] = None) -> UnifiedLogConfig:
    """
    设置统一日志配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        UnifiedLogConfig: 日志配置实例
    """
    global _log_config
    _log_config = UnifiedLogConfig(config_path)
    return _log_config


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    if _log_config is None:
        setup_logging()
    return _log_config.get_logger(name)


def get_request_logger() -> logging.Logger:
    """
    获取请求日志记录器
    
    Returns:
        logging.Logger: 请求日志记录器实例
    """
    if _log_config is None:
        setup_logging()
    return _log_config.get_request_logger()


def log_startup_info(service_name: str, config_info: Dict[str, Any]):
    """
    记录服务启动信息
    
    Args:
        service_name: 服务名称
        config_info: 配置信息字典
    """
    if _log_config is None:
        setup_logging()
    _log_config.log_startup_info(service_name, config_info)


def log_shutdown_info(service_name: str):
    """
    记录服务关闭信息
    
    Args:
        service_name: 服务名称
    """
    if _log_config is None:
        setup_logging()
    _log_config.log_shutdown_info(service_name)


# 便捷函数
def info(message: str, logger_name: str = __name__):
    """记录信息日志"""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = __name__):
    """记录警告日志"""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = __name__):
    """记录错误日志"""
    get_logger(logger_name).error(message)


def debug(message: str, logger_name: str = __name__):
    """记录调试日志"""
    get_logger(logger_name).debug(message)


if __name__ == "__main__":
    # 测试日志配置
    setup_logging()
    
    logger = get_logger("test")
    request_logger = get_request_logger()
    
    logger.info("这是一条测试信息")
    logger.warning("这是一条测试警告")
    logger.error("这是一条测试错误")
    
    request_logger.info("这是一条测试请求日志")
    
    print("日志配置测试完成，请检查 logs 目录下的日志文件")
