#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动本地模式 Gemma 服务 (1B 模型)
"""

import sys
import os
from pathlib import Path

# 指定配置文件路径（本地配置）
config_path = Path(__file__).parent / "config.local.yaml"
os.environ["GEMMA_CONFIG"] = str(config_path)

# 验证配置文件存在
if not config_path.exists():
    print(f"❌ 配置文件不存在: {config_path}")
    sys.exit(1)

# 导入并运行服务
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gemma_service import main
    print("✅ 成功导入 gemma_service")
    print(f"📁 使用配置文件: {config_path}")
    print("🚀 启动本地环境 Gemma 服务 (1B 模型)...")
    main()
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)
