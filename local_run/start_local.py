#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动本地模式 Gemma 服务 (1B 模型)
"""

import sys
import os
from pathlib import Path

# 指定配置文件路径（本地配置）
os.environ["GEMMA_CONFIG"] = str(Path(__file__).parent / "config.local.yaml")

# 导入并运行服务
sys.path.insert(0, str(Path(__file__).parent))
from local_run.gemma_service import main

if __name__ == "__main__":
    main()
