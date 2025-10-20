#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动生产环境 Gemma 服务 (27B 模型)
"""

import sys
import os
from pathlib import Path

# 指定配置文件路径（生产配置）
os.environ["GEMMA_CONFIG"] = str(Path(__file__).parent / "config.production.yaml")

# 导入并运行服务
sys.path.insert(0, str(Path(__file__).parent))
from local_run.gemma_service import main

if __name__ == "__main__":
    main()
