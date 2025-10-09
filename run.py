#!/usr/bin/env python3
"""
AI Fusion 主入口文件
启动AI Fusion交互式聊天系统
"""

import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入主模块
from ai_fusion.core.main import main

if __name__ == "__main__":
    main()
