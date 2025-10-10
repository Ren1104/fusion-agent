"""
AI Fusion - 多模型智能融合系统
Multi-Model Intelligent Fusion System
"""

__version__ = "1.0.0"
__author__ = "AI Fusion Team"

from ai_fusion.core.main import main, process_question, interactive_chat
from ai_fusion.core.flow import create_ai_fusion_flow

__all__ = [
    'main',
    'process_question',
    'interactive_chat',
    'create_ai_fusion_flow'
]
