"""
AI Fusion 模型提供商模块
支持多种 AI 模型提供商的适配器
"""

from ai_fusion.providers.base import BaseProvider, ModelInfo
from ai_fusion.providers.universal_provider import UniversalProvider
from ai_fusion.providers.openai_provider import OpenAIProvider
from ai_fusion.providers.anthropic_provider import AnthropicProvider
from ai_fusion.providers.alibaba_provider import AlibabaProvider

__all__ = [
    'BaseProvider',
    'ModelInfo',
    'UniversalProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'AlibabaProvider',
]
