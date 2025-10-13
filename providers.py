"""
所有模型提供商实现
支持 OpenAI、Anthropic、Alibaba 等多个服务商
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
from enum import Enum
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


# ============================================
# 基础类型定义
# ============================================

class ModelCapability(str, Enum):
    """模型能力枚举"""
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVITY = "creativity"
    MATH = "math"
    MULTILINGUAL = "multilingual"
    LONG_CONTEXT = "long_context"
    FAST_RESPONSE = "fast_response"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"


class PerformanceLevel(str, Enum):
    """性能级别"""
    OUTSTANDING = "outstanding"
    EXCELLENT = "excellent"
    GOOD = "good"
    MEDIUM = "medium"
    BASIC = "basic"


@dataclass
class ModelInfo:
    """模型信息"""
    model_id: str
    display_name: str
    provider: str
    capabilities: List[ModelCapability] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    suitable_tasks: List[str] = field(default_factory=list)
    performance_profile: Dict[str, PerformanceLevel] = field(default_factory=dict)
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    cost_tier: str = "medium"
    speed_tier: str = "medium"
    is_available: bool = True
    version: Optional[str] = None
    special_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.performance_profile:
            self.performance_profile = {
                "reasoning": PerformanceLevel.MEDIUM,
                "creativity": PerformanceLevel.MEDIUM,
                "coding": PerformanceLevel.MEDIUM,
                "factual": PerformanceLevel.MEDIUM,
            }


# ============================================
# Provider 抽象基类
# ============================================

class BaseProvider(ABC):
    """基础模型提供商抽象类"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
        self._models_cache: Optional[List[ModelInfo]] = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商名称"""
        pass

    @abstractmethod
    async def discover_models(self) -> List[ModelInfo]:
        """发现并返回该提供商支持的所有模型"""
        pass

    @abstractmethod
    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """调用模型生成回答"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查提供商是否可用"""
        pass

    async def get_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """获取模型列表（带缓存）"""
        if self._models_cache is None or force_refresh:
            self._models_cache = await self.discover_models()
        return self._models_cache

    def validate_config(self) -> Dict[str, Any]:
        """验证配置"""
        return {
            "valid": self.is_available(),
            "provider": self.provider_name,
            "has_api_key": bool(self.api_key),
            "base_url": self.base_url
        }


# ============================================
# Universal Provider (OpenAI 兼容)
# ============================================

class UniversalProvider(BaseProvider):
    """
    通用 OpenAI 兼容 API 提供商
    支持任何实现了 OpenAI Chat Completions API 的服务
    """

    # 默认模型知识库
    DEFAULT_MODELS_KNOWLEDGE = {
        "qwen-max": {
            "display_name": "通义千问 Max",
            "capabilities": [ModelCapability.REASONING, ModelCapability.MULTILINGUAL, ModelCapability.LONG_CONTEXT],
            "strengths": ["中文理解顶尖", "知识覆盖广", "多语言支持强"],
            "suitable_tasks": ["中文内容处理", "多语言翻译", "知识问答"],
            "performance_profile": {
                "reasoning": PerformanceLevel.EXCELLENT,
                "creativity": PerformanceLevel.GOOD,
                "coding": PerformanceLevel.GOOD,
                "factual": PerformanceLevel.EXCELLENT,
            },
            "context_window": 32000,
            "cost_tier": "medium",
            "speed_tier": "medium"
        },
        "qwen-plus": {
            "display_name": "通义千问 Plus",
            "capabilities": [ModelCapability.MULTILINGUAL, ModelCapability.FAST_RESPONSE],
            "strengths": ["中文处理优秀", "平衡性能好", "性价比高"],
            "suitable_tasks": ["中文问答", "通用任务", "信息处理"],
            "performance_profile": {
                "reasoning": PerformanceLevel.GOOD,
                "creativity": PerformanceLevel.GOOD,
                "coding": PerformanceLevel.GOOD,
                "factual": PerformanceLevel.GOOD,
            },
            "context_window": 32000,
            "cost_tier": "low",
            "speed_tier": "fast"
        },
        "claude_sonnet4": {
            "display_name": "Claude Sonnet 4",
            "capabilities": [ModelCapability.REASONING, ModelCapability.CODING, ModelCapability.CREATIVITY],
            "strengths": ["逻辑推理卓越", "代码能力顶尖", "创意写作优秀"],
            "suitable_tasks": ["复杂编程", "技术写作", "深度分析"],
            "performance_profile": {
                "reasoning": PerformanceLevel.EXCELLENT,
                "creativity": PerformanceLevel.EXCELLENT,
                "coding": PerformanceLevel.EXCELLENT,
                "factual": PerformanceLevel.GOOD,
            },
            "context_window": 200000,
            "cost_tier": "high",
            "speed_tier": "medium"
        },
        "gpt-41-0414-global": {
            "display_name": "GPT-4.1",
            "capabilities": [ModelCapability.REASONING, ModelCapability.CODING, ModelCapability.MATH],
            "strengths": ["数学和科学计算强", "逻辑推理严谨", "代码优化能力"],
            "suitable_tasks": ["数学问题", "科学计算", "算法设计"],
            "performance_profile": {
                "reasoning": PerformanceLevel.EXCELLENT,
                "creativity": PerformanceLevel.GOOD,
                "coding": PerformanceLevel.EXCELLENT,
                "factual": PerformanceLevel.EXCELLENT,
            },
            "context_window": 128000,
            "cost_tier": "high",
            "speed_tier": "medium"
        },
        "claude37_sonnet_new": {
            "display_name": "Claude 3.7 Sonnet",
            "capabilities": [ModelCapability.REASONING, ModelCapability.FAST_RESPONSE],
            "strengths": ["平衡的综合能力", "快速响应", "稳定性好"],
            "suitable_tasks": ["日常问答", "信息总结", "通用任务"],
            "performance_profile": {
                "reasoning": PerformanceLevel.GOOD,
                "creativity": PerformanceLevel.GOOD,
                "coding": PerformanceLevel.GOOD,
                "factual": PerformanceLevel.GOOD,
            },
            "context_window": 200000,
            "cost_tier": "medium",
            "speed_tier": "fast"
        },
        "gpt-41-mini-0414-global": {
            "display_name": "GPT-4.1 Mini",
            "capabilities": [ModelCapability.FAST_RESPONSE],
            "strengths": ["极速响应", "轻量级任务", "性价比高"],
            "suitable_tasks": ["快速问答", "简单编程", "批量处理"],
            "performance_profile": {
                "reasoning": PerformanceLevel.MEDIUM,
                "creativity": PerformanceLevel.MEDIUM,
                "coding": PerformanceLevel.GOOD,
                "factual": PerformanceLevel.GOOD,
            },
            "context_window": 128000,
            "cost_tier": "low",
            "speed_tier": "very_fast"
        },
        "glm-4.5": {
            "display_name": "智谱 GLM-4.5",
            "capabilities": [ModelCapability.REASONING, ModelCapability.MULTILINGUAL],
            "strengths": ["多模态能力", "中文优化", "综合分析能力"],
            "suitable_tasks": ["多模态任务", "创新设计", "综合分析"],
            "performance_profile": {
                "reasoning": PerformanceLevel.GOOD,
                "creativity": PerformanceLevel.EXCELLENT,
                "coding": PerformanceLevel.GOOD,
                "factual": PerformanceLevel.GOOD,
            },
            "context_window": 128000,
            "cost_tier": "medium",
            "speed_tier": "medium"
        },
        "qwen3-coder-480b-a35b-instruct": {
            "display_name": "Qwen3 Coder 480B",
            "capabilities": [ModelCapability.CODING],
            "strengths": ["代码生成顶尖", "算法理解深", "多语言编程"],
            "suitable_tasks": ["代码生成", "算法实现", "代码审查"],
            "performance_profile": {
                "reasoning": PerformanceLevel.EXCELLENT,
                "creativity": PerformanceLevel.GOOD,
                "coding": PerformanceLevel.OUTSTANDING,
                "factual": PerformanceLevel.EXCELLENT,
            },
            "context_window": 32000,
            "cost_tier": "high",
            "speed_tier": "medium",
            "special_features": ["480B超大规模", "代码专精"]
        },
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url or "https://api.openai.com/v1", **kwargs)
        self.client = None
        if self.is_available():
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.custom_models_list = kwargs.get("models_list", None)

    @property
    def provider_name(self) -> str:
        return "universal"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        if not self.is_available():
            return []

        models = []

        # 获取模型列表
        if self.custom_models_list:
            model_ids = self.custom_models_list
        elif os.getenv("AVAILABLE_MODELS"):
            model_ids = [m.strip() for m in os.getenv("AVAILABLE_MODELS").split(",")]
        else:
            model_ids = list(self.DEFAULT_MODELS_KNOWLEDGE.keys())

        # 创建 ModelInfo
        for model_id in model_ids:
            if model_id in self.DEFAULT_MODELS_KNOWLEDGE:
                info = self.DEFAULT_MODELS_KNOWLEDGE[model_id]
                models.append(ModelInfo(model_id=model_id, provider=self.provider_name, **info))
            else:
                # 未知模型使用默认配置
                models.append(ModelInfo(
                    model_id=model_id,
                    display_name=model_id,
                    provider=self.provider_name,
                    capabilities=[ModelCapability.REASONING],
                    strengths=["通用AI能力"],
                    suitable_tasks=["通用任务"]
                ))

        return models

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        if not self.client:
            raise ValueError(f"{self.provider_name} provider is not available")

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 2000,
            **kwargs
        )
        return response.choices[0].message.content


# ============================================
# Anthropic Provider
# ============================================

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API 提供商"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.client = None
        if self.is_available():
            self.client = AsyncAnthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        if not self.is_available():
            return []

        return [
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                provider=self.provider_name,
                capabilities=[ModelCapability.REASONING, ModelCapability.CODING, ModelCapability.CREATIVITY, ModelCapability.LONG_CONTEXT],
                strengths=["卓越的逻辑推理", "顶级代码能力", "创意写作优秀", "长文本理解"],
                suitable_tasks=["复杂编程", "技术写作", "创意内容", "深度分析"],
                performance_profile={
                    "reasoning": PerformanceLevel.EXCELLENT,
                    "creativity": PerformanceLevel.EXCELLENT,
                    "coding": PerformanceLevel.EXCELLENT,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=200000,
                max_output_tokens=8192,
                cost_tier="high",
                speed_tier="medium",
                special_features=["200K上下文", "强逻辑推理"]
            ),
            ModelInfo(
                model_id="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                provider=self.provider_name,
                capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.CODING],
                strengths=["极速响应", "高效处理", "性价比优秀"],
                suitable_tasks=["快速问答", "简单编程", "文本处理"],
                performance_profile={
                    "reasoning": PerformanceLevel.GOOD,
                    "creativity": PerformanceLevel.GOOD,
                    "coding": PerformanceLevel.GOOD,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=200000,
                max_output_tokens=4096,
                cost_tier="low",
                speed_tier="very_fast"
            ),
        ]

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        if not self.client:
            raise ValueError(f"{self.provider_name} provider is not available")

        # Anthropic 需要将系统消息单独处理
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            else:
                chat_messages.append(msg)

        response = await self.client.messages.create(
            model=model_id,
            messages=chat_messages if chat_messages else messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            **kwargs
        )
        return response.content[0].text


# ============================================
# Model Registry
# ============================================

@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_class: Type[BaseProvider]
    api_key_env: str
    base_url_env: Optional[str] = None
    enabled: bool = True
    custom_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_config is None:
            self.custom_config = {}


class ModelRegistry:
    """
    模型注册中心
    负责管理所有提供商和模型
    """

    # 默认提供商配置
    DEFAULT_PROVIDERS = {
        "universal": ProviderConfig(
            provider_class=UniversalProvider,
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL"
        ),
        "anthropic": ProviderConfig(
            provider_class=AnthropicProvider,
            api_key_env="ANTHROPIC_API_KEY"
        ),
    }

    def __init__(self, custom_providers: Optional[Dict[str, ProviderConfig]] = None):
        self.providers: Dict[str, BaseProvider] = {}
        self._all_models: List[ModelInfo] = []
        self._models_by_provider: Dict[str, List[ModelInfo]] = {}

        # 合并配置
        provider_configs = {**self.DEFAULT_PROVIDERS}
        if custom_providers:
            provider_configs.update(custom_providers)

        # 初始化提供商
        for name, config in provider_configs.items():
            if not config.enabled:
                continue

            api_key = os.getenv(config.api_key_env)
            base_url = os.getenv(config.base_url_env) if config.base_url_env else None

            provider = config.provider_class(api_key=api_key, base_url=base_url, **config.custom_config)

            if provider.is_available():
                self.providers[name] = provider
                print(f"✅ 已加载提供商: {provider.provider_name}")
            else:
                print(f"⚠️ 提供商 {name} 不可用（未配置 API Key）")

    async def discover_all_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """发现所有可用模型"""
        if self._all_models and not force_refresh:
            return self._all_models

        self._all_models = []
        self._models_by_provider = {}

        for provider_name, provider in self.providers.items():
            try:
                models = await provider.get_models(force_refresh=force_refresh)
                self._all_models.extend(models)
                self._models_by_provider[provider_name] = models
                print(f"📦 {provider_name}: 发现 {len(models)} 个模型")
            except Exception as e:
                print(f"❌ 发现 {provider_name} 模型失败: {e}")

        return self._all_models

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """根据模型ID获取模型信息"""
        for model in self._all_models:
            if model.model_id == model_id:
                return model
        return None

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """调用指定模型"""
        model_info = self.get_model(model_id)
        if not model_info:
            raise ValueError(f"未找到模型: {model_id}")

        provider = self.providers.get(model_info.provider)
        if not provider:
            raise ValueError(f"提供商 {model_info.provider} 不可用")

        return await provider.call_model(model_id, messages, **kwargs)

    def list_available_providers(self) -> List[str]:
        """列出所有可用的提供商"""
        return list(self.providers.keys())
