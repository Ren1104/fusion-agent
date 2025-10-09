"""
模型注册中心
统一管理所有模型提供商，支持动态发现和调用
"""

import os
from typing import List, Dict, Optional, Any, Type
from dataclasses import dataclass

from ai_fusion.providers.base import BaseProvider, ModelInfo
from ai_fusion.providers.openai_provider import OpenAIProvider
from ai_fusion.providers.anthropic_provider import AnthropicProvider
from ai_fusion.providers.alibaba_provider import AlibabaProvider
from ai_fusion.providers.universal_provider import UniversalProvider


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

    负责:
    1. 自动发现所有配置的提供商
    2. 根据 API Key 动态加载模型
    3. 统一模型调用接口
    4. 模型信息查询和过滤
    """

    # 内置提供商配置
    DEFAULT_PROVIDERS = {
        # 第三方兼容服务（支持多模型）
        "universal": ProviderConfig(
            provider_class=UniversalProvider,
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL"
        ),
        # 官方 OpenAI
        "openai": ProviderConfig(
            provider_class=OpenAIProvider,
            api_key_env="OPENAI_OFFICIAL_API_KEY",
            base_url_env=None
        ),
        # 官方 Anthropic
        "anthropic": ProviderConfig(
            provider_class=AnthropicProvider,
            api_key_env="ANTHROPIC_API_KEY"
        ),
        # 官方阿里云通义千问
        "alibaba": ProviderConfig(
            provider_class=AlibabaProvider,
            api_key_env="QWEN_API_KEY",
            base_url_env="QWEN_BASE_URL"
        ),
    }

    def __init__(self, custom_providers: Optional[Dict[str, ProviderConfig]] = None):
        """
        初始化模型注册中心

        Args:
            custom_providers: 自定义提供商配置（会覆盖默认配置）
        """
        self.providers: Dict[str, BaseProvider] = {}
        self._all_models: List[ModelInfo] = []
        self._models_by_provider: Dict[str, List[ModelInfo]] = {}

        # 合并默认和自定义提供商配置
        provider_configs = {**self.DEFAULT_PROVIDERS}
        if custom_providers:
            provider_configs.update(custom_providers)

        # 初始化所有提供商
        for name, config in provider_configs.items():
            if not config.enabled:
                continue

            # 从环境变量获取配置
            api_key = os.getenv(config.api_key_env)
            base_url = os.getenv(config.base_url_env) if config.base_url_env else None

            # 创建提供商实例
            provider = config.provider_class(
                api_key=api_key,
                base_url=base_url,
                **config.custom_config
            )

            # 只注册可用的提供商
            if provider.is_available():
                self.providers[name] = provider
                print(f"✅ 已加载提供商: {provider.provider_name}")
            else:
                print(f"⚠️ 提供商 {name} 不可用（未配置 API Key）")

    async def discover_all_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        发现所有可用模型

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            所有可用模型列表
        """
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

    def get_models_by_provider(self, provider_name: str) -> List[ModelInfo]:
        """获取指定提供商的所有模型"""
        return self._models_by_provider.get(provider_name, [])

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """根据模型ID获取模型信息"""
        for model in self._all_models:
            if model.model_id == model_id:
                return model
        return None

    def filter_models(
        self,
        capabilities: Optional[List[str]] = None,
        cost_tier: Optional[str] = None,
        speed_tier: Optional[str] = None,
        min_context_window: Optional[int] = None
    ) -> List[ModelInfo]:
        """
        根据条件筛选模型

        Args:
            capabilities: 需要的能力列表
            cost_tier: 成本等级
            speed_tier: 速度等级
            min_context_window: 最小上下文窗口

        Returns:
            符合条件的模型列表
        """
        filtered = self._all_models

        if capabilities:
            filtered = [
                m for m in filtered
                if any(cap in [c.value for c in m.capabilities] for cap in capabilities)
            ]

        if cost_tier:
            filtered = [m for m in filtered if m.cost_tier == cost_tier]

        if speed_tier:
            filtered = [m for m in filtered if m.speed_tier == speed_tier]

        if min_context_window:
            filtered = [m for m in filtered if m.context_window >= min_context_window]

        return filtered

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用指定模型

        Args:
            model_id: 模型ID
            messages: 对话消息
            **kwargs: 其他参数

        Returns:
            模型响应
        """
        # 查找模型所属的提供商
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

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有提供商的状态信息"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = provider.validate_config()
        return status

    def add_custom_provider(
        self,
        name: str,
        provider_instance: BaseProvider
    ):
        """
        动态添加自定义提供商

        Args:
            name: 提供商名称
            provider_instance: 提供商实例
        """
        if provider_instance.is_available():
            self.providers[name] = provider_instance
            print(f"✅ 已添加自定义提供商: {name}")
        else:
            print(f"⚠️ 自定义提供商 {name} 不可用")
