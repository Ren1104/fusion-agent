"""
Alibaba 提供商实现
支持通义千问系列模型
"""

import os
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI

from ai_fusion.providers.base import (
    BaseProvider,
    ModelInfo,
    ModelCapability,
    PerformanceLevel
)


class AlibabaProvider(BaseProvider):
    """Alibaba 通义千问 API 提供商"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            **kwargs
        )
        self.client = None
        if self.is_available():
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def provider_name(self) -> str:
        return "alibaba"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        """发现通义千问可用模型"""
        if not self.is_available():
            return []

        models = [
            ModelInfo(
                model_id="qwen-max",
                display_name="通义千问 Max",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.MULTILINGUAL,
                    ModelCapability.LONG_CONTEXT
                ],
                strengths=[
                    "中文理解顶尖",
                    "知识覆盖广",
                    "多语言支持强",
                    "推理能力强"
                ],
                suitable_tasks=[
                    "中文内容处理",
                    "多语言翻译",
                    "知识问答",
                    "专业文档"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.EXCELLENT,
                    "creativity": PerformanceLevel.GOOD,
                    "coding": PerformanceLevel.GOOD,
                    "factual": PerformanceLevel.EXCELLENT,
                },
                context_window=32000,
                max_output_tokens=8192,
                cost_tier="medium",
                speed_tier="medium",
                special_features=["中文优化", "知识丰富"]
            ),
            ModelInfo(
                model_id="qwen-plus",
                display_name="通义千问 Plus",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.MULTILINGUAL,
                    ModelCapability.FAST_RESPONSE
                ],
                strengths=[
                    "中文处理优秀",
                    "平衡性能好",
                    "性价比高"
                ],
                suitable_tasks=[
                    "中文问答",
                    "通用任务",
                    "信息处理"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.GOOD,
                    "creativity": PerformanceLevel.GOOD,
                    "coding": PerformanceLevel.GOOD,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=32000,
                max_output_tokens=8192,
                cost_tier="low",
                speed_tier="fast",
                special_features=["中文友好", "性价比高"]
            ),
            ModelInfo(
                model_id="qwen-turbo",
                display_name="通义千问 Turbo",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.FAST_RESPONSE
                ],
                strengths=[
                    "极速响应",
                    "轻量高效"
                ],
                suitable_tasks=[
                    "快速问答",
                    "简单任务"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.MEDIUM,
                    "creativity": PerformanceLevel.MEDIUM,
                    "coding": PerformanceLevel.MEDIUM,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=8000,
                max_output_tokens=2000,
                cost_tier="low",
                speed_tier="very_fast",
                special_features=["极速响应", "超低成本"]
            ),
        ]

        # 支持自定义模型扩展
        custom_models = self.config.get("custom_models", [])
        for custom_model in custom_models:
            models.append(ModelInfo(**custom_model))

        return models

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """调用通义千问模型"""
        if not self.client:
            raise ValueError(f"{self.provider_name} provider is not available")

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return response.choices[0].message.content
