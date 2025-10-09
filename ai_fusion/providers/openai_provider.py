"""
OpenAI 提供商实现
支持 GPT 系列模型
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


class OpenAIProvider(BaseProvider):
    """OpenAI API 提供商"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs
        )
        self.client = None
        if self.is_available():
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        """
        发现 OpenAI 可用模型

        注意：这里使用静态定义，因为 OpenAI 的 models API 返回的信息有限
        如果有 API key，可以通过 client.models.list() 动态获取
        """
        if not self.is_available():
            return []

        # 定义 OpenAI 模型知识库
        models = [
            ModelInfo(
                model_id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.CODING,
                    ModelCapability.MATH,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION
                ],
                strengths=[
                    "强大的逻辑推理能力",
                    "优秀的代码生成",
                    "数学和科学计算",
                    "支持视觉理解",
                    "函数调用能力"
                ],
                suitable_tasks=[
                    "复杂编程任务",
                    "数学问题求解",
                    "逻辑推理",
                    "系统设计",
                    "代码审查"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.EXCELLENT,
                    "creativity": PerformanceLevel.GOOD,
                    "coding": PerformanceLevel.EXCELLENT,
                    "factual": PerformanceLevel.EXCELLENT,
                },
                context_window=128000,
                max_output_tokens=4096,
                cost_tier="high",
                speed_tier="medium",
                special_features=["视觉理解", "函数调用", "JSON模式"]
            ),
            ModelInfo(
                model_id="gpt-4o-mini",
                display_name="GPT-4o Mini",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.CODING,
                    ModelCapability.FAST_RESPONSE
                ],
                strengths=[
                    "快速响应",
                    "性价比高",
                    "轻量级任务",
                    "批量处理"
                ],
                suitable_tasks=[
                    "日常问答",
                    "简单编程",
                    "文本处理",
                    "快速原型"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.GOOD,
                    "creativity": PerformanceLevel.MEDIUM,
                    "coding": PerformanceLevel.GOOD,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=128000,
                max_output_tokens=16384,
                cost_tier="low",
                speed_tier="very_fast",
                special_features=["极速响应", "高性价比"]
            ),
        ]

        # 如果配置了自定义模型列表，可以从这里扩展
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
        """调用 OpenAI 模型"""
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
