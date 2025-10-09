"""
Anthropic 提供商实现
支持 Claude 系列模型
"""

import os
from typing import List, Dict, Optional, Any
from anthropic import AsyncAnthropic

from ai_fusion.providers.base import (
    BaseProvider,
    ModelInfo,
    ModelCapability,
    PerformanceLevel
)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API 提供商"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self.client = None
        if self.is_available():
            self.client = AsyncAnthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        """发现 Claude 可用模型"""
        if not self.is_available():
            return []

        models = [
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.REASONING,
                    ModelCapability.CODING,
                    ModelCapability.CREATIVITY,
                    ModelCapability.LONG_CONTEXT
                ],
                strengths=[
                    "卓越的逻辑推理",
                    "顶级代码能力",
                    "创意写作优秀",
                    "长文本理解",
                    "细致的上下文把握"
                ],
                suitable_tasks=[
                    "复杂编程",
                    "技术写作",
                    "创意内容",
                    "深度分析",
                    "系统设计"
                ],
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
                special_features=["200K上下文", "强逻辑推理", "道德安全"]
            ),
            ModelInfo(
                model_id="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                provider=self.provider_name,
                capabilities=[
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.CODING
                ],
                strengths=[
                    "极速响应",
                    "高效处理",
                    "性价比优秀"
                ],
                suitable_tasks=[
                    "快速问答",
                    "简单编程",
                    "文本处理"
                ],
                performance_profile={
                    "reasoning": PerformanceLevel.GOOD,
                    "creativity": PerformanceLevel.GOOD,
                    "coding": PerformanceLevel.GOOD,
                    "factual": PerformanceLevel.GOOD,
                },
                context_window=200000,
                max_output_tokens=4096,
                cost_tier="low",
                speed_tier="very_fast",
                special_features=["极速响应", "高性价比"]
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
        """调用 Claude 模型"""
        if not self.client:
            raise ValueError(f"{self.provider_name} provider is not available")

        # 转换消息格式（如果需要）
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
