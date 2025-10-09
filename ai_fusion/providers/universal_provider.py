"""
通用 OpenAI 兼容提供商
支持所有通过 OpenAI 兼容接口调用的模型
这是当前项目的主要提供商
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


class UniversalProvider(BaseProvider):
    """
    通用 OpenAI 兼容 API 提供商

    支持任何实现了 OpenAI Chat Completions API 的服务
    包括：官方 OpenAI、Azure OpenAI、第三方代理服务等
    """

    # 默认模型知识库（基于你当前项目的模型列表）
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
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs
        )
        self.client = None
        if self.is_available():
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # 支持自定义模型列表
        self.custom_models_list = kwargs.get("models_list", None)

    @property
    def provider_name(self) -> str:
        return "universal"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def discover_models(self) -> List[ModelInfo]:
        """发现可用模型"""
        if not self.is_available():
            return []

        models = []

        # 如果提供了自定义模型列表，使用它
        if self.custom_models_list:
            model_ids = self.custom_models_list
        # 否则从环境变量读取
        elif os.getenv("AVAILABLE_MODELS"):
            model_ids = [m.strip() for m in os.getenv("AVAILABLE_MODELS").split(",")]
        # 使用默认模型列表
        else:
            model_ids = list(self.DEFAULT_MODELS_KNOWLEDGE.keys())

        # 为每个模型创建 ModelInfo
        for model_id in model_ids:
            if model_id in self.DEFAULT_MODELS_KNOWLEDGE:
                # 使用已知的模型信息
                info = self.DEFAULT_MODELS_KNOWLEDGE[model_id]
                models.append(ModelInfo(
                    model_id=model_id,
                    provider=self.provider_name,
                    **info
                ))
            else:
                # 未知模型，使用默认配置
                models.append(ModelInfo(
                    model_id=model_id,
                    display_name=model_id,
                    provider=self.provider_name,
                    capabilities=[ModelCapability.REASONING],
                    strengths=["通用AI能力"],
                    suitable_tasks=["通用任务"],
                    performance_profile={
                        "reasoning": PerformanceLevel.MEDIUM,
                        "creativity": PerformanceLevel.MEDIUM,
                        "coding": PerformanceLevel.MEDIUM,
                        "factual": PerformanceLevel.MEDIUM,
                    }
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
        """调用模型"""
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
