"""
基础提供商抽象类
定义所有模型提供商必须实现的接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


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
    # 基本信息
    model_id: str                           # 模型唯一标识
    display_name: str                       # 显示名称
    provider: str                           # 提供商名称

    # 能力描述
    capabilities: List[ModelCapability] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    suitable_tasks: List[str] = field(default_factory=list)

    # 性能指标
    performance_profile: Dict[str, PerformanceLevel] = field(default_factory=dict)

    # 技术参数
    context_window: int = 128000            # 上下文窗口大小
    max_output_tokens: int = 4096          # 最大输出token数
    supports_streaming: bool = True         # 是否支持流式输出

    # 成本和速度（相对值，用于排序）
    cost_tier: str = "medium"              # low/medium/high
    speed_tier: str = "medium"             # slow/medium/fast/very_fast

    # 元数据
    is_available: bool = True               # 是否可用
    version: Optional[str] = None          # 版本信息
    special_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        # 如果 performance_profile 为空，设置默认值
        if not self.performance_profile:
            self.performance_profile = {
                "reasoning": PerformanceLevel.MEDIUM,
                "creativity": PerformanceLevel.MEDIUM,
                "coding": PerformanceLevel.MEDIUM,
                "factual": PerformanceLevel.MEDIUM,
            }


class BaseProvider(ABC):
    """
    基础模型提供商抽象类
    所有具体的提供商必须继承此类并实现所有抽象方法
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        初始化提供商

        Args:
            api_key: API密钥
            base_url: API基础URL
            **kwargs: 其他配置参数
        """
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
        """
        发现并返回该提供商支持的所有模型

        Returns:
            模型信息列表
        """
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
        """
        调用模型生成回答

        Args:
            model_id: 模型ID
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            模型生成的回答
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查提供商是否可用（API密钥是否配置）

        Returns:
            是否可用
        """
        pass

    async def get_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        获取模型列表（带缓存）

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            模型信息列表
        """
        if self._models_cache is None or force_refresh:
            self._models_cache = await self.discover_models()
        return self._models_cache

    def validate_config(self) -> Dict[str, Any]:
        """
        验证配置

        Returns:
            验证结果字典
        """
        return {
            "valid": self.is_available(),
            "provider": self.provider_name,
            "has_api_key": bool(self.api_key),
            "base_url": self.base_url
        }
