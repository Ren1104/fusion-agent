"""
Langfuse 追踪工具
用于监控所有大模型调用，记录输入输出和 Token 消耗
"""

import os
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan
from langfuse.types import TraceContext

load_dotenv()


class LangfuseTracer:
    """Langfuse 追踪器封装类（兼容新版 SDK）"""

    def __init__(self):
        """初始化 Langfuse 客户端"""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST")

        self.enabled = all([public_key, secret_key, host])
        self.client: Optional[Langfuse] = None

        if not self.enabled:
            print("⚠️ Langfuse 未配置，追踪功能已禁用")
            return

        try:
            # Langfuse >=2.0 推荐使用 base_url 参数
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=host,
            )
            print("✅ Langfuse 追踪已启用")
        except Exception as exc:
            print(f"⚠️ Langfuse 初始化失败: {exc}")
            self.enabled = False

    # ------------------------------------------------------------------ #
    # Trace & Span helpers
    # ------------------------------------------------------------------ #

    def create_trace(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ) -> Optional[LangfuseSpan]:
        """
        创建一个新的根 span 作为 trace，对应一次完整的工作流运行。
        """
        if not self.enabled or not self.client:
            return None

        meta = dict(metadata or {})
        if user_id:
            meta["user_id"] = user_id

        try:
            span = self.client.start_observation(
                name=name,
                as_type="span",
                input=input_data,
                metadata=meta or None,
            )
            return span
        except Exception as exc:
            print(f"⚠️ 创建 trace 失败: {exc}")
            return None

    def create_span(
        self,
        trace_id: Optional[str],
        *,
        name: str,
        parent_observation_id: Optional[str] = None,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LangfuseSpan]:
        """在指定 trace 下创建一个子 span（用于节点级别追踪）"""
        if not self.enabled or not self.client or not trace_id:
            return None

        trace_context: TraceContext = {"trace_id": trace_id}
        if parent_observation_id:
            trace_context["parent_span_id"] = parent_observation_id

        try:
            span = self.client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="span",
                input=input_data,
                metadata=metadata or None,
            )
            return span
        except Exception as exc:
            print(f"⚠️ 创建 span 失败: {exc}")
            return None

    def start_generation(
        self,
        trace_id: Optional[str],
        *,
        name: str,
        model: str,
        parent_observation_id: Optional[str] = None,
        input_messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[LangfuseGeneration]:
        """开始一次模型调用的 generation 追踪"""
        if not self.enabled or not self.client or not trace_id:
            return None

        trace_context: TraceContext = {"trace_id": trace_id}
        if parent_observation_id:
            trace_context["parent_span_id"] = parent_observation_id

        try:
            generation = self.client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="generation",
                input=input_messages,
                metadata=metadata or None,
                model=model,
                model_parameters=model_parameters or None,
            )
            return generation
        except Exception as exc:
            print(f"⚠️ 创建 generation 失败: {exc}")
            return None

    def finish_observation(
        self,
        observation: Optional[Any],
        *,
        output_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """统一的 observation 结束逻辑（适用于 span 与 generation）"""
        if not observation or not self.enabled:
            return

        try:
            observation.update(
                output=output_data,
                metadata=metadata or None,
                usage_details=usage,
                level=level,
                status_message=status_message,
            )
        except Exception as exc:
            print(f"⚠️ 更新 observation 失败: {exc}")

        try:
            observation.end()
        except Exception as exc:
            print(f"⚠️ 结束 observation 失败: {exc}")

    def flush(self):
        """确保所有数据都已上传到 Langfuse"""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as exc:
                print(f"⚠️ Langfuse flush 失败: {exc}")


# 全局 tracer 实例
_global_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """获取全局 Langfuse tracer 实例"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = LangfuseTracer()
    return _global_tracer


def create_trace(
    name: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_data: Optional[Any] = None,
) -> Optional[LangfuseSpan]:
    """便捷函数：创建 trace"""
    return get_tracer().create_trace(name, user_id=user_id, metadata=metadata, input_data=input_data)


def create_span(
    trace_id: Optional[str],
    name: str,
    parent_observation_id: Optional[str] = None,
    input_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[LangfuseSpan]:
    """便捷函数：创建 span"""
    return get_tracer().create_span(
        trace_id,
        name=name,
        parent_observation_id=parent_observation_id,
        input_data=input_data,
        metadata=metadata,
    )


def start_generation(
    trace_id: Optional[str],
    name: str,
    model: str,
    parent_observation_id: Optional[str] = None,
    input_messages: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
) -> Optional[LangfuseGeneration]:
    """便捷函数：开始记录一次模型 generation"""
    return get_tracer().start_generation(
        trace_id,
        name=name,
        model=model,
        parent_observation_id=parent_observation_id,
        input_messages=input_messages,
        metadata=metadata,
        model_parameters=model_parameters,
    )


def finish_observation(
    observation: Optional[Any],
    *,
    output_data: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    usage: Optional[Dict[str, int]] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
) -> None:
    """便捷函数：结束 span/generation"""
    get_tracer().finish_observation(
        observation,
        output_data=output_data,
        metadata=metadata,
        usage=usage,
        level=level,
        status_message=status_message,
    )


def flush():
    """便捷函数：flush 数据"""
    get_tracer().flush()
