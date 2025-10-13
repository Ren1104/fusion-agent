"""
AI Fusion 流程编排
定义主流程创建函数和节点连接逻辑
"""

from pocketflow import AsyncFlow
from nodes import (
    ModelSelectorNode,
    ParallelLLMNode,
    FusionAgentNode,
    QualityAnalyzerNode,
    ReportGeneratorNode
)


def create_ai_fusion_flow():
    """
    创建并配置AI Fusion完整流程

    流程说明：
    1. ModelSelector   - 智能选择3个最适合的模型
    2. ParallelLLM     - 并发调用选中的模型
    3. FusionAgent     - 融合多个模型的回答
    4. QualityAnalyzer - 分析回答质量
    5. ReportGenerator - 生成详细报告

    Returns:
        AsyncFlow: 配置好的异步流程
    """
    # 创建节点实例
    model_selector = ModelSelectorNode()
    parallel_llm = ParallelLLMNode()
    fusion_agent = FusionAgentNode()
    quality_analyzer = QualityAnalyzerNode()
    report_generator = ReportGeneratorNode()

    # 连接节点（使用PocketFlow标准语法）
    model_selector - "continue" >> parallel_llm
    parallel_llm - "continue" >> fusion_agent
    fusion_agent - "analyze" >> quality_analyzer
    quality_analyzer - "report" >> report_generator

    # 创建并返回主流程
    flow = AsyncFlow(start=model_selector)

    return flow
