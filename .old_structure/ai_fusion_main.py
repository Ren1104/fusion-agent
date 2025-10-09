"""
AI Fusion - 基于PocketFlow实现的智能模型融合系统
根据用户问题自动选择三个最合适的LLM，并发调用后融合回答
"""

import asyncio
import sys
import os

# 添加PocketFlow路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PocketFlow'))

from pocketflow import AsyncNode, AsyncFlow, AsyncParallelBatchNode
from ai_fusion_nodes import ModelSelectorNode, ParallelLLMNode, FusionAgentNode
from ai_fusion_utils import get_available_models, validate_environment
from ai_fusion_reporter import AIFusionReporter
from ai_fusion_quality_analyzer import AIFusionQualityAnalyzer


class AIFusionFlow:
    """AI Fusion主流程控制器"""

    def __init__(self):
        self.setup_nodes()
        self.setup_flow()
        self.reporter = AIFusionReporter()
        self.quality_analyzer = AIFusionQualityAnalyzer()

    def setup_nodes(self):
        """初始化所有节点"""
        self.model_selector = ModelSelectorNode()
        self.parallel_llm = ParallelLLMNode()
        self.fusion_agent = FusionAgentNode()

    def setup_flow(self):
        """设置流程连接"""
        # 建立节点连接链，使用PocketFlow的正确语法
        self.model_selector - "continue" >> self.parallel_llm
        self.parallel_llm - "continue" >> self.fusion_agent

        # 创建主流程
        self.flow = AsyncFlow(start=self.model_selector)

    async def process_question(self, question: str) -> str:
        """
        处理用户问题的主入口
        
        Args:
            question: 用户输入的问题
            
        Returns:
            融合后的最终回答
        """
        shared = {
            "user_question": question,
            "available_models": get_available_models(),
            "selected_models": [],
            "llm_responses": [],
            "final_answer": ""
        }

        print(f"\n🤖 AI Fusion 正在处理您的问题: {question}")
        print("=" * 50)

        try:
            # 运行AI Fusion流程
            result = await self.flow.run_async(shared)
            final_answer = shared.get("final_answer", "处理失败")
            
            # 生成分析报告
            if shared.get("llm_responses") and shared.get("question_type"):
                # 执行质量分析
                print("\n🔍 正在进行质量分析...")
                quality_analysis = await self.quality_analyzer.analyze_quality(
                    question=question,
                    llm_responses=shared["llm_responses"],
                    fusion_answer=final_answer
                )
                
                # 打印简要摘要
                self.reporter.print_summary(shared["llm_responses"], final_answer, quality_analysis)
                
                # 生成详细报告
                selected_model_names = [m.name for m in shared.get("selected_models", [])]
                report_path = self.reporter.generate_report(
                    question=question,
                    question_type=shared.get("question_type", "未知"),
                    llm_responses=shared["llm_responses"],
                    final_answer=final_answer,
                    selected_models=selected_model_names,
                    quality_analysis=quality_analysis,
                    selection_analysis=shared.get("selection_analysis", {})
                )
            
            return final_answer

        except Exception as e:
            print(f"❌ 处理过程中出现错误: {str(e)}")
            return f"抱歉，处理您的问题时出现了错误: {str(e)}"


async def interactive_chat():
    """交互式聊天界面"""
    print("🌟 欢迎使用 AI Fusion 智能融合系统!")
    print("本系统会自动选择三个最合适的LLM模型来回答您的问题")
    print("输入 'exit' 或 'quit' 退出程序")
    print("=" * 60)

    # 验证环境
    if not validate_environment():
        print("❌ 环境验证失败，请检查API密钥配置")
        return

    ai_fusion = AIFusionFlow()

    while True:
        try:
            user_input = input("\n💬 请输入您的问题: ").strip()

            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 感谢使用 AI Fusion，再见!")
                break

            if not user_input:
                print("⚠️ 请输入有效的问题")
                continue

            # 处理用户问题
            answer = await ai_fusion.process_question(user_input)
            print(f"\n🎯 AI Fusion 融合回答:\n{answer}")
            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生未预期的错误: {str(e)}")


def main():
    """主函数"""
    print("🚀 启动 AI Fusion 系统...")

    try:
        asyncio.run(interactive_chat())
    except Exception as e:
        print(f"❌ 系统启动失败: {str(e)}")


if __name__ == "__main__":
    main()
