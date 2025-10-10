"""
AI Fusion节点实现
包含模型选择器、并发LLM调用和回答融合等核心节点
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime

from pocketflow import AsyncNode
from ai_fusion.utils.utils import call_llm_async, ModelConfig
from ai_fusion.analysis.smart_selector import AIFusionSmartSelector


class ModelSelectorNode(AsyncNode):
    """
    模型选择器节点
    根据用户问题的特征自动选择三个最合适的LLM模型
    """

    def __init__(self):
        super().__init__(max_retries=2, wait=1)
        self.smart_selector = AIFusionSmartSelector()
        # 保留传统选择策略作为回退
        self.fallback_criteria = {
            "技术/编程": ["gpt-41-0414-global", "claude_sonnet4", "qwen-max"],
            "创意写作": ["claude_sonnet4", "qwen-max", "claude37_sonnet_new"],
            "数学/逻辑": ["gpt-41-0414-global", "claude_sonnet4", "qwen-max"],
            "日常对话": ["gpt-41-mini-0414-global", "qwen-plus", "claude37_sonnet_new"],
            "专业知识": ["claude_sonnet4", "qwen-max", "gpt-41-0414-global"],
            "翻译": ["qwen-max", "claude_sonnet4", "gpt-41-0414-global"],
            "分析总结": ["claude_sonnet4", "qwen-max", "gpt-41-0414-global"],
            "默认": ["claude_sonnet4", "gpt-41-0414-global", "qwen-max"]
        }

    async def prep_async(self, shared):
        """准备阶段：获取用户问题和可用模型"""
        question = shared.get("user_question", "")
        available_models = shared.get("available_models", [])

        if not question:
            raise ValueError("用户问题不能为空")

        if not available_models:
            raise ValueError("没有可用的模型")

        return {
            "question": question,
            "available_models": available_models
        }

    async def exec_async(self, inputs):
        """执行阶段：智能分析问题并选择合适的模型"""
        question = inputs["question"]
        available_models = inputs["available_models"]

        print("🧠 正在进行智能模型选择分析...")

        try:
            # 使用智能选择器
            recommendation = await self.smart_selector.intelligent_model_selection(
                question, available_models
            )

            selected_models = recommendation.get('selected_models', [])
            analysis_method = recommendation.get('analysis_method', 'intelligent_llm')

            # 显示分析结果
            self._display_selection_analysis(recommendation)

            return {
                "selected_model_names": selected_models,
                "question_type": recommendation.get('problem_analysis', {}).get('question_type', '智能分析'),
                "selection_analysis": recommendation,
                "analysis_method": analysis_method
            }

        except Exception as e:
            print(f"⚠️ 智能模型选择失败，使用传统方法: {str(e)}")
            return await self._fallback_selection(question, available_models)

    def _display_selection_analysis(self, recommendation: Dict[str, Any]):
        """显示选择分析结果"""
        problem_analysis = recommendation.get('problem_analysis', {})
        recommended_models = recommendation.get('recommended_models', [])

        print(f"📋 问题分析:")
        print(f"   类型: {problem_analysis.get('question_type', '未知')}")
        print(f"   复杂度: {problem_analysis.get('complexity_level', '未知')}")
        print(f"   所需能力: {', '.join(problem_analysis.get('required_capabilities', []))}")

        print(f"🎯 推荐模型组合:")
        for model in recommended_models:
            score = model.get('suitability_score', 0)
            reasons = ', '.join(model.get('reasons', []))
            print(f"   {model.get('rank', 0)}. {model.get('model_name', '')} (适合度: {score}/10)")
            print(f"      理由: {reasons}")
            print(f"      贡献: {model.get('expected_contribution', '')}")

        strategy = recommendation.get('combination_strategy', '')
        confidence = recommendation.get('confidence_level', '')
        if strategy:
            print(f"🔗 组合策略: {strategy}")
        if confidence:
            print(f"🎯 置信度: {confidence}")

    async def _fallback_selection(self, question: str, available_models: List[ModelConfig]) -> Dict[str, Any]:
        """传统选择方法作为回退"""
        print("🔄 使用传统问题类型分析...")

        # 简化的问题类型分析
        analysis_prompt = f"""
请分析以下问题的类型，从以下类别中选择最适合的一个：
- 技术/编程
- 创意写作  
- 数学/逻辑
- 日常对话
- 专业知识
- 翻译
- 分析总结

问题: {question}

请只返回类别名称，不要额外解释。
"""

        try:
            question_type = await call_llm_async(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="gpt-41-0414-global"
            )
            question_type = question_type.strip()
            print(f"📊 问题类型: {question_type}")

        except Exception as e:
            print(f"⚠️ 类型分析失败，使用默认: {str(e)}")
            question_type = "默认"

        # 使用传统选择策略
        preferred_models = self.fallback_criteria.get(question_type, self.fallback_criteria["默认"])
        available_model_names = [model.name for model in available_models]

        selected_models = []
        for preferred in preferred_models:
            if preferred in available_model_names and len(selected_models) < 3:
                selected_models.append(preferred)

        # 补充不足的模型
        while len(selected_models) < 3 and len(selected_models) < len(available_model_names):
            for model_name in available_model_names:
                if model_name not in selected_models:
                    selected_models.append(model_name)
                    break

        print(f"✅ 选择结果: {selected_models}")

        return {
            "selected_model_names": selected_models,
            "question_type": question_type,
            "analysis_method": "fallback"
        }

    async def post_async(self, shared, prep_res, exec_res):
        """后处理阶段：保存选择的模型到共享状态"""
        if exec_res:
            selected_models = []
            available_models = shared["available_models"]

            # 构建选择的模型配置列表
            for model_name in exec_res["selected_model_names"]:
                for model_config in available_models:
                    if model_config.name == model_name:
                        selected_models.append(model_config)
                        break

            shared["selected_models"] = selected_models
            shared["question_type"] = exec_res["question_type"]
            shared["selection_analysis"] = exec_res.get("selection_analysis", {})
            shared["analysis_method"] = exec_res.get("analysis_method", "unknown")

            print(f"✅ 已选择 {len(selected_models)} 个模型: {[m.name for m in selected_models]}")
            return "continue"

        return None


class ParallelLLMNode(AsyncNode):
    """
    并发LLM调用节点
    同时调用三个选定的LLM模型获取回答
    """

    def __init__(self):
        super().__init__(max_retries=2, wait=1)

    async def prep_async(self, shared):
        """准备阶段：获取选定的模型和用户问题"""
        question = shared.get("user_question", "")
        selected_models = shared.get("selected_models", [])

        if not question:
            raise ValueError("用户问题不能为空")

        if not selected_models:
            raise ValueError("没有选定的模型")

        return {
            "question": question,
            "models": selected_models
        }

    async def call_single_llm(self, model_config: ModelConfig, question: str, model_index: int):
        """调用单个LLM模型"""
        start_time = time.time()
        try:
            print(f"🤖 正在调用模型 {model_index + 1}: {model_config.name}")

            messages = [
                {"role": "user", "content": question}
            ]

            response = await call_llm_async(
                messages=messages,
                model=model_config.name,
                api_key=model_config.api_key,
                base_url=model_config.base_url
            )

            end_time = time.time()
            response_time = end_time - start_time

            print(f"✅ 模型 {model_index + 1} ({model_config.name}) 回答完成，耗时: {response_time:.2f}秒")
            print(f"📝 模型 {model_index + 1} 响应内容:")
            print(f"{'=' * 50}")
            print(response[:200] + "..." if len(response) > 200 else response)
            print(f"{'=' * 50}\n")

            return {
                "model_name": model_config.name,
                "response": response,
                "response_time": response_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ 模型 {model_index + 1} ({model_config.name}) 调用失败: {str(e)}，耗时: {response_time:.2f}秒")
            return {
                "model_name": model_config.name,
                "response": "",
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def exec_async(self, inputs):
        """执行阶段：并发调用所有选定的LLM"""
        question = inputs["question"]
        models = inputs["models"]

        print(f"🚀 开始并发调用 {len(models)} 个LLM模型...")

        # 并发调用所有模型
        tasks = [
            self.call_single_llm(model, question, i)
            for i, model in enumerate(models)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        successful_responses = []
        failed_responses = []

        for result in results:
            if isinstance(result, Exception):
                failed_responses.append({
                    "model_name": "unknown",
                    "error": str(result),
                    "success": False
                })
            elif result["success"]:
                successful_responses.append(result)
            else:
                failed_responses.append(result)

        print(f"📊 调用结果: {len(successful_responses)} 成功, {len(failed_responses)} 失败")

        return {
            "successful_responses": successful_responses,
            "failed_responses": failed_responses
        }

    async def post_async(self, shared, prep_res, exec_res):
        """后处理阶段：保存LLM回答到共享状态"""
        if exec_res:
            shared["llm_responses"] = exec_res["successful_responses"]
            shared["failed_responses"] = exec_res["failed_responses"]

            if exec_res["successful_responses"]:
                print("✅ LLM并发调用完成，开始融合回答...")
                return "continue"
            else:
                print("❌ 所有LLM调用都失败了")
                return None

        return None


class FusionAgentNode(AsyncNode):
    """
    回答融合代理节点
    分析和融合多个LLM的回答，生成最终的综合回答
    """

    def __init__(self):
        super().__init__(max_retries=3, wait=2)

    async def prep_async(self, shared):
        """准备阶段：获取所有LLM的回答"""
        question = shared.get("user_question", "")
        responses = shared.get("llm_responses", [])
        question_type = shared.get("question_type", "未知")

        if not question:
            raise ValueError("用户问题不能为空")

        if not responses:
            raise ValueError("没有LLM回答可供融合")

        return {
            "question": question,
            "responses": responses,
            "question_type": question_type
        }

    async def exec_async(self, inputs):
        """执行阶段：融合多个LLM的回答"""
        question = inputs["question"]
        responses = inputs["responses"]
        question_type = inputs["question_type"]

        print("🧠 正在使用AI代理融合多个回答...")

        # 构建融合提示
        fusion_prompt = self._build_fusion_prompt(question, responses, question_type)

        try:
            # 使用高质量模型进行融合
            fused_answer = await call_llm_async(
                messages=[{"role": "user", "content": fusion_prompt}],
                model="claude_sonnet4"  # 使用允许的高质量模型进行融合
            )

            return fused_answer

        except Exception as e:
            print(f"❌ 回答融合失败: {str(e)}")
            # 如果融合失败，返回第一个可用的回答
            if responses:
                return f"融合失败，以下是第一个模型的回答：\n\n{responses[0]['response']}"
            else:
                return "抱歉，无法生成回答。"

    def _build_fusion_prompt(self, question: str, responses: List[Dict], question_type: str) -> str:
        """构建融合提示"""
        prompt = f"""你是一个专业的AI回答融合专家。用户提出了一个关于"{question_type}"的问题，我已经从多个AI模型获得了不同的回答。

请分析这些回答，提取各自的优点，然后融合成一个全面、准确、有用的最终回答。

用户问题: {question}

各模型的回答:
"""

        for i, response in enumerate(responses, 1):
            prompt += f"\n【模型 {i}: {response['model_name']}】\n{response['response']}\n"

        prompt += """
请根据以上回答，生成一个融合后的最终回答。要求：

1. 综合各个回答的优点和有用信息
2. 确保答案准确、完整且易于理解
3. 如果各回答有冲突，请指出并给出最可靠的信息
4. 保持回答的逻辑性和条理性
5. 适当添加补充信息使回答更全面

最终融合回答:"""

        return prompt

    async def post_async(self, shared, prep_res, exec_res):
        """后处理阶段：保存融合后的最终回答"""
        if exec_res:
            shared["final_answer"] = exec_res
            print("✅ 回答融合完成！")
            return "analyze"  # 继续到质量分析节点

        return None


class QualityAnalyzerNode(AsyncNode):
    """
    质量分析节点
    对融合回答和各模型回答进行质量分析
    """

    def __init__(self):
        super().__init__(max_retries=2, wait=1)
        from ai_fusion.analysis.quality_analyzer import AIFusionQualityAnalyzer
        self.analyzer = AIFusionQualityAnalyzer()

    async def prep_async(self, shared):
        """准备阶段：获取问题、回答和融合结果"""
        question = shared.get("user_question", "")
        llm_responses = shared.get("llm_responses", [])
        final_answer = shared.get("final_answer", "")

        if not question:
            raise ValueError("用户问题不能为空")

        if not llm_responses:
            print("⚠️ 没有LLM回答，跳过质量分析")
            return None

        return {
            "question": question,
            "llm_responses": llm_responses,
            "final_answer": final_answer
        }

    async def exec_async(self, inputs):
        """执行阶段：进行质量分析"""
        if inputs is None:
            return None

        print("\n🔍 正在进行质量分析...")

        quality_analysis = await self.analyzer.analyze_quality(
            question=inputs["question"],
            llm_responses=inputs["llm_responses"],
            fusion_answer=inputs["final_answer"]
        )

        return quality_analysis

    async def post_async(self, shared, prep_res, exec_res):
        """后处理阶段：保存质量分析结果"""
        if exec_res:
            shared["quality_analysis"] = exec_res
            print("✅ 质量分析完成！")
            return "report"  # 继续到报告生成节点
        else:
            # 如果没有分析结果，直接跳到报告节点
            return "report"


class ReportGeneratorNode(AsyncNode):
    """
    报告生成节点
    生成详细的Markdown分析报告
    """

    def __init__(self):
        super().__init__(max_retries=1, wait=1)
        from ai_fusion.reporting.reporter import AIFusionReporter
        self.reporter = AIFusionReporter()

    async def prep_async(self, shared):
        """准备阶段：收集所有必要的数据"""
        question = shared.get("user_question", "")
        question_type = shared.get("question_type", "未知")
        llm_responses = shared.get("llm_responses", [])
        final_answer = shared.get("final_answer", "")
        quality_analysis = shared.get("quality_analysis")
        selection_analysis = shared.get("selection_analysis", {})
        selected_models = shared.get("selected_models", [])

        return {
            "question": question,
            "question_type": question_type,
            "llm_responses": llm_responses,
            "final_answer": final_answer,
            "selected_models": [m.name for m in selected_models],
            "quality_analysis": quality_analysis,
            "selection_analysis": selection_analysis
        }

    async def exec_async(self, inputs):
        """执行阶段：生成报告"""
        # 打印简要摘要
        if inputs["quality_analysis"]:
            self.reporter.print_summary(
                inputs["llm_responses"],
                inputs["final_answer"],
                inputs["quality_analysis"]
            )

        # 生成详细报告
        report_path = self.reporter.generate_report(
            question=inputs["question"],
            question_type=inputs["question_type"],
            llm_responses=inputs["llm_responses"],
            final_answer=inputs["final_answer"],
            selected_models=inputs["selected_models"],
            quality_analysis=inputs["quality_analysis"],
            selection_analysis=inputs["selection_analysis"]
        )

        return report_path

    async def post_async(self, shared, prep_res, exec_res):
        """后处理阶段：保存报告路径"""
        if exec_res:
            shared["report_path"] = exec_res
            print(f"✅ 报告已生成: {exec_res}")
            return "complete"  # 流程结束

        return None
