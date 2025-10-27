"""
AI Fusion 节点实现
包含模型选择器、并发LLM调用和回答融合等核心节点
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from pocketflow import AsyncNode
from analyzer import call_llm_async, ModelConfig, AIFusionSmartSelector, AIFusionQualityAnalyzer
from reporter import AIFusionReporter
from langfuse_tracer import create_span, finish_observation


class ModelSelectorNode(AsyncNode):
    """
    模型选择器节点
    根据用户问题的特征自动选择三个最合适的LLM模型
    """

    def __init__(self):
        super().__init__(max_retries=2, wait=1)
        self.smart_selector = None  # 将在 prep_async 中初始化
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
        """准备阶段：获取用户问题、可用模型、registry 和 trace_id"""
        question = shared.get("user_question", "")
        available_models = shared.get("available_models", [])
        registry = shared.get("registry")
        trace_id = shared.get("trace_id")
        trace_observation_id = shared.get("langfuse_trace_observation_id")

        if not question:
            raise ValueError("用户问题不能为空")

        if not available_models:
            raise ValueError("没有可用的模型")

        # 初始化 smart_selector（使用 registry）
        if self.smart_selector is None:
            self.smart_selector = AIFusionSmartSelector(registry=registry)

        return {
            "question": question,
            "available_models": available_models,
            "trace_id": trace_id,
            "trace_observation_id": trace_observation_id
        }

    async def exec_async(self, inputs):
        """执行阶段：智能分析问题并选择合适的模型"""
        question = inputs["question"]
        available_models = inputs["available_models"]
        trace_id = inputs.get("trace_id")
        parent_observation_id = inputs.get("trace_observation_id")

        selector_span = create_span(
            trace_id,
            name="ModelSelector",
            parent_observation_id=parent_observation_id,
            input_data={
                "question": question,
                "available_models": [model.name for model in available_models],
            },
            metadata={"node": "ModelSelector"},
        )
        current_parent_id = selector_span.id if selector_span else parent_observation_id

        print("🧠 正在进行智能模型选择分析...")

        try:
            # 使用智能选择器
            recommendation = await self.smart_selector.intelligent_model_selection(
                question,
                available_models,
                trace_id=trace_id,
                parent_observation_id=current_parent_id,
            )

            selected_models = recommendation.get('selected_models', [])
            analysis_method = recommendation.get('analysis_method', 'intelligent_llm')

            # 显示分析结果
            self._display_selection_analysis(recommendation)

            result = {
                "selected_model_names": selected_models,
                "question_type": recommendation.get('problem_analysis', {}).get('question_type', '智能分析'),
                "selection_analysis": recommendation,
                "analysis_method": analysis_method
            }

            finish_observation(
                selector_span,
                output_data={
                    "selected_model_names": selected_models,
                    "analysis_method": analysis_method,
                    "question_type": result["question_type"],
                },
                metadata={"node": "ModelSelector"},
            )

            return result

        except Exception as e:
            print(f"⚠️ 智能模型选择失败，使用传统方法: {str(e)}")
            result = await self._fallback_selection(
                question,
                available_models,
                trace_id=trace_id,
                parent_observation_id=current_parent_id,
            )
            finish_observation(
                selector_span,
                output_data={
                    "selected_model_names": result["selected_model_names"],
                    "analysis_method": result.get("analysis_method"),
                    "question_type": result.get("question_type"),
                },
                metadata={"node": "ModelSelector", "fallback": True},
                level="WARNING",
                status_message=str(e),
            )
            return result
        finally:
            if selector_span:
                # finish_observation 在 try/except 控制分支里已经调用，这里确保待处理对象已结束
                pass

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

    async def _fallback_selection(
        self,
        question: str,
        available_models: List[ModelConfig],
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                model="gpt-41-0414-global",
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                langfuse_metadata={"node": "ModelSelector", "stage": "fallback_type_detection"}
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
        """准备阶段：获取选定的模型、用户问题、registry 和 trace_id"""
        question = shared.get("user_question", "")
        selected_models = shared.get("selected_models", [])
        registry = shared.get("registry")
        trace_id = shared.get("trace_id")
        trace_observation_id = shared.get("langfuse_trace_observation_id")

        if not question:
            raise ValueError("用户问题不能为空")

        if not selected_models:
            raise ValueError("没有选定的模型")

        return {
            "question": question,
            "models": selected_models,
            "registry": registry,
            "trace_id": trace_id,
            "trace_observation_id": trace_observation_id
        }

    async def call_single_llm(
        self,
        model_config: ModelConfig,
        question: str,
        model_index: int,
        registry=None,
        trace_id=None,
        parent_observation_id: Optional[str] = None,
    ):
        """调用单个LLM模型"""
        start_time = time.time()
        try:
            print(f"🤖 正在调用模型 {model_index + 1}: {model_config.name}")

            messages = [
                {"role": "user", "content": question}
            ]

            # 使用 registry 调用模型，传递 trace_id
            response = await call_llm_async(
                messages=messages,
                model=model_config.name,
                registry=registry,
                trace_id=trace_id,
                return_response_obj=True,
                parent_observation_id=parent_observation_id,
                langfuse_metadata={
                    "component": "parallel_llm",
                    "model_index": model_index + 1,
                    "model_name": model_config.name,
                },
            )
            response_text = response.text
            usage_details = response.usage

            end_time = time.time()
            response_time = end_time - start_time

            print(f"✅ 模型 {model_index + 1} ({model_config.name}) 回答完成，耗时: {response_time:.2f}秒")
            print(f"📝 模型 {model_index + 1} 响应内容:")
            print(f"{'=' * 50}")
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            print(f"{'=' * 50}\n")

            return {
                "model_name": model_config.name,
                "response": response_text,
                "token_usage": usage_details,
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
        registry = inputs["registry"]
        trace_id = inputs["trace_id"]
        trace_observation_id = inputs.get("trace_observation_id")

        print(f"🚀 开始并发调用 {len(models)} 个LLM模型...")

        parallel_span = create_span(
            trace_id,
            name="ParallelLLM",
            parent_observation_id=trace_observation_id,
            input_data={
                "question": question,
                "selected_models": [model.name for model in models],
            },
            metadata={"node": "ParallelLLM"},
        )
        generation_parent_id = parallel_span.id if parallel_span else trace_observation_id

        # 并发调用所有模型，传递 trace_id
        tasks = [
            self.call_single_llm(
                model,
                question,
                i,
                registry=registry,
                trace_id=trace_id,
                parent_observation_id=generation_parent_id,
            )
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

        finish_observation(
            parallel_span,
            output_data={
                "successful": [
                    {
                        "model_name": r["model_name"],
                        "response_time": r["response_time"],
                        "token_usage": r.get("token_usage"),
                    }
                    for r in successful_responses
                ],
                "failed": [
                    {"model_name": r.get("model_name"), "error": r.get("error")}
                    for r in failed_responses
                ],
            },
            metadata={"node": "ParallelLLM"},
            level="ERROR" if successful_responses == [] else None,
        )

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
        """准备阶段：获取所有LLM的回答、registry 和 trace_id"""
        question = shared.get("user_question", "")
        responses = shared.get("llm_responses", [])
        question_type = shared.get("question_type", "未知")
        registry = shared.get("registry")
        trace_id = shared.get("trace_id")
        trace_observation_id = shared.get("langfuse_trace_observation_id")

        if not question:
            raise ValueError("用户问题不能为空")

        if not responses:
            raise ValueError("没有LLM回答可供融合")

        return {
            "question": question,
            "responses": responses,
            "question_type": question_type,
            "registry": registry,
            "trace_id": trace_id,
            "trace_observation_id": trace_observation_id
        }

    async def exec_async(self, inputs):
        """执行阶段：融合多个LLM的回答"""
        question = inputs["question"]
        responses = inputs["responses"]
        question_type = inputs["question_type"]
        registry = inputs["registry"]
        trace_id = inputs["trace_id"]
        trace_observation_id = inputs.get("trace_observation_id")

        print("🧠 正在使用AI代理融合多个回答...")

        # 构建融合提示
        fusion_prompt = self._build_fusion_prompt(question, responses, question_type)

        fusion_span = create_span(
            trace_id,
            name="FusionAgent",
            parent_observation_id=trace_observation_id,
            input_data={
                "question": question,
                "question_type": question_type,
                "model_count": len(responses),
            },
            metadata={"node": "FusionAgent"},
        )
        generation_parent_id = fusion_span.id if fusion_span else trace_observation_id

        try:
            # 使用高质量模型进行融合
            import os
            fusion_model = os.getenv("AI_FUSION_MODEL", "claude_sonnet4")

            response = await call_llm_async(
                messages=[{"role": "user", "content": fusion_prompt}],
                model=fusion_model,
                registry=registry,
                trace_id=trace_id,
                return_response_obj=True,
                parent_observation_id=generation_parent_id,
                langfuse_metadata={
                    "component": "fusion_agent",
                    "question_type": question_type,
                },
            )
            fused_answer = response.text

            finish_observation(
                fusion_span,
                output_data={
                    "fused_answer": fused_answer,
                    "token_usage": response.usage,
                },
                metadata={"node": "FusionAgent"},
            )

            return fused_answer

        except Exception as e:
            print(f"❌ 回答融合失败: {str(e)}")
            finish_observation(
                fusion_span,
                output_data={"error": str(e)},
                metadata={"node": "FusionAgent"},
                level="ERROR",
                status_message=str(e),
            )
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
        self.analyzer = None  # 将在 prep_async 中初始化

    async def prep_async(self, shared):
        """准备阶段：获取问题、回答和融合结果以及registry"""
        question = shared.get("user_question", "")
        llm_responses = shared.get("llm_responses", [])
        final_answer = shared.get("final_answer", "")
        registry = shared.get("registry")
        trace_id = shared.get("trace_id")
        trace_observation_id = shared.get("langfuse_trace_observation_id")

        if not question:
            raise ValueError("用户问题不能为空")

        if not llm_responses:
            print("⚠️ 没有LLM回答，跳过质量分析")
            return None

        # 初始化 analyzer（使用 registry）
        if self.analyzer is None:
            self.analyzer = AIFusionQualityAnalyzer(registry=registry)

        return {
            "question": question,
            "llm_responses": llm_responses,
            "final_answer": final_answer,
            "trace_id": trace_id,
            "trace_observation_id": trace_observation_id
        }

    async def exec_async(self, inputs):
        """执行阶段：进行质量分析"""
        if inputs is None:
            return None

        trace_id = inputs.get("trace_id")
        trace_observation_id = inputs.get("trace_observation_id")

        analysis_span = create_span(
            trace_id,
            name="QualityAnalyzer",
            parent_observation_id=trace_observation_id,
            input_data={
                "question": inputs["question"],
                "final_answer": inputs["final_answer"][:200] if inputs["final_answer"] else "",
                "response_count": len(inputs["llm_responses"]),
            },
            metadata={"node": "QualityAnalyzer"},
        )
        parent_observation_id = analysis_span.id if analysis_span else trace_observation_id

        print("\n🔍 正在进行质量分析...")

        try:
            quality_analysis = await self.analyzer.analyze_quality(
                question=inputs["question"],
                llm_responses=inputs["llm_responses"],
                fusion_answer=inputs["final_answer"],
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
            )

            finish_observation(
                analysis_span,
                output_data={"quality_analysis": quality_analysis},
                metadata={"node": "QualityAnalyzer"},
            )

            return quality_analysis
        except Exception as exc:
            finish_observation(
                analysis_span,
                output_data={"error": str(exc)},
                metadata={"node": "QualityAnalyzer"},
                level="ERROR",
                status_message=str(exc),
            )
            raise

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
