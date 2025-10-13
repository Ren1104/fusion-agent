"""
AI Fusion 报告生成器
生成详细的markdown分析报告，包含模型性能对比和回答质量分析
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class AIFusionReporter:
    """AI Fusion报告生成器"""

    def __init__(self):
        # 从环境变量获取报告目录，如果未设置则使用默认值"reports"
        self.report_dir = os.getenv("AI_FUSION_REPORT_DIR", "reports")
        self._ensure_report_dir()

    def _ensure_report_dir(self):
        """确保报告目录存在"""
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def generate_report(
        self,
        question: str,
        question_type: str,
        llm_responses: List[Dict],
        final_answer: str,
        selected_models: List[str],
        quality_analysis: Optional[Dict[str, Any]] = None,
        selection_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成完整的分析报告

        Args:
            question: 用户问题
            question_type: 问题类型
            llm_responses: 各模型的响应数据
            final_answer: 融合后的最终回答
            selected_models: 选择的模型列表

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_fusion_report_{timestamp}.md"
        filepath = os.path.join(self.report_dir, filename)

        # 生成报告内容
        report_content = self._build_report_content(
            question, question_type, llm_responses, final_answer, selected_models, quality_analysis, selection_analysis
        )

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📊 分析报告已生成: {filepath}")
        return filepath

    def _build_report_content(
        self,
        question: str,
        question_type: str,
        llm_responses: List[Dict],
        final_answer: str,
        selected_models: List[str],
        quality_analysis: Optional[Dict[str, Any]] = None,
        selection_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建报告内容 - 优化版，专注于融合回答和最佳模型表现"""

        # 统计信息
        total_models = len(llm_responses)
        fusion_length = len(final_answer)

        report = f"""# AI Fusion 分析报告

## 📋 基本信息
- **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **问题类型**: {question_type}
- **参与模型数量**: {total_models}

## ❓ 用户问题
```
{question}
```

## 🎯 AI Fusion 融合回答
**字符数**: {fusion_length}

```
{final_answer}
```

## 🏆 最佳模型表现

{self._generate_best_models_section(llm_responses, quality_analysis)}

---
*本报告由 AI Fusion 系统自动生成*
"""

        return report

    def _generate_best_models_section(self, llm_responses: List[Dict], quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成最佳模型表现部分 - 显示最快和最高质量的模型，包含详细的评分计算说明"""
        section = ""

        # 找出响应时间最快的模型
        successful_responses = [r for r in llm_responses if r.get('success')]
        if successful_responses:
            fastest_model = min(successful_responses, key=lambda x: x.get('response_time', float('inf')))
            section += f"### 🚀 响应速度最快\n\n"
            section += f"**{fastest_model['model_name']}**\n"
            section += f"- 响应时间: {fastest_model.get('response_time', 0):.2f}秒\n"
            section += f"- 回答长度: {len(fastest_model['response'])}字符\n\n"

        # 找出质量最高的模型并详细说明评分计算
        if quality_analysis and 'quality_ranking' in quality_analysis:
            ranking = quality_analysis['quality_ranking']
            # 找到质量最高的非融合回答
            best_model_entry = next((item for item in ranking if not item['is_fusion']), None)

            if best_model_entry:
                section += f"### 🏆 质量评分最高\n\n"
                section += f"**{best_model_entry['source']}**\n\n"

                # 综合评分
                section += f"#### 📊 综合评分: {best_model_entry['overall_score']:.1f}/10\n\n"

                # 详细评分分解
                section += f"#### 🔍 评分细节\n\n"

                # 获取LLM评估详情
                llm_evals = quality_analysis.get('llm_evaluations', {})
                model_eval = llm_evals.get(best_model_entry['source'])

                if model_eval:
                    # 完整性评分
                    completeness = best_model_entry['completeness']
                    section += f"**1. 完整性评分: {completeness:.1f}/10**\n"
                    section += f"- 评估标准: 回答是否覆盖问题的所有关键方面\n"
                    if hasattr(model_eval, 'completeness_reasoning'):
                        section += f"- 评分理由: {model_eval.completeness_reasoning}\n"
                    section += "\n"

                    # 准确性评分
                    accuracy = best_model_entry['accuracy']
                    section += f"**2. 准确性评分: {accuracy:.1f}/10**\n"
                    section += f"- 评估标准: 信息是否准确无误，逻辑是否严密\n"
                    if hasattr(model_eval, 'accuracy_reasoning'):
                        section += f"- 评分理由: {model_eval.accuracy_reasoning}\n"
                    section += "\n"

                    # 清晰度评分
                    clarity = best_model_entry['clarity']
                    section += f"**3. 清晰度评分: {clarity:.1f}/10**\n"
                    section += f"- 评估标准: 表达是否清晰易懂，结构是否合理\n"
                    if hasattr(model_eval, 'clarity_reasoning'):
                        section += f"- 评分理由: {model_eval.clarity_reasoning}\n"
                    section += "\n"

                    # 相关性评分
                    relevance = best_model_entry['relevance']
                    section += f"**4. 相关性评分: {relevance:.1f}/10**\n"
                    section += f"- 评估标准: 内容是否切题，是否直接回答了问题\n"
                    if hasattr(model_eval, 'relevance_reasoning'):
                        section += f"- 评分理由: {model_eval.relevance_reasoning}\n"
                    section += "\n"

                    # 综合评分计算说明
                    section += f"**综合评分计算方式:**\n"
                    section += f"```\n"
                    section += f"综合评分 = (完整性 + 准确性 + 清晰度 + 相关性) / 4\n"
                    section += f"         = ({completeness:.1f} + {accuracy:.1f} + {clarity:.1f} + {relevance:.1f}) / 4\n"
                    section += f"         = {best_model_entry['overall_score']:.1f}/10\n"
                    section += f"```\n\n"

                    # 总体评价
                    if hasattr(model_eval, 'overall_assessment'):
                        section += f"**💭 总体评价:**\n{model_eval.overall_assessment}\n\n"
                else:
                    # 如果没有详细评估数据，显示基本信息
                    section += f"- 完整性: {best_model_entry['completeness']:.1f}/10\n"
                    section += f"- 准确性: {best_model_entry['accuracy']:.1f}/10\n"
                    section += f"- 清晰度: {best_model_entry['clarity']:.1f}/10\n"
                    section += f"- 相关性: {best_model_entry['relevance']:.1f}/10\n\n"
                    section += f"**综合评分** = (完整性 + 准确性 + 清晰度 + 相关性) / 4 = {best_model_entry['overall_score']:.1f}/10\n\n"
        else:
            section += f"### 🏆 质量评分最高\n\n"
            section += f"_质量分析数据不可用_\n\n"

        return section

    def print_summary(self, llm_responses: List[Dict], final_answer: str, quality_analysis: Optional[Dict[str, Any]] = None):
        """打印简要摘要"""
        print("\n" + "="*60)
        print("📊 AI Fusion 分析摘要")
        print("="*60)

        for i, response in enumerate(llm_responses, 1):
            if response['success']:
                print(f"🤖 模型 {i}: {response['model_name']}")
                print(f"   ⏱️  响应时间: {response.get('response_time', 0):.2f}秒")
                print(f"   📝 回答长度: {len(response['response'])}字符")
                print(f"   ✅ 状态: 成功")
            else:
                print(f"❌ 模型 {i}: {response['model_name']} - 失败")
                print(f"   ⏱️  响应时间: {response.get('response_time', 0):.2f}秒")
                print(f"   📝 错误: {response.get('error', 'Unknown')}")
            print()

        print(f"🎯 融合回答长度: {len(final_answer)}字符")
        print("="*60)
