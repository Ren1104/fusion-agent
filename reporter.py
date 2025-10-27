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

{self._generate_quality_overview_section(quality_analysis)}

{self._generate_speed_quality_section(quality_analysis)}

---
*本报告由 AI Fusion 系统自动生成*
"""

        return report

    def _generate_best_models_section(self, llm_responses: List[Dict], quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成最佳模型表现部分 - 显示最快和最高质量的模型，包含详细的评分计算说明"""
        section = ""

        successful_responses = [r for r in llm_responses if r.get('success')]
        if successful_responses:
            fastest_model = min(successful_responses, key=lambda x: x.get('response_time', float('inf')))
            section += f"### 🚀 响应速度最快\n\n"
            section += f"**{fastest_model['model_name']}**\n"
            section += f"- 响应时间: {fastest_model.get('response_time', 0):.2f}秒\n"
            section += f"- 回答长度: {len(fastest_model['response'])}字符\n\n"

        if quality_analysis and 'quality_ranking' in quality_analysis:
            ranking = quality_analysis['quality_ranking']
            details_map = quality_analysis.get('llm_evaluation_details', {})
            best_model_entry = next((item for item in ranking if not item['is_fusion']), None)

            if best_model_entry:
                section += f"### 🏆 质量评分最高\n\n"
                section += f"**{best_model_entry['source']}**\n\n"
                section += f"#### 📊 综合评分: {best_model_entry['overall_score']:.1f}/10\n\n"
                section += "#### 🔍 评分细节\n\n"
                section += self._render_dimension_scores(best_model_entry)

                model_details = details_map.get(best_model_entry['source'], {})
                if model_details:
                    section += f"**💡 特征摘要:** {model_details.get('unique_characteristics', '_暂无描述_')}\n\n"
                    strengths = []
                    weaknesses = []
                    for dimension in ('completeness', 'accuracy', 'clarity', 'relevance'):
                        strengths.extend(model_details.get(dimension, {}).get('strengths', []))
                        weaknesses.extend(model_details.get(dimension, {}).get('weaknesses', []))
                    section += f"**✅ 关键优势:** {self._format_list(strengths)}\n\n"
                    section += f"**❌ 改进空间:** {self._format_list(weaknesses)}\n\n"
                    suggestions = model_details.get('core_suggestions', [])
                    if suggestions:
                        section += "**🛠 改进建议:**\n"
                        for suggestion in suggestions[:3]:
                            section += f"- {suggestion}\n"
                        section += "\n"
        else:
            section += f"### 🏆 质量评分最高\n\n"
            section += f"_质量分析数据不可用_\n\n"

        return section

    def _format_list(self, items: Optional[List[str]], fallback: str = "_暂无数据_") -> str:
        """格式化列表内容为可读文本"""
        if not items:
            return fallback
        cleaned = [item.strip() for item in items if item and item.strip()]
        if not cleaned:
            return fallback
        return "；".join(cleaned[:4])

    def _render_dimension_scores(self, ranking_entry: Dict[str, Any]) -> str:
        """渲染维度得分"""
        completeness = ranking_entry['completeness']
        accuracy = ranking_entry['accuracy']
        clarity = ranking_entry['clarity']
        relevance = ranking_entry['relevance']

        return (
            f"- 完整性: {completeness:.1f}/10\n"
            f"- 准确性: {accuracy:.1f}/10\n"
            f"- 清晰度: {clarity:.1f}/10\n"
            f"- 相关性: {relevance:.1f}/10\n\n"
        )

    def _generate_quality_overview_section(self, quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成整体质量分析概览"""
        if not quality_analysis:
            return ""

        ranking = quality_analysis.get('quality_ranking')
        llm_evaluations = quality_analysis.get('llm_evaluations', {})
        evaluation_details = quality_analysis.get('llm_evaluation_details', {})

        if not ranking or not llm_evaluations:
            return ""

        overview = "## 🔍 质量分析概览\n\n"
        overview += self._build_quality_table(ranking, llm_evaluations)
        overview += "\n"
        overview += self._build_model_insights(ranking, evaluation_details)

        fusion_effectiveness = quality_analysis.get('fusion_effectiveness')
        if fusion_effectiveness:
            overview += "\n" + self._build_fusion_effectiveness(fusion_effectiveness)

        return overview

    def _build_quality_table(self, ranking: List[Dict[str, Any]], llm_evaluations: Dict[str, Any]) -> str:
        """构建质量评分表格"""
        header = "| 模型 | 综合 | 完整性 | 准确性 | 清晰度 | 相关性 | 字符数 |\n"
        header += "| --- | --- | --- | --- | --- | --- | --- |\n"

        rows = []
        for entry in ranking:
            metrics = llm_evaluations.get(entry['source'])
            char_count = getattr(metrics, "char_count", entry.get('word_count', 0))
            rows.append(
                f"| {entry['source']} | {entry['overall_score']:.1f} | "
                f"{entry['completeness']:.1f} | {entry['accuracy']:.1f} | "
                f"{entry['clarity']:.1f} | {entry['relevance']:.1f} | {char_count} |"
            )

        return header + "\n".join(rows) + "\n"

    def _build_model_insights(
        self,
        ranking: List[Dict[str, Any]],
        evaluation_details: Dict[str, Dict[str, Any]]
    ) -> str:
        """汇总模型的优势与改进方向"""
        if not evaluation_details:
            return ""

        insight_text = "### 📈 模型表现洞察\n\n"
        for entry in ranking:
            source = entry['source']
            details = evaluation_details.get(source)
            if not details:
                continue

            strengths = []
            weaknesses = []
            for dimension in ('completeness', 'accuracy', 'clarity', 'relevance'):
                strengths.extend(details.get(dimension, {}).get('strengths', []))
                weaknesses.extend(details.get(dimension, {}).get('weaknesses', []))

            unique = details.get('unique_characteristics', "")
            suggestions = details.get('core_suggestions', [])

            insight_text += f"#### {source}\n\n"
            insight_text += f"- **特征摘要**: {unique or '_暂无描述_'}\n"
            insight_text += f"- **主要优势**: {self._format_list(strengths)}\n"
            insight_text += f"- **主要不足**: {self._format_list(weaknesses)}\n"
            if suggestions:
                insight_text += f"- **改进建议**: {self._format_list(suggestions)}\n"
            insight_text += "\n"

        return insight_text

    def _build_fusion_effectiveness(self, fusion_effectiveness: Dict[str, Any]) -> str:
        """渲染融合效果分析"""
        summary = fusion_effectiveness.get('summary')
        dimension_improvements = fusion_effectiveness.get('dimension_improvements', {})
        value = fusion_effectiveness.get('fusion_value_score')

        section = "### 🤝 融合效果\n\n"
        if summary:
            section += f"- **综合结论**: {summary}\n"

        if dimension_improvements:
            improvements = []
            for dim, stats in dimension_improvements.items():
                delta = stats.get('average_improvement')
                if delta is not None:
                    improvements.append(f"{dim}: {delta:+.2f}")
            if improvements:
                section += f"- **维度提升**: {'；'.join(improvements)}\n"

        if value is not None:
            section += f"- **融合价值评分**: {value:.1f}/100\n"

        return section + "\n"

    def _generate_speed_quality_section(self, quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成速度-质量权衡分析部分"""
        if not quality_analysis:
            return ""

        tradeoff = quality_analysis.get('speed_quality_tradeoff')
        if not tradeoff or not tradeoff.get('available'):
            return ""

        section = "## ⏱️ 速度-质量权衡\n\n"

        fastest = tradeoff.get('fastest_model')
        quality = tradeoff.get('highest_quality_model')
        efficient = tradeoff.get('most_efficient_model')

        if fastest:
            section += (
                f"- **最快响应模型**: {fastest['name']} "
                f"(耗时 {fastest['response_time']:.2f}s，质量 {fastest['quality_score']:.1f})\n"
            )
        if quality:
            section += (
                f"- **质量最佳模型**: {quality['name']} "
                f"(耗时 {quality['response_time']:.2f}s，质量 {quality['quality_score']:.1f})\n"
            )
        if efficient and efficient.get('efficiency_score') is not None:
            section += (
                f"- **性价比最佳**: {efficient['name']} "
                f"(效率 {efficient['efficiency_score']:.2f}，质量 {efficient['quality_score']:.1f})\n"
            )

        correlation = tradeoff.get('correlation_analysis')
        if correlation:
            section += f"- **速度与质量关系**: {correlation.get('description', '暂无结论')}\n"

        recommendations = tradeoff.get('scenario_recommendations', [])
        if recommendations:
            section += "\n**场景化建议：**\n"
            if isinstance(recommendations, dict):
                items = list(recommendations.items())
                for key, value in items[:3]:
                    label = {
                        'time_critical': "时间敏感",
                        'quality_critical': "质量优先",
                        'balanced': "综合平衡",
                        'production': "生产环境"
                    }.get(key, key)
                    section += f"- {label}: {value}\n"
            else:
                for rec in recommendations[:3]:
                    section += f"- {rec}\n"

        return section + "\n"

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
