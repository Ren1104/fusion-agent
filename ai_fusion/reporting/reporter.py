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
        self.report_dir = "reports"
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
        生成完整的markdown分析报告
        
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

    def _generate_quality_analysis_section(self, quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成质量分析部分"""
        if not quality_analysis:
            return "质量分析功能未启用。"
        
        section = ""
        
        # 质量排名
        if 'quality_ranking' in quality_analysis:
            section += "### 📊 质量评分排名\n\n"
            section += "| 排名 | 来源 | 综合评分 | 完整性 | 准确性 | 清晰度 | 相关性 | 字数 |\n"
            section += "|------|------|----------|--------|--------|--------|--------|------|\n"
            
            for item in quality_analysis['quality_ranking']:
                source_display = "🎯 融合回答" if item['is_fusion'] else f"🤖 {item['source']}"
                section += f"| {item['rank']} | {source_display} | {item['overall_score']:.1f} | {item['completeness']:.1f} | {item['accuracy']:.1f} | {item['clarity']:.1f} | {item['relevance']:.1f} | {item['word_count']} |\n"
            
            section += "\n"
        
        # 对比分析
        if 'comparison_analysis' in quality_analysis:
            comp_analysis = quality_analysis['comparison_analysis']
            
            # 融合优势
            if comp_analysis.get('fusion_advantages'):
                section += "### 🚀 融合回答优势\n\n"
                for advantage in comp_analysis['fusion_advantages']:
                    section += f"- ✅ {advantage}\n"
                section += "\n"
            
            # 各模型强项 - 个性化分析
            if comp_analysis.get('model_strengths'):
                section += "### 💪 各模型在本次任务中的优势表现\n\n"
                for model, strengths in comp_analysis['model_strengths'].items():
                    section += f"**{model}**: {', '.join(strengths)}\n"
                section += "\n"
            
            # **新增：深度个性化档案**
            content_analysis = quality_analysis.get('content_analysis', {})
            individualized_profiles = content_analysis.get('individualized_profiles', {})
            if individualized_profiles and isinstance(individualized_profiles, dict):
                profiles_data = individualized_profiles.get('individualized_profiles', {})
                if profiles_data:
                    section += "### 🎯 深度个性化档案\n\n"
                    section += "> 基于内容深度分析，为每个模型生成的独特特征档案\n\n"

                    for model_name, profile in profiles_data.items():
                        section += f"#### {model_name}\n\n"

                        # 标志性特征
                        signature = profile.get('signature_characteristics', '')
                        if signature:
                            section += f"**🏷️ 标志性特征**: {signature}\n\n"

                        # 内容风格
                        style = profile.get('content_style', '')
                        if style:
                            section += f"**📝 内容风格**: {style}\n\n"

                        # 解答角度与深度
                        approach = profile.get('approach_depth', '')
                        if approach:
                            section += f"**🎯 解答角度**: {approach}\n\n"

                        # 独特贡献点
                        contributions = profile.get('unique_contributions', [])
                        if contributions:
                            section += f"**💡 独特贡献**:\n"
                            for contrib in contributions:
                                section += f"- {contrib}\n"
                            section += "\n"

                        # 比较优势与不足
                        advantage = profile.get('comparative_advantage', '')
                        weakness = profile.get('comparative_weakness', '')
                        if advantage or weakness:
                            section += f"**⚖️ 对比分析**:\n"
                            if advantage:
                                section += f"- ✅ **优势**: {advantage}\n"
                            if weakness:
                                section += f"- ⚠️ **不足**: {weakness}\n"
                            section += "\n"

                        # 适用场景
                        scenarios = profile.get('best_use_scenarios', [])
                        if scenarios:
                            section += f"**🎬 最佳适用场景**:\n"
                            for scenario in scenarios:
                                section += f"- {scenario}\n"
                            section += "\n"

                        section += "---\n\n"

                    # 差异化总结
                    diff_summary = individualized_profiles.get('differentiation_summary', '')
                    if diff_summary:
                        section += f"**🔍 差异化总结**: {diff_summary}\n\n"

            # # 详细个体分析（作为补充）
            # if comp_analysis.get('individual_analysis'):
            #     section += "### 📊 各模型详细表现分析\n\n"
            #     for model, analysis in comp_analysis['individual_analysis'].items():
            #         section += f"#### {model}\n\n"
            #
            #         highlights = analysis.get('performance_highlights', [])
            #         if highlights:
            #             section += f"- **性能亮点**: {', '.join(highlights)}\n"
            #
            #         ranking = analysis.get('relative_ranking', {})
            #         if ranking:
            #             section += f"- **本次排名**: 完整性{ranking.get('completeness', 'N/A')}, 准确性{ranking.get('accuracy', 'N/A')}, 清晰度{ranking.get('clarity', 'N/A')}, 相关性{ranking.get('relevance', 'N/A')}\n"
            #
            #         style = analysis.get('style_characteristics', [])
            #         if style:
            #             section += f"- **回答风格**: {', '.join(style)}\n"
            #
            #         potential = analysis.get('improvement_potential', [])
            #         if potential:
            #             section += f"- **分析建议**: {', '.join(potential)}\n"
            #
            #         section += "\n"
            
            # 统计摘要
            if comp_analysis.get('statistical_summary'):
                stats = comp_analysis['statistical_summary']
                section += "### 📈 质量统计摘要\n\n"
                section += f"- **融合回答评分**: {stats.get('fusion_overall_score', 0):.1f}/10\n"
                section += f"- **模型平均评分**: {stats.get('models_avg_score', 0):.1f}/10\n"
                section += f"- **质量提升**: {stats.get('improvement', 0):+.1f}分\n"
                section += f"- **最佳单模型评分**: {stats.get('best_individual_score', 0):.1f}/10\n"
                section += f"- **vs最佳单模型**: {stats.get('fusion_vs_best', 0):+.1f}分\n\n"
        
        return section
    
    def _generate_quality_summary(self, quality_analysis: Optional[Dict[str, Any]]) -> str:
        """生成质量摘要"""
        if not quality_analysis:
            return ""
        
        summary = "\n### 🎯 质量分析总结\n\n"
        
        if 'comparison_analysis' in quality_analysis:
            stats = quality_analysis['comparison_analysis'].get('statistical_summary', {})
            improvement = stats.get('improvement', 0)
            
            if improvement > 0:
                summary += f"- **质量提升**: 融合回答比单模型平均质量提升了 {improvement:.1f} 分\n"
            elif improvement < 0:
                summary += f"- **质量对比**: 融合回答比单模型平均质量低 {abs(improvement):.1f} 分\n"
            else:
                summary += f"- **质量对比**: 融合回答与单模型平均质量持平\n"
            
            fusion_vs_best = stats.get('fusion_vs_best', 0)
            if fusion_vs_best > 0:
                summary += f"- **最佳对比**: 融合回答超越了最佳单模型 {fusion_vs_best:.1f} 分\n"
            elif fusion_vs_best < 0:
                summary += f"- **最佳对比**: 融合回答比最佳单模型低 {abs(fusion_vs_best):.1f} 分\n"
            else:
                summary += f"- **最佳对比**: 融合回答与最佳单模型质量相当\n"
        
        return summary
    
    def _generate_intelligent_recommendations(
        self, 
        question_type: str,
        llm_responses: List[Dict],
        final_answer: str,
        quality_analysis: Optional[Dict[str, Any]] = None,
        selection_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成智能建议和总结"""
        
        recommendations = []
        
        # 性能分析建议
        if llm_responses:
            # 找出最快和最慢的模型
            response_times = [(r['model_name'], r.get('response_time', 0)) for r in llm_responses if r.get('success')]
            if response_times:
                fastest = min(response_times, key=lambda x: x[1])
                slowest = max(response_times, key=lambda x: x[1])
                
                recommendations.append(f"### ⚡ 性能表现")
                recommendations.append(f"- **响应速度冠军**: {fastest[0]} ({fastest[1]:.2f}秒)")
                
                if slowest[1] - fastest[1] > 2:
                    recommendations.append(f"- **优化建议**: {slowest[0]} 响应较慢({slowest[1]:.2f}秒)，在时间敏感场景下可优先选择更快的模型")
                
                avg_time = sum(t for _, t in response_times) / len(response_times)
                recommendations.append(f"- **平均响应时间**: {avg_time:.2f}秒")
        
        # 模型选择建议
        if selection_analysis:
            analysis_method = selection_analysis.get('analysis_method', '')
            confidence = selection_analysis.get('confidence_level', '')
            
            recommendations.append(f"\n### 🎯 模型选择效果")
            
            if analysis_method == 'intelligent_llm':
                recommendations.append(f"- **选择方式**: 🧠 智能LLM分析，置信度: {confidence}")
                recommendations.append(f"- **问题适配**: 当前模型组合经过AI深度分析，针对\"{question_type}\"类型问题进行了优化选择")
            elif analysis_method == 'fallback':
                recommendations.append(f"- **选择方式**: 🔄 传统规则匹配，建议检查智能选择器状态")
                recommendations.append(f"- **改进建议**: 智能选择功能异常，建议排查网络或API状态")
            
            problem_analysis = selection_analysis.get('problem_analysis', {})
            if problem_analysis:
                complexity = problem_analysis.get('complexity_level', '')
                if complexity == '复杂':
                    recommendations.append(f"- **复杂度评估**: 高复杂度问题，多模型融合策略得当")
                elif complexity == '简单':
                    recommendations.append(f"- **效率优化**: 简单问题可考虑使用更快的轻量级模型组合")
        
        # 质量分析建议
        if quality_analysis:
            ranking = quality_analysis.get('quality_ranking', [])
            if ranking:
                best_model = ranking[0] if ranking else None
                fusion_rank = next((i+1 for i, r in enumerate(ranking) if r['is_fusion']), None)
                
                recommendations.append(f"\n### 📊 质量表现洞察")
                
                if fusion_rank == 1:
                    recommendations.append(f"- **融合效果**: 🏆 融合回答获得最高质量评分，多模型协作效果显著")
                elif fusion_rank == 2:
                    recommendations.append(f"- **融合效果**: 🥈 融合回答质量优秀，略逊于最佳单模型但整体表现稳定")
                elif fusion_rank and fusion_rank <= 3:
                    recommendations.append(f"- **融合效果**: 🥉 融合回答进入前三，在某些维度上有所提升")
                else:
                    recommendations.append(f"- **融合效果**: ⚠️ 融合回答未达到预期，建议检查模型选择策略")
                
                if best_model and not best_model['is_fusion']:
                    recommendations.append(f"- **单模型优秀**: {best_model['source']} 在本次任务中表现最佳(评分: {best_model['overall_score']:.1f})")
            
            # 分析各模型优势
            comp_analysis = quality_analysis.get('comparison_analysis', {})
            model_strengths = comp_analysis.get('model_strengths', {})
            
            if model_strengths:
                recommendations.append(f"\n### 🔍 模型特长发现")
                for model, strengths in model_strengths.items():
                    if strengths:
                        primary_strength = strengths[0]  # 取主要优势
                        recommendations.append(f"- **{model}**: {primary_strength}，建议在相关场景中优先考虑")
        
        # 基于问题类型的专业建议
        recommendations.append(f"\n### 💭 针对\"{question_type}\"问题的专业建议")
        
        type_recommendations = {
            '技术/编程': [
                "对于编程问题，建议优先选择逻辑推理能力强的模型",
                "可考虑让不同模型分别处理算法设计、代码实现和优化建议",
                "融合时注意代码的可执行性和最佳实践"
            ],
            '创意写作': [
                "创意类问题适合发挥各模型的想象力差异",
                "建议融合时保持创意的多样性和原创性",
                "可让不同模型提供不同的创作角度和风格"
            ],
            '数学/逻辑': [
                "数学问题需要确保计算的准确性",
                "建议让多个模型验证计算过程和结果",
                "融合时应优先考虑逻辑严谨性"
            ],
            '日常对话': [
                "日常对话注重自然性和实用性",
                "可适当简化模型配置以提高响应速度",
                "融合时保持回答的亲和力和易懂性"
            ],
            '专业知识': [
                "专业问题需要确保知识的权威性和准确性",
                "建议选择知识丰富且在该领域有优势的模型",
                "融合时注意专业术语的准确使用"
            ],
            '翻译': [
                "翻译任务需要平衡准确性和流畅性",
                "建议选择多语言能力强的模型组合",
                "融合时注意保持原文意境和目标语言习惯"
            ],
            '分析总结': [
                "分析类问题需要逻辑清晰和结构完整",
                "建议让不同模型从不同角度进行分析",
                "融合时确保结论的逻辑性和说服力"
            ]
        }
        
        specific_recs = type_recommendations.get(question_type, [
            "根据问题特点选择合适的模型组合",
            "注意发挥各模型的互补优势",
            "融合时确保回答的准确性和完整性"
        ])
        
        for rec in specific_recs:
            recommendations.append(f"- {rec}")
        
        # 未来优化建议
        recommendations.append(f"\n### 🚀 未来优化方向")
        
        if quality_analysis and selection_analysis:
            # 基于本次表现给出具体建议
            stats = quality_analysis.get('comparison_analysis', {}).get('statistical_summary', {})
            improvement = stats.get('improvement', 0)
            
            if improvement > 1:
                recommendations.append(f"- **继续优化**: 当前融合效果优秀，可尝试更多样化的问题类型")
            elif improvement > 0:
                recommendations.append(f"- **微调优化**: 融合有效果但提升有限，可考虑调整模型权重或选择策略")
            else:
                recommendations.append(f"- **策略调整**: 融合效果不理想，建议重新评估模型选择标准")
            
            recommendations.append(f"- **数据积累**: 持续收集各类问题的模型表现数据，优化选择算法")
            recommendations.append(f"- **个性定制**: 根据用户偏好和使用场景定制专属的模型组合方案")
        
        return "\n".join(recommendations)
    
    def _generate_selection_analysis_section(self, selection_analysis: Optional[Dict[str, Any]]) -> str:
        """生成智能选择分析部分"""
        if not selection_analysis:
            return "智能选择分析功能未启用。"
        
        section = ""
        
        # 问题分析
        problem_analysis = selection_analysis.get('problem_analysis', {})
        if problem_analysis:
            section += "### 📋 问题深度分析\n\n"
            section += f"- **问题类型**: {problem_analysis.get('question_type', '未知')}\n"
            section += f"- **复杂度等级**: {problem_analysis.get('complexity_level', '未知')}\n"
            
            capabilities = problem_analysis.get('required_capabilities', [])
            if capabilities:
                section += f"- **所需能力**: {', '.join(capabilities)}\n"
            
            challenges = problem_analysis.get('key_challenges', [])
            if challenges:
                section += f"- **主要挑战**: {', '.join(challenges)}\n"
            
            section += "\n"
        
        # 模型推荐详情
        recommended_models = selection_analysis.get('recommended_models', [])
        if recommended_models:
            section += "### 🎯 智能推荐详情\n\n"
            section += "| 排名 | 模型名称 | 适合度评分 | 选择理由 | 预期贡献 |\n"
            section += "|------|----------|------------|----------|----------|\n"
            
            for model in recommended_models:
                rank = model.get('rank', 0)
                name = model.get('model_name', '')
                score = model.get('suitability_score', 0)
                reasons = '; '.join(model.get('reasons', []))
                contribution = model.get('expected_contribution', '')
                
                section += f"| {rank} | {name} | {score}/10 | {reasons} | {contribution} |\n"
            
            section += "\n"
        
        # 组合策略
        strategy = selection_analysis.get('combination_strategy', '')
        confidence = selection_analysis.get('confidence_level', '')
        analysis_method = selection_analysis.get('analysis_method', '')
        
        if strategy or confidence or analysis_method:
            section += "### 🔗 选择策略与置信度\n\n"
            
            if strategy:
                section += f"- **组合策略**: {strategy}\n"
            if confidence:
                section += f"- **置信度**: {confidence}\n"
            if analysis_method:
                method_display = {
                    'intelligent_llm': '🧠 LLM智能分析',
                    'fallback': '🔄 传统规则匹配',
                    'unknown': '❓ 未知方法'
                }.get(analysis_method, analysis_method)
                section += f"- **分析方法**: {method_display}\n"
            
            section += "\n"
        
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