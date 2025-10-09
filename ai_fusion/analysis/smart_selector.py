"""
AI Fusion 智能模型选择器
使用LLM分析问题并推荐最适合的模型组合
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ai_fusion.utils.helpers import call_llm_async, ModelConfig


@dataclass
class ModelCapability:
    """模型能力描述"""
    name: str
    provider: str
    strengths: List[str]
    suitable_tasks: List[str]
    performance_profile: Dict[str, str]
    special_features: List[str]


class AIFusionSmartSelector:
    """AI Fusion智能模型选择器"""
    
    def __init__(self):
        self.analyzer_model = "claude_sonnet4"  # 用于分析的模型
        self.model_knowledge = self._build_model_knowledge()
    
    def _build_model_knowledge(self) -> Dict[str, ModelCapability]:
        """构建模型知识库 - 包含所有可用模型的详细能力描述"""
        return {
            # ============= Claude 系列 =============
            "claude_sonnet4": ModelCapability(
                name="claude_sonnet4",
                provider="anthropic",
                strengths=[
                    "逻辑推理能力卓越", "代码理解和生成顶尖", "复杂问题深度分析",
                    "创意写作优秀", "多步骤任务处理", "结构化输出精准",
                    "长文本理解", "细致的上下文把握"
                ],
                suitable_tasks=[
                    "编程和技术问题", "逻辑推理", "创意写作", "复杂分析",
                    "学术研究", "产品设计", "策略规划", "代码审查",
                    "系统架构设计", "技术文档编写"
                ],
                performance_profile={
                    "reasoning": "excellent",      # 推理能力：卓越
                    "creativity": "excellent",     # 创造力：卓越
                    "coding": "excellent",         # 编程能力：卓越
                    "factual": "good",             # 事实准确性：良好
                    "speed": "medium",             # 响应速度：中等
                    "context": "excellent"         # 上下文理解：卓越
                },
                special_features=["支持200K+长文本", "强逻辑推理链", "创意性强", "道德安全意识高"]
            ),

            "claude37_sonnet_new": ModelCapability(
                name="claude37_sonnet_new",
                provider="anthropic",
                strengths=[
                    "平衡的综合能力", "快速响应", "日常对话流畅",
                    "信息整理清晰", "中等复杂度任务", "稳定性好"
                ],
                suitable_tasks=[
                    "日常问答", "信息总结", "翻译", "简单分析",
                    "文档整理", "基础编程", "通用任务", "客服对话",
                    "内容润色", "快速原型开发"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "good",
                    "coding": "good",
                    "factual": "good",
                    "speed": "fast",
                    "context": "good"
                },
                special_features=["响应快速", "稳定可靠", "平衡发展", "成本效益好"]
            ),

            # ============= GPT 系列 =============
            "gpt-41-0414-global": ModelCapability(
                name="gpt-41-0414-global",
                provider="openai",
                strengths=[
                    "数学和科学计算强", "逻辑推理严谨", "结构化分析精确",
                    "代码优化能力", "技术文档处理", "多语言支持",
                    "工具调用能力", "函数调用精准"
                ],
                suitable_tasks=[
                    "数学问题", "科学计算", "算法设计", "数据分析",
                    "技术文档", "系统设计", "工程问题", "API开发",
                    "数据可视化", "统计分析"
                ],
                performance_profile={
                    "reasoning": "excellent",
                    "creativity": "good",
                    "coding": "excellent",
                    "factual": "excellent",
                    "speed": "medium",
                    "context": "excellent"
                },
                special_features=["数学能力强", "逻辑严谨", "技术导向", "128K上下文", "工具使用"]
            ),

            "gpt-41-mini-0414-global": ModelCapability(
                name="gpt-41-mini-0414-global",
                provider="openai",
                strengths=[
                    "极速响应", "轻量级任务", "日常对话",
                    "简单问答高效", "成本优化", "批量处理适合"
                ],
                suitable_tasks=[
                    "快速问答", "简单翻译", "基础对话", "信息查找",
                    "轻量级编程", "格式转换", "效率任务", "批量处理",
                    "实时对话", "快速原型"
                ],
                performance_profile={
                    "reasoning": "medium",
                    "creativity": "medium",
                    "coding": "good",
                    "factual": "good",
                    "speed": "very_fast",
                    "context": "good"
                },
                special_features=["极速响应", "资源节约", "简洁高效", "性价比极高"]
            ),

            "gpt-5-mini-0807-global": ModelCapability(
                name="gpt-5-mini-0807-global",
                provider="openai",
                strengths=[
                    "新一代轻量模型", "速度与质量平衡", "改进的推理能力",
                    "更好的指令遵循", "多任务处理", "成本效益优"
                ],
                suitable_tasks=[
                    "通用问答", "文本生成", "简单编程", "信息提取",
                    "对话系统", "内容审核", "分类任务", "摘要生成",
                    "情感分析", "实体识别"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "good",
                    "coding": "good",
                    "factual": "good",
                    "speed": "very_fast",
                    "context": "good"
                },
                special_features=["GPT-5系列", "速度快", "质量提升", "指令遵循好"]
            ),

            # ============= 通义千问系列 =============
            "qwen-max": ModelCapability(
                name="qwen-max",
                provider="alibaba",
                strengths=[
                    "中文理解顶尖", "知识覆盖广", "多语言支持强",
                    "文化语境理解深", "本土化内容精准", "推理能力强",
                    "长文本处理", "专业领域知识"
                ],
                suitable_tasks=[
                    "中文内容处理", "翻译任务", "文化相关问题",
                    "本土化内容", "多语言对话", "知识问答",
                    "专业文档", "学术研究", "商业分析"
                ],
                performance_profile={
                    "reasoning": "excellent",
                    "creativity": "good",
                    "coding": "good",
                    "factual": "excellent",
                    "speed": "medium",
                    "context": "excellent"
                },
                special_features=["中文优化", "知识丰富", "文化理解", "32K上下文"]
            ),

            "qwen-plus": ModelCapability(
                name="qwen-plus",
                provider="alibaba",
                strengths=[
                    "中文处理优秀", "平衡性能好", "多领域知识",
                    "实用性强", "性价比高", "响应稳定"
                ],
                suitable_tasks=[
                    "中文问答", "通用任务", "知识整理",
                    "实用工具", "日常助手", "信息处理",
                    "内容生成", "文本分析"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "good",
                    "coding": "good",
                    "factual": "good",
                    "speed": "fast",
                    "context": "good"
                },
                special_features=["中文友好", "实用导向", "性价比高", "稳定可靠"]
            ),

            "qwen3-max-preview": ModelCapability(
                name="qwen3-max-preview",
                provider="alibaba",
                strengths=[
                    "第三代架构", "推理能力增强", "多模态理解",
                    "代码能力提升", "知识更新", "长文本优化"
                ],
                suitable_tasks=[
                    "复杂推理", "代码生成", "多模态分析",
                    "长文档处理", "专业问答", "创新设计",
                    "技术研究", "数据分析"
                ],
                performance_profile={
                    "reasoning": "excellent",
                    "creativity": "excellent",
                    "coding": "excellent",
                    "factual": "excellent",
                    "speed": "medium",
                    "context": "excellent"
                },
                special_features=["Qwen3架构", "多模态", "长文本", "最新技术"]
            ),

            # ============= 智谱AI系列 =============
            "glm-4.5": ModelCapability(
                name="glm-4.5",
                provider="zhipu",
                strengths=[
                    "多模态能力强", "视觉理解好", "综合分析能力",
                    "创新性思维", "中文优化", "跨媒体处理"
                ],
                suitable_tasks=[
                    "多模态任务", "创新设计", "综合分析",
                    "图文理解", "跨领域问题", "创意项目",
                    "视觉问答", "内容创作"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "excellent",
                    "coding": "good",
                    "factual": "good",
                    "speed": "medium",
                    "context": "good"
                },
                special_features=["多模态", "创新思维", "综合能力", "中文优秀"]
            ),

            # ============= Qwen3-Coder系列 (代码专精) =============
            "qwen3-coder-480b-a35b-instruct": ModelCapability(
                name="qwen3-coder-480b-a35b-instruct",
                provider="alibaba",
                strengths=[
                    "代码生成顶尖", "算法理解深", "多语言编程",
                    "代码补全精准", "bug修复能力", "架构设计",
                    "480B参数规模", "指令遵循好"
                ],
                suitable_tasks=[
                    "代码生成", "算法实现", "代码审查", "bug修复",
                    "重构优化", "技术文档", "API设计", "测试编写",
                    "性能优化", "架构设计"
                ],
                performance_profile={
                    "reasoning": "excellent",
                    "creativity": "good",
                    "coding": "outstanding",     # 编程能力：杰出
                    "factual": "excellent",
                    "speed": "medium",
                    "context": "excellent"
                },
                special_features=["480B超大规模", "代码专精", "多语言", "指令遵循"]
            ),

            "qwen3-coder-plus1": ModelCapability(
                name="qwen3-coder-plus1",
                provider="alibaba",
                strengths=[
                    "代码生成优秀", "编程效率高", "多语言支持",
                    "快速开发", "实用导向", "性价比好"
                ],
                suitable_tasks=[
                    "快速开发", "代码补全", "简单重构", "脚本编写",
                    "工具开发", "自动化任务", "原型开发", "代码转换"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "medium",
                    "coding": "excellent",
                    "factual": "good",
                    "speed": "fast",
                    "context": "good"
                },
                special_features=["代码优化", "快速响应", "实用性强", "性价比高"]
            ),

            "qwen3-coder-plus": ModelCapability(
                name="qwen3-coder-plus",
                provider="alibaba",
                strengths=[
                    "代码能力强", "平衡性能", "多场景适用",
                    "开发效率", "稳定可靠"
                ],
                suitable_tasks=[
                    "通用编程", "代码生成", "技术问答", "开发辅助",
                    "代码解释", "学习辅导", "代码优化"
                ],
                performance_profile={
                    "reasoning": "good",
                    "creativity": "medium",
                    "coding": "excellent",
                    "factual": "good",
                    "speed": "fast",
                    "context": "good"
                },
                special_features=["代码能力", "平衡发展", "稳定性好", "实用"]
            ),

            # ============= OpenMatrix系列 (开源生态) =============
            "openmatrix-qwen3-235b-inst-fp8": ModelCapability(
                name="openmatrix-qwen3-235b-inst-fp8",
                provider="openmatrix",
                strengths=[
                    "超大规模235B", "指令遵循精准", "FP8优化",
                    "推理效率高", "知识广博", "多任务能力"
                ],
                suitable_tasks=[
                    "复杂推理", "专业问答", "深度分析", "知识整合",
                    "多步骤任务", "系统设计", "研究辅助", "创新思考"
                ],
                performance_profile={
                    "reasoning": "excellent",
                    "creativity": "good",
                    "coding": "good",
                    "factual": "excellent",
                    "speed": "medium",
                    "context": "excellent"
                },
                special_features=["235B规模", "FP8量化", "开源生态", "高效推理"]
            )
        }
    
    async def intelligent_model_selection(
        self, 
        question: str, 
        available_models: List[ModelConfig]
    ) -> Dict[str, Any]:
        """
        智能模型选择
        
        Args:
            question: 用户问题
            available_models: 可用模型列表
            
        Returns:
            选择结果包含推荐模型和分析理由
        """
        print("🧠 正在进行智能模型分析...")
        
        # 1. 构建模型知识提示
        model_descriptions = self._build_model_descriptions(available_models)
        
        # 2. 创建分析提示
        analysis_prompt = self._create_analysis_prompt(question, model_descriptions)
        
        # 3. LLM分析和推荐
        try:
            response = await call_llm_async(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=self.analyzer_model,
                max_tokens=1500,
                temperature=0.3
            )
            
            # 4. 解析推荐结果
            recommendation = self._parse_recommendation(response, available_models)
            
            return recommendation
            
        except Exception as e:
            print(f"⚠️ 智能选择失败，使用回退策略: {str(e)}")
            return self._fallback_selection(question, available_models)
    
    def _build_model_descriptions(self, available_models: List[ModelConfig]) -> str:
        """构建可用模型的描述"""
        descriptions = []
        
        for model in available_models:
            model_name = model.name
            if model_name in self.model_knowledge:
                capability = self.model_knowledge[model_name]
                
                desc = f"""
**{model_name}**:
- 核心优势: {', '.join(capability.strengths)}
- 适合任务: {', '.join(capability.suitable_tasks)}
- 性能特点: 推理能力{capability.performance_profile['reasoning']}, 创造力{capability.performance_profile['creativity']}, 编程能力{capability.performance_profile['coding']}, 事实准确性{capability.performance_profile['factual']}, 响应速度{capability.performance_profile['speed']}
- 特殊功能: {', '.join(capability.special_features)}
"""
                descriptions.append(desc)
            else:
                # 对于未知模型，提供基本信息
                descriptions.append(f"**{model_name}**: 通用AI模型，具备基础的问答和分析能力")
        
        return "\n".join(descriptions)
    
    def _create_analysis_prompt(self, question: str, model_descriptions: str) -> str:
        """创建分析提示"""
        return f"""
你是一个专业的AI模型选择专家。请分析用户问题并从可用模型中推荐最适合的3个模型组合。

用户问题：
{question}

可用模型及其能力：
{model_descriptions}

请按以下步骤进行分析：

1. **问题分析**：分析问题的类型、复杂度、所需能力
2. **需求匹配**：确定解决这个问题需要什么样的AI能力
3. **模型评估**：评估每个模型在这个问题上的适合度
4. **组合推荐**：选择3个最适合的模型，考虑能力互补

请严格按照以下JSON格式输出：

```json
{{
    "problem_analysis": {{
        "question_type": "问题类型",
        "complexity_level": "复杂度等级(简单/中等/复杂)",
        "required_capabilities": ["所需能力1", "所需能力2", "所需能力3"],
        "key_challenges": ["主要挑战1", "主要挑战2"]
    }},
    "recommended_models": [
        {{
            "model_name": "模型名称",
            "rank": 1,
            "suitability_score": 9.5,
            "reasons": ["选择理由1", "选择理由2"],
            "expected_contribution": "预期贡献"
        }},
        {{
            "model_name": "模型名称", 
            "rank": 2,
            "suitability_score": 8.8,
            "reasons": ["选择理由1", "选择理由2"],
            "expected_contribution": "预期贡献"
        }},
        {{
            "model_name": "模型名称",
            "rank": 3, 
            "suitability_score": 8.2,
            "reasons": ["选择理由1", "选择理由2"],
            "expected_contribution": "预期贡献"
        }}
    ],
    "combination_strategy": "组合策略说明",
    "confidence_level": "高/中/低"
}}
```

注意：
- 只从提供的可用模型中选择
- 优先考虑能力互补的组合
- 确保推荐的模型名称完全匹配可用模型列表
- 适合度评分范围为0-10分
"""
    
    def _parse_recommendation(
        self, 
        response: str, 
        available_models: List[ModelConfig]
    ) -> Dict[str, Any]:
        """解析LLM推荐结果"""
        
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                recommendation = json.loads(json_str)
                
                # 验证推荐的模型是否在可用列表中
                available_model_names = [m.name for m in available_models]
                valid_models = []
                
                for rec_model in recommendation.get('recommended_models', []):
                    model_name = rec_model.get('model_name', '')
                    if model_name in available_model_names:
                        valid_models.append(rec_model)
                
                # 如果有效模型少于3个，用回退策略补充
                if len(valid_models) < 3:
                    print(f"⚠️ 智能推荐的模型数量不足({len(valid_models)})，使用回退策略补充")
                    fallback = self._fallback_selection("", available_models)
                    
                    # 补充模型
                    existing_names = [m['model_name'] for m in valid_models]
                    for model_name in fallback['selected_models'][:3]:
                        if model_name not in existing_names and len(valid_models) < 3:
                            valid_models.append({
                                'model_name': model_name,
                                'rank': len(valid_models) + 1,
                                'suitability_score': 7.0,
                                'reasons': ['回退选择'],
                                'expected_contribution': '提供备选方案'
                            })
                
                recommendation['recommended_models'] = valid_models[:3]
                recommendation['selected_models'] = [m['model_name'] for m in valid_models[:3]]
                recommendation['analysis_method'] = 'intelligent_llm'
                
                return recommendation
                
        except Exception as e:
            print(f"⚠️ 解析推荐结果失败: {str(e)}")
        
        # 解析失败时使用回退策略
        return self._fallback_selection("", available_models)
    
    def _fallback_selection(
        self,
        question: str,
        available_models: List[ModelConfig]
    ) -> Dict[str, Any]:
        """回退选择策略 - 基于模型能力的优先级排序"""

        # 优先级策略：综合能力 > 专精能力 > 轻量模型
        priority_order = [
            # 顶级综合能力模型
            "claude_sonnet4", "qwen3-max-preview", "gpt-41-0414-global",
            # 代码专精模型
            "qwen3-coder-480b-a35b-instruct",
            # 优秀综合模型
            "qwen-max", "openmatrix-qwen3-235b-inst-fp8",
            # 平衡性能模型
            "claude37_sonnet_new", "qwen-plus", "qwen3-coder-plus1",
            # 多模态和创新模型
            "glm-4.5",
            # 快速轻量模型
            "gpt-5-mini-0807-global", "gpt-41-mini-0414-global",
            "qwen3-coder-plus"
        ]
        
        available_names = [m.name for m in available_models]
        selected = []
        
        # 按优先级选择
        for model_name in priority_order:
            if model_name in available_names and len(selected) < 3:
                selected.append(model_name)
        
        # 如果不足3个，从剩余模型中选择
        while len(selected) < 3 and len(selected) < len(available_names):
            for model in available_models:
                if model.name not in selected:
                    selected.append(model.name)
                    break
        
        return {
            'problem_analysis': {
                'question_type': '通用问题',
                'complexity_level': '中等',
                'required_capabilities': ['综合分析', '准确回答'],
                'key_challenges': ['信息整合', '逻辑推理']
            },
            'recommended_models': [
                {
                    'model_name': model_name,
                    'rank': i + 1,
                    'suitability_score': 8.0 - i * 0.5,
                    'reasons': ['回退策略选择'],
                    'expected_contribution': f'提供{["主要", "辅助", "补充"][i]}观点'
                }
                for i, model_name in enumerate(selected[:3])
            ],
            'selected_models': selected[:3],
            'combination_strategy': '基于模型优先级的回退选择',
            'confidence_level': '中',
            'analysis_method': 'fallback'
        }