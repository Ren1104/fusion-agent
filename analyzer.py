"""
AI Fusion 分析模块
包含智能模型选择和质量分析功能
"""

import json
import re
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ============================================
# 向后兼容: ModelConfig 定义
# ============================================

@dataclass
class ModelConfig:
    """模型配置(向后兼容)"""
    name: str
    provider: str = "unknown"
    api_key: str = ""
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7


# ============================================
# LLM 调用函数(兼容旧代码)
# ============================================

async def call_llm_async(
    messages,
    model,
    max_tokens=2000,
    temperature=0.7,
    registry=None,
    trace_id=None,
    return_response_obj: bool = False,
    parent_observation_id: Optional[str] = None,
    langfuse_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    兼容的 LLM 调用函数
    实际调用 providers.ModelRegistry

    Args:
        messages: 消息列表
        model: 模型ID
        max_tokens: 最大token数
        temperature: 温度参数
        registry: ModelRegistry 实例 (可选，如果不提供会创建新实例)
        trace_id: Langfuse trace ID (可选)
        return_response_obj: 是否返回完整响应对象（包含 usage 等信息）
        parent_observation_id: Langfuse 父 span ID
        langfuse_metadata: 附加的 Langfuse 元数据
        **kwargs: 其他参数
    """
    from providers import ModelRegistry

    # 如果没有提供 registry，创建一个新的（向后兼容）
    if registry is None:
        registry = ModelRegistry()
        await registry.discover_all_models()

    response = await registry.call_model(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        trace_id=trace_id,
        parent_observation_id=parent_observation_id,
        langfuse_metadata=langfuse_metadata,
        **kwargs
    )

    if return_response_obj:
        return response
    return response.text




import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
# Imports handled in header


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

    def __init__(self, registry=None):
        self.analyzer_model = "claude_sonnet4"  # 用于分析的模型
        self.model_knowledge = self._build_model_knowledge()
        self.registry = registry  # ModelRegistry 实例
    
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
        available_models: List[ModelConfig],
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
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
                temperature=0.3,
                registry=self.registry,  # 传递 registry
                trace_id=trace_id,
                parent_observation_id=parent_observation_id,
                langfuse_metadata={
                    "component": "model_selector",
                    "stage": "intelligent_selection"
                },
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


import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
# Imports handled in header
import hashlib
import json


@dataclass
class QualityMetrics:
    """质量评估指标"""
    completeness_score: float  # 完整性评分 (0-10)
    accuracy_score: float      # 准确性评分 (0-10)
    clarity_score: float       # 清晰度评分 (0-10)
    relevance_score: float     # 相关性评分 (0-10)
    overall_score: float       # 综合评分 (0-10)
    word_count: int           # 词数
    sentence_count: int       # 句数
    readability_score: float  # 可读性评分 (0-10)
    information_density: float # 信息密度 (0-10)


class AIFusionQualityAnalyzer:
    """AI Fusion质量分析器"""

    def __init__(self, registry=None):
        self.evaluator_model = "claude_sonnet4"  # 用于评估的模型
        self.registry = registry  # ModelRegistry 实例
        self._current_trace_id: Optional[str] = None
        self._current_parent_observation_id: Optional[str] = None
    
    async def analyze_quality(
        self,
        question: str,
        llm_responses: List[Dict],
        fusion_answer: str,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        分析回答质量
        
        Args:
            question: 原始问题
            llm_responses: 各模型的回答数据
            fusion_answer: 融合后的回答
            
        Returns:
            质量分析结果
        """
        print("🔍 开始质量分析...")

        self._current_trace_id = trace_id
        self._current_parent_observation_id = parent_observation_id

        try:
            # 1. 计算基础指标
            basic_metrics = {}
            for response in llm_responses:
                if response['success']:
                    basic_metrics[response['model_name']] = self._calculate_basic_metrics(
                        response['response']
                    )

            # 融合回答的基础指标
            basic_metrics['fusion_answer'] = self._calculate_basic_metrics(fusion_answer)

            # 2. LLM评估（异步并发）
            llm_evaluations = await self._evaluate_with_llm(
                question, llm_responses, fusion_answer
            )

            # 3. 内容语义分析（增强差异化能力）
            content_analysis = await self._perform_content_semantic_analysis(
                question, llm_responses, fusion_answer
            )

            # 4. 对比分析（增强）
            comparison_analysis = self._perform_enhanced_comparison_analysis(
                basic_metrics, llm_evaluations, content_analysis
            )

            # 5. 逻辑一致性验证
            consistency_check = self._perform_consistency_validation(
                llm_evaluations, comparison_analysis, content_analysis
            )

            # 6. 质量排名（经过一致性校正）
            quality_ranking = self._calculate_validated_quality_ranking(
                basic_metrics, llm_evaluations, consistency_check
            )

            # 7. 融合效果量化分析（新增）
            fusion_effectiveness = self._analyze_fusion_effectiveness(
                llm_evaluations, comparison_analysis, content_analysis
            )

            # 8. 速度-质量权衡分析（新增）
            speed_quality_tradeoff = self._analyze_speed_quality_tradeoff(
                llm_responses, llm_evaluations
            )

            return {
                'basic_metrics': basic_metrics,
                'llm_evaluations': llm_evaluations,
                'content_analysis': content_analysis,
                'comparison_analysis': comparison_analysis,
                'consistency_check': consistency_check,
                'quality_ranking': quality_ranking,
                'fusion_effectiveness': fusion_effectiveness,  # 新增融合效果分析
                'speed_quality_tradeoff': speed_quality_tradeoff  # 新增速度质量权衡分析
            }
        finally:
            # 清理上下文
            self._current_trace_id = None
            self._current_parent_observation_id = None
    
    def _calculate_basic_metrics(self, text: str) -> QualityMetrics:
        """计算基础质量指标"""
        if not text:
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # 字符数统计（更准确的中文文本长度计算）
        char_count = len(text)
        # 中文词汇统计（按字符和空格分割）
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)
        word_count = len(words)
        # 句子统计（改进的分句逻辑）
        sentences = re.split(r'[.!?。！？]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # 可读性评分（基于平均句长）
        avg_sentence_length = word_count / max(sentence_count, 1)
        readability_score = max(0, min(10, 10 - (avg_sentence_length - 15) * 0.2))
        
        # 信息密度（基于内容多样性）
        unique_words = len(set(text.lower().split()))
        information_density = min(10, (unique_words / max(word_count, 1)) * 20)
        
        # 基础评分（会在LLM评估中更新）
        return QualityMetrics(
            completeness_score=0,  # 待LLM评估
            accuracy_score=0,      # 待LLM评估
            clarity_score=readability_score,
            relevance_score=0,     # 待LLM评估
            overall_score=0,       # 待计算
            word_count=char_count,  # 使用字符数而非词数
            sentence_count=sentence_count,
            readability_score=readability_score,
            information_density=information_density
        )
    
    async def _evaluate_with_llm(
        self,
        question: str,
        llm_responses: List[Dict],
        fusion_answer: str
    ) -> Dict[str, QualityMetrics]:
        """使用LLM评估回答质量（采用对比评分机制）"""

        # 准备评估任务
        answer_sources = {}

        # 收集所有成功的回答
        for response in llm_responses:
            if response['success']:
                answer_sources[response['model_name']] = response['response']

        # 添加融合回答
        answer_sources['fusion_answer'] = fusion_answer

        # **第一阶段：批量对比评分（新增）**
        # 这会让评估器一次性看到所有回答，从而给出相对评分
        comparative_scores = await self._comparative_batch_evaluation(
            question, answer_sources
        )

        # **第二阶段：单独评估（保留原有详细评估）**
        evaluation_tasks = []
        source_names = []

        for source_name, answer_text in answer_sources.items():
            # 传入对比评分作为参考
            base_score = comparative_scores.get(source_name, 7.0)
            evaluation_tasks.append(
                self._evaluate_single_answer(
                    question,
                    answer_text,
                    source_name,
                    base_reference_score=base_score  # 传入参考分
                )
            )
            source_names.append(source_name)

        # 并发执行详细评估
        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # 整理结果
        llm_evaluations = {}
        for i, source_name in enumerate(source_names):
            if i < len(evaluation_results) and not isinstance(evaluation_results[i], Exception):
                llm_evaluations[source_name] = evaluation_results[i]
            else:
                # 如果评估失败，使用默认值
                print(f"⚠️ {source_name} 评估失败，使用默认值")
                llm_evaluations[source_name] = QualityMetrics(5, 5, 5, 5, 5, 0, 0, 5, 5)

        return llm_evaluations

    async def _comparative_batch_evaluation(
        self,
        question: str,
        answer_sources: Dict[str, str]
    ) -> Dict[str, float]:
        """
        批量对比评分：一次性评估所有回答，给出相对评分
        这能确保评分的区分度和一致性
        """
        print("🔍 正在进行批量对比评分...")

        # 构建对比评分prompt
        comparative_prompt = f"""
你是一位严格的质量评估专家。现在有{len(answer_sources)}个AI模型对同一问题给出了回答。

**问题：**
{question}

**各模型回答：**
"""

        for i, (source_name, answer_text) in enumerate(answer_sources.items(), 1):
            # 限制每个回答的长度以避免prompt过长
            truncated_answer = answer_text[:500] + "..." if len(answer_text) > 500 else answer_text
            comparative_prompt += f"""
【回答{i}: {source_name}】
{truncated_answer}

"""

        comparative_prompt += """
**对比评分任务：**

请对所有回答进行相对评分（0-10分），要求：

1. **必须拉开评分差距**：最高分和最低分的差距至少1.5分
2. **基于质量排序**：先判断哪个回答质量最好、哪个最差、哪个居中
3. **参考评分区间**：
   - 最优秀的回答：7.5-9.0分
   - 中等质量的回答：6.0-7.5分
   - 较差的回答：4.5-6.0分

4. **评分标准**：综合考虑完整性、准确性、清晰度、相关性

**输出格式（严格遵守）：**

```
回答1({源名称}): X.X分 - [一句话评价]
回答2({源名称}): X.X分 - [一句话评价]
回答3({源名称}): X.X分 - [一句话评价]
...
```

**排名说明：**
[简要说明为什么这样排序，各回答的主要差异在哪里]
"""

        try:
            response = await call_llm_async(
                messages=[{"role": "user", "content": comparative_prompt}],
                model=self.evaluator_model,
                max_tokens=1000,
                temperature=0.2,
                registry=self.registry,
                trace_id=self._current_trace_id,
                parent_observation_id=self._current_parent_observation_id,
                langfuse_metadata={
                    "component": "quality_analyzer",
                    "stage": "comparative_scoring"
                },
            )

            # 解析对比评分结果
            comparative_scores = {}
            for line in response.split('\n'):
                # 匹配格式：回答X(源名称): X.X分
                for source_name in answer_sources.keys():
                    if source_name in line and ':' in line:
                        # 尝试提取评分
                        score_match = re.search(r'(\d+\.?\d*)分', line)
                        if score_match:
                            score = float(score_match.group(1))
                            comparative_scores[source_name] = max(0, min(10, score))
                            break

            # 验证是否所有回答都有评分
            for source_name in answer_sources.keys():
                if source_name not in comparative_scores:
                    print(f"⚠️ {source_name} 未获得对比评分，使用默认值")
                    comparative_scores[source_name] = 7.0

            # 验证评分区分度
            scores = list(comparative_scores.values())
            if len(scores) > 1:
                score_range = max(scores) - min(scores)
                if score_range < 1.0:
                    print(f"⚠️ 对比评分区分度不足({score_range:.1f}分)，将进行调整")
                    # 强制拉开差距
                    sorted_items = sorted(comparative_scores.items(), key=lambda x: x[1], reverse=True)
                    for i, (name, _) in enumerate(sorted_items):
                        comparative_scores[name] = 8.5 - i * 0.8  # 从8.5开始递减

            print(f"✅ 对比评分完成，评分区间: {min(scores):.1f} - {max(scores):.1f}")
            return comparative_scores

        except Exception as e:
            print(f"⚠️ 对比评分失败: {str(e)}, 使用默认评分")
            # 返回默认评分
            return {name: 7.0 for name in answer_sources.keys()}
    
    async def _evaluate_single_answer(
        self,
        question: str,
        answer: str,
        source_name: str,
        base_reference_score: float = 7.0
    ) -> QualityMetrics:
        """评估单个回答的质量"""

        evaluation_prompt = f"""
你是一位严格的内容质量评估专家，负责对AI模型的回答进行客观、准确、有区分度的评分。

原始问题：
{question}

待评估回答（来源: {source_name}）：
{answer}

**参考评分：** 通过对比评估，该回答的初步综合质量为 {base_reference_score:.1f}/10分。
请在此基础上，进行更细致的各维度评分。

## 评分要求

请从以下5个维度对回答进行评分（0-10分，保留一位小数）：

1. **完整性(Completeness)**: 回答是否完整地覆盖了问题的各个方面
2. **准确性(Accuracy)**: 回答中的信息、概念、示例是否准确可靠
3. **清晰度(Clarity)**: 回答的表达是否清晰、逻辑是否连贯、易于理解
4. **相关性(Relevance)**: 回答是否紧扣问题主题，没有离题或冗余内容
5. **综合质量(Overall)**: 整体回答质量（必须是前4个维度的平均值±0.5分）

## 评分标准（严格执行）

**使用相对评分而非绝对评分：**
- 9.0-10.0分: 卓越表现，超越预期，几乎无瑕疵
- 7.5-8.9分: 优秀表现，质量很高但有小的改进空间
- 6.0-7.4分: 良好表现，基本满足需求但有明显不足
- 4.5-5.9分: 及格表现，能回答问题但质量一般
- 3.0-4.4分: 较差表现，存在明显问题和遗漏
- 0.0-2.9分: 不合格，严重错误或完全未回答

**禁止评分行为：**
- ❌ 禁止给所有维度都打8-10分的高分
- ❌ 禁止不同回答获得完全相同的评分
- ❌ 禁止给出模糊的评分理由

**必须执行：**
- ✅ 必须引用回答中的具体内容片段作为评分依据
- ✅ 必须明确指出至少2个优点和2个不足
- ✅ 必须基于实际内容质量给分，不受回答长度影响
- ✅ 综合评分必须接近前4个维度的平均值

## 输出格式（严格遵守）

完整性评分: X.X
准确性评分: X.X
清晰度评分: X.X
相关性评分: X.X
综合评分: X.X

**评分依据（必填）：**

【完整性】
✅ 优点: [引用具体内容片段，说明哪些方面做得好]
❌ 不足: [引用具体内容片段或指出缺失的内容]

【准确性】
✅ 优点: [引用准确的内容示例]
❌ 不足: [指出不准确、误导或有争议的内容]

【清晰度】
✅ 优点: [说明表达清晰的部分]
❌ 不足: [指出表达不清或逻辑混乱的部分]

【相关性】
✅ 优点: [说明紧扣主题的部分]
❌ 不足: [指出离题、冗余或不够聚焦的内容]

**独特特征：**
[说明这个回答与其他典型回答相比的独特之处，至少50字]

**核心建议：**
[给出3个具体的改进建议]
"""
        
        try:
            print(f"🤖 正在评估 {source_name} 的回答质量...")

            response = await call_llm_async(
                messages=[{"role": "user", "content": evaluation_prompt}],
                model=self.evaluator_model,
                max_tokens=2000,  # 增加token数以容纳详细的评分依据
                temperature=0.2,   # 降低温度以提高评分一致性
                registry=self.registry,
                trace_id=self._current_trace_id,
                parent_observation_id=self._current_parent_observation_id,
                langfuse_metadata={
                    "component": "quality_analyzer",
                    "stage": "single_answer_evaluation",
                    "source": source_name,
                },
            )
            
            # 解析评分结果
            scores = self._parse_evaluation_response(response)

            # 计算准确的字符数和句子数
            char_count = len(answer)
            words = re.findall(r'[\w\u4e00-\u9fff]+', answer)
            word_count = len(words)
            sentences = re.split(r'[.!?。！？]+', answer)
            sentence_count = len([s for s in sentences if s.strip()])

            # **关键修复：使用对比评分作为基准，只允许小幅调整**
            # 这样可以保持对比评分的区分度，同时允许细微的维度差异

            completeness = scores.get('completeness', base_reference_score)
            accuracy = scores.get('accuracy', base_reference_score)
            clarity = scores.get('clarity', base_reference_score)
            relevance = scores.get('relevance', base_reference_score)

            # 各维度评分不能偏离对比评分太多（±1.0分以内）
            completeness = max(base_reference_score - 1.0, min(base_reference_score + 1.0, completeness))
            accuracy = max(base_reference_score - 1.0, min(base_reference_score + 1.0, accuracy))
            clarity = max(base_reference_score - 1.0, min(base_reference_score + 1.0, clarity))
            relevance = max(base_reference_score - 1.0, min(base_reference_score + 1.0, relevance))

            # 综合评分必须接近对比评分
            dimension_avg = (completeness + accuracy + clarity + relevance) / 4
            overall = (base_reference_score * 0.7 + dimension_avg * 0.3)  # 70%权重来自对比评分

            return QualityMetrics(
                completeness_score=round(completeness, 1),
                accuracy_score=round(accuracy, 1),
                clarity_score=round(clarity, 1),
                relevance_score=round(relevance, 1),
                overall_score=round(overall, 1),
                word_count=char_count,  # 使用字符数
                sentence_count=sentence_count,
                readability_score=scores.get('clarity', 5.0),
                information_density=min(10, len(set([w.lower() for w in words])) / max(word_count, 1) * 20)
            )
            
        except Exception as e:
            print(f"⚠️ 评估 {source_name} 时出错: {str(e)}")
            return QualityMetrics(5, 5, 5, 5, 5, 0, 0, 5, 5)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, float]:
        """解析LLM评估响应"""
        scores = {}
        
        patterns = {
            'completeness': r'完整性评分[：:]\s*(\d+\.?\d*)',
            'accuracy': r'准确性评分[：:]\s*(\d+\.?\d*)',
            'clarity': r'清晰度评分[：:]\s*(\d+\.?\d*)',
            'relevance': r'相关性评分[：:]\s*(\d+\.?\d*)',
            'overall': r'综合评分[：:]\s*(\d+\.?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    scores[key] = max(0, min(10, score))  # 限制在0-10范围内
                except ValueError:
                    scores[key] = 5.0
            else:
                scores[key] = 5.0
        
        return scores
    
    async def _perform_content_semantic_analysis(
        self,
        question: str,
        llm_responses: List[Dict],
        fusion_answer: str
    ) -> Dict[str, Any]:
        """执行内容语义分析，识别真实差异"""
        print("🔍 正在进行内容语义分析...")

        content_analysis = {
            'content_uniqueness': {},
            'approach_differences': {},
            'perspective_analysis': {},
            'semantic_similarity': {},
            'content_themes': {},
            'structure_patterns': {},
            'individualized_profiles': {}  # 新增：个性化档案
        }

        try:
            # 1. 分析各模型的角度和方法差异
            approach_analysis = await self._analyze_response_approaches(
                question, [r for r in llm_responses if r['success']]
            )
            content_analysis['approach_differences'] = approach_analysis

            # 2. 计算内容相似度
            similarity_matrix = self._calculate_content_similarity(
                [r['response'] for r in llm_responses if r['success']], fusion_answer
            )
            content_analysis['semantic_similarity'] = similarity_matrix

            # 3. 提取关键主题和观点
            themes_analysis = await self._extract_content_themes(
                question, [r for r in llm_responses if r['success']], fusion_answer
            )
            content_analysis['content_themes'] = themes_analysis

            # 4. 分析结构模式
            structure_analysis = self._analyze_structure_patterns(
                [r for r in llm_responses if r['success']], fusion_answer
            )
            content_analysis['structure_patterns'] = structure_analysis

            # 5. 评估内容独特性
            uniqueness_scores = self._calculate_content_uniqueness(
                [r for r in llm_responses if r['success']], fusion_answer
            )
            content_analysis['content_uniqueness'] = uniqueness_scores

            # 6. **新增：深度个性化分析**
            individualized_profiles = await self._deep_individualized_analysis(
                question, [r for r in llm_responses if r['success']]
            )
            content_analysis['individualized_profiles'] = individualized_profiles

        except Exception as e:
            print(f"⚠️ 内容语义分析失败: {str(e)}")
            # 返回默认结构
            pass

        return content_analysis
    
    def _perform_enhanced_comparison_analysis(
        self, 
        basic_metrics: Dict[str, QualityMetrics],
        llm_evaluations: Dict[str, QualityMetrics],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行增强的对比分析（融入内容语义分析）"""
        
        analysis = {
            'fusion_advantages': [],
            'model_strengths': {},
            'improvement_areas': [],
            'statistical_summary': {},
            'individual_analysis': {}
        }
        
        # 获取融合回答的评分
        fusion_eval = llm_evaluations.get('fusion_answer')
        if not fusion_eval:
            return analysis
        
        # 分析融合回答的优势
        model_evaluations = {k: v for k, v in llm_evaluations.items() if k != 'fusion_answer'}
        
        if model_evaluations:
            # 计算各维度平均分
            avg_scores = {
                'completeness': sum(eval.completeness_score for eval in model_evaluations.values()) / len(model_evaluations),
                'accuracy': sum(eval.accuracy_score for eval in model_evaluations.values()) / len(model_evaluations),
                'clarity': sum(eval.clarity_score for eval in model_evaluations.values()) / len(model_evaluations),
                'relevance': sum(eval.relevance_score for eval in model_evaluations.values()) / len(model_evaluations),
                'overall': sum(eval.overall_score for eval in model_evaluations.values()) / len(model_evaluations)
            }
            
            # 融合回答的优势分析
            if fusion_eval.completeness_score > avg_scores['completeness']:
                analysis['fusion_advantages'].append(f"完整性提升 {fusion_eval.completeness_score - avg_scores['completeness']:.1f}分")
            
            if fusion_eval.accuracy_score > avg_scores['accuracy']:
                analysis['fusion_advantages'].append(f"准确性提升 {fusion_eval.accuracy_score - avg_scores['accuracy']:.1f}分")
            
            if fusion_eval.clarity_score > avg_scores['clarity']:
                analysis['fusion_advantages'].append(f"清晰度提升 {fusion_eval.clarity_score - avg_scores['clarity']:.1f}分")
            
            if fusion_eval.overall_score > avg_scores['overall']:
                analysis['fusion_advantages'].append(f"综合质量提升 {fusion_eval.overall_score - avg_scores['overall']:.1f}分")
            
            # 个性化模型分析 - 结合内容语义分析
            analysis['model_strengths'] = self._analyze_enhanced_model_strengths(
                model_evaluations, avg_scores, content_analysis
            )
            
            # 详细的个体分析
            analysis['individual_analysis'] = self._analyze_individual_performance(
                model_evaluations, basic_metrics
            )
            
            # 统计摘要
            analysis['statistical_summary'] = {
                'fusion_overall_score': fusion_eval.overall_score,
                'models_avg_score': avg_scores['overall'],
                'improvement': fusion_eval.overall_score - avg_scores['overall'],
                'best_individual_score': max(eval.overall_score for eval in model_evaluations.values()),
                'fusion_vs_best': fusion_eval.overall_score - max(eval.overall_score for eval in model_evaluations.values())
            }
        
        return analysis
    
    def _calculate_quality_ranking(
        self, 
        basic_metrics: Dict[str, QualityMetrics],
        llm_evaluations: Dict[str, QualityMetrics]
    ) -> List[Dict[str, Any]]:
        """计算质量排名"""
        
        ranking = []
        
        for source_name, evaluation in llm_evaluations.items():
            basic = basic_metrics.get(source_name)
            
            # 确保评分的一致性检查
            overall_score = evaluation.overall_score
            dimension_avg = (evaluation.completeness_score + evaluation.accuracy_score + 
                           evaluation.clarity_score + evaluation.relevance_score) / 4
            
            # 如果综合评分与各维度平均分差异过大，进行调整
            if abs(overall_score - dimension_avg) > 2.0:
                overall_score = dimension_avg
                print(f"⚠️ {source_name} 的评分存在不一致，已调整综合评分为 {overall_score:.1f}")
            
            ranking.append({
                'source': source_name,
                'overall_score': overall_score,
                'completeness': evaluation.completeness_score,
                'accuracy': evaluation.accuracy_score,
                'clarity': evaluation.clarity_score,
                'relevance': evaluation.relevance_score,
                'word_count': basic.word_count if basic else 0,
                'is_fusion': source_name == 'fusion_answer'
            })
        
        # 按综合评分排序，分数相同时按准确性排序
        ranking.sort(key=lambda x: (x['overall_score'], x['accuracy']), reverse=True)
        
        # 添加排名，处理并列情况
        current_rank = 1
        for i, item in enumerate(ranking):
            if i > 0 and ranking[i-1]['overall_score'] != item['overall_score']:
                current_rank = i + 1
            item['rank'] = current_rank
        
        return ranking
    
    def _analyze_individual_model_strengths(
        self, 
        model_evaluations: Dict[str, QualityMetrics], 
        avg_scores: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """分析每个模型的个性化优势"""
        strengths = {}
        
        for model_name, eval_result in model_evaluations.items():
            model_strengths = []
            
            # 相对优势分析（相比平均水平）
            if eval_result.completeness_score > avg_scores['completeness'] + 0.5:
                diff = eval_result.completeness_score - avg_scores['completeness']
                model_strengths.append(f"完整性超群(+{diff:.1f})")
            elif eval_result.completeness_score >= 8.0:
                model_strengths.append("完整性优秀")
            
            if eval_result.accuracy_score > avg_scores['accuracy'] + 0.5:
                diff = eval_result.accuracy_score - avg_scores['accuracy']
                model_strengths.append(f"准确性出众(+{diff:.1f})")
            elif eval_result.accuracy_score >= 8.0:
                model_strengths.append("准确性可靠")
            
            if eval_result.clarity_score > avg_scores['clarity'] + 0.5:
                diff = eval_result.clarity_score - avg_scores['clarity']
                model_strengths.append(f"表达最清晰(+{diff:.1f})")
            elif eval_result.clarity_score >= 8.0:
                model_strengths.append("表达清晰")
            
            if eval_result.relevance_score > avg_scores['relevance'] + 0.5:
                diff = eval_result.relevance_score - avg_scores['relevance']
                model_strengths.append(f"高度相关(+{diff:.1f})")
            elif eval_result.relevance_score >= 8.0:
                model_strengths.append("切题精准")
            
            # 综合优势
            if eval_result.overall_score > avg_scores['overall'] + 1.0:
                model_strengths.append("综合表现最佳")
            elif eval_result.overall_score > avg_scores['overall'] + 0.5:
                model_strengths.append("综合表现优秀")
            
            # 特色分析
            if eval_result.word_count > 0:
                # 内容丰富度分析
                if eval_result.information_density >= 8.0:
                    model_strengths.append("信息密度高")
                
                # 回答风格分析
                if eval_result.word_count > 300:
                    model_strengths.append("回答详尽")
                elif eval_result.word_count < 100:
                    model_strengths.append("回答简洁")
            
            if model_strengths:
                strengths[model_name] = model_strengths
            else:
                # 即使没有突出优势，也要给出基本评价
                best_dimension = max([
                    ('完整性', eval_result.completeness_score),
                    ('准确性', eval_result.accuracy_score), 
                    ('清晰度', eval_result.clarity_score),
                    ('相关性', eval_result.relevance_score)
                ], key=lambda x: x[1])
                strengths[model_name] = [f"{best_dimension[0]}相对较好({best_dimension[1]:.1f}/10)"]
        
        return strengths
    
    def _analyze_enhanced_model_strengths(
        self,
        model_evaluations: Dict[str, QualityMetrics],
        avg_scores: Dict[str, float],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """增强的模型优势分析（优先使用深度个性化分析结果）"""

        enhanced_strengths = {}

        # **优先使用深度个性化分析结果**
        individualized_profiles = content_analysis.get('individualized_profiles', {})
        if individualized_profiles and isinstance(individualized_profiles, dict):
            profiles_data = individualized_profiles.get('individualized_profiles', {})

            if profiles_data:
                print("✅ 使用深度个性化分析结果生成模型优势描述")
                # 使用深度分析的结果
                for model_name in model_evaluations.keys():
                    if model_name in profiles_data:
                        profile = profiles_data[model_name]
                        strength_list = []

                        # 1. 标志性特征（最重要）
                        signature = profile.get('signature_characteristics', '')
                        if signature:
                            strength_list.append(signature)

                        # 2. 独特贡献点（最多2个）
                        contributions = profile.get('unique_contributions', [])
                        if contributions:
                            for contrib in contributions[:2]:
                                # 截断过长的描述
                                truncated = contrib[:80] + "..." if len(contrib) > 80 else contrib
                                strength_list.append(truncated)

                        # 3. 比较优势
                        advantage = profile.get('comparative_advantage', '')
                        if advantage:
                            truncated = advantage[:100] + "..." if len(advantage) > 100 else advantage
                            strength_list.append(truncated)

                        # 4. 内容风格（可选）
                        style = profile.get('content_style', '')
                        if style and len(strength_list) < 4:
                            truncated = style[:80] + "..." if len(style) > 80 else style
                            strength_list.append(truncated)

                        enhanced_strengths[model_name] = strength_list[:5]  # 最多5条
                    else:
                        # 如果没有深度分析结果，使用回退方案
                        enhanced_strengths[model_name] = self._fallback_strength_analysis(
                            model_name, model_evaluations[model_name], avg_scores, content_analysis
                        )
            else:
                # 深度分析为空，使用回退方案
                print("⚠️ 深度个性化分析结果为空，使用回退方案")
                enhanced_strengths = self._fallback_enhanced_analysis(
                    model_evaluations, avg_scores, content_analysis
                )
        else:
            # 没有深度分析，使用回退方案
            print("⚠️ 未找到深度个性化分析结果，使用回退方案")
            enhanced_strengths = self._fallback_enhanced_analysis(
                model_evaluations, avg_scores, content_analysis
            )

        return enhanced_strengths

    def _fallback_strength_analysis(
        self,
        model_name: str,
        evaluation: QualityMetrics,
        avg_scores: Dict[str, float],
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """单个模型的回退优势分析"""
        strength_list = []

        # 获取内容差异化信息
        uniqueness_scores = content_analysis.get('content_uniqueness', {})
        structure_patterns = content_analysis.get('structure_patterns', {})
        content_themes = content_analysis.get('content_themes', {})

        # 1. 相对优势分析
        if evaluation.completeness_score > avg_scores.get('completeness', 7.0) + 0.5:
            diff = evaluation.completeness_score - avg_scores.get('completeness', 7.0)
            strength_list.append(f"完整性超群(+{diff:.1f})")
        elif evaluation.completeness_score >= 8.0:
            strength_list.append("完整性优秀")

        if evaluation.accuracy_score > avg_scores.get('accuracy', 7.0) + 0.5:
            diff = evaluation.accuracy_score - avg_scores.get('accuracy', 7.0)
            strength_list.append(f"准确性出众(+{diff:.1f})")
        elif evaluation.accuracy_score >= 8.0:
            strength_list.append("准确性可靠")

        # 2. 独特性分析
        uniqueness = uniqueness_scores.get(model_name, 0)
        if uniqueness > 0.3:
            strength_list.append(f"内容独特性高({uniqueness:.1%})")
        elif uniqueness > 0.15:
            strength_list.append(f"有一定独特性({uniqueness:.1%})")

        # 3. 结构特征
        if model_name in structure_patterns:
            structure = structure_patterns[model_name]
            if structure.get('list_items', 0) > 2:
                strength_list.append("结构清晰，善用列表")
            if structure.get('code_blocks', 0) > 0:
                strength_list.append("提供代码示例")

        # 4. 主题覆盖
        model_unique_points = content_themes.get('model_unique_points', {})
        if model_name in model_unique_points:
            unique_points = model_unique_points[model_name]
            if unique_points:
                strength_list.append(f"独特观点: {', '.join(unique_points[:1])}")

        # 如果列表为空，至少给出一个基本评价
        if not strength_list:
            best_dim = max([
                ('完整性', evaluation.completeness_score),
                ('准确性', evaluation.accuracy_score),
                ('清晰度', evaluation.clarity_score),
                ('相关性', evaluation.relevance_score)
            ], key=lambda x: x[1])
            strength_list.append(f"{best_dim[0]}相对较好({best_dim[1]:.1f}/10)")

        return strength_list[:5]

    def _fallback_enhanced_analysis(
        self,
        model_evaluations: Dict[str, QualityMetrics],
        avg_scores: Dict[str, float],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """回退的增强分析（当深度分析失败时）"""
        enhanced_strengths = {}

        for model_name, evaluation in model_evaluations.items():
            enhanced_strengths[model_name] = self._fallback_strength_analysis(
                model_name, evaluation, avg_scores, content_analysis
            )

        return enhanced_strengths
    
    def _analyze_individual_performance(
        self, 
        model_evaluations: Dict[str, QualityMetrics],
        basic_metrics: Dict[str, QualityMetrics]
    ) -> Dict[str, Dict[str, Any]]:
        """详细的个体性能分析"""
        individual_analysis = {}
        
        # 计算各维度的最高分，用于相对比较
        max_scores = {
            'completeness': max(eval.completeness_score for eval in model_evaluations.values()),
            'accuracy': max(eval.accuracy_score for eval in model_evaluations.values()),
            'clarity': max(eval.clarity_score for eval in model_evaluations.values()),
            'relevance': max(eval.relevance_score for eval in model_evaluations.values()),
            'overall': max(eval.overall_score for eval in model_evaluations.values())
        }
        
        for model_name, eval_result in model_evaluations.items():
            basic_metric = basic_metrics.get(model_name)
            
            analysis = {
                'performance_highlights': [],
                'relative_ranking': {},
                'style_characteristics': [],
                'improvement_potential': []
            }
            
            # 性能亮点 - 只有唯一最高分才算"最佳"
            # 统计每个维度有多少个模型达到最高分
            dimension_counts = {
                'completeness': sum(1 for eval in model_evaluations.values() if eval.completeness_score == max_scores['completeness']),
                'accuracy': sum(1 for eval in model_evaluations.values() if eval.accuracy_score == max_scores['accuracy']),
                'clarity': sum(1 for eval in model_evaluations.values() if eval.clarity_score == max_scores['clarity']),
                'relevance': sum(1 for eval in model_evaluations.values() if eval.relevance_score == max_scores['relevance'])
            }

            # 只有唯一最高分才标记为"最佳"，否则标记为"并列最佳"或"优秀"
            if eval_result.completeness_score == max_scores['completeness']:
                if dimension_counts['completeness'] == 1:
                    analysis['performance_highlights'].append("完整性最佳")
                elif eval_result.completeness_score >= 8.0:
                    analysis['performance_highlights'].append(f"完整性优秀(并列第1)")

            if eval_result.accuracy_score == max_scores['accuracy']:
                if dimension_counts['accuracy'] == 1:
                    analysis['performance_highlights'].append("准确性最高")
                elif eval_result.accuracy_score >= 8.0:
                    analysis['performance_highlights'].append(f"准确性优秀(并列第1)")

            if eval_result.clarity_score == max_scores['clarity']:
                if dimension_counts['clarity'] == 1:
                    analysis['performance_highlights'].append("表达最清晰")
                elif eval_result.clarity_score >= 8.0:
                    analysis['performance_highlights'].append(f"清晰度优秀(并列第1)")

            if eval_result.relevance_score == max_scores['relevance']:
                if dimension_counts['relevance'] == 1:
                    analysis['performance_highlights'].append("相关性最强")
                elif eval_result.relevance_score >= 8.0:
                    analysis['performance_highlights'].append(f"相关性优秀(并列第1)")
            
            # 相对排名（在本次任务中的表现）
            models_list = list(model_evaluations.items())
            for dimension in ['completeness', 'accuracy', 'clarity', 'relevance', 'overall']:
                sorted_models = sorted(models_list, 
                                     key=lambda x: getattr(x[1], f'{dimension}_score'), 
                                     reverse=True)
                rank = next(i for i, (name, _) in enumerate(sorted_models, 1) if name == model_name)
                analysis['relative_ranking'][dimension] = f"{rank}/{len(models_list)}"
            
            # 风格特征分析（个性化）
            if basic_metric and basic_metric.word_count > 0:
                char_count = basic_metric.word_count  # 字符数
                
                # 更精细的内容特征分析
                if char_count > 500:
                    analysis['style_characteristics'].append("详细全面型，深度阐释")
                elif char_count > 300:
                    analysis['style_characteristics'].append("中等详细型，结构完整")
                elif char_count < 150:
                    analysis['style_characteristics'].append("简洁精炼型，直击要点") 
                else:
                    analysis['style_characteristics'].append("适中平衡型，结构清晰")
                
                # 信息密度的个性化分析
                if basic_metric.information_density > 8.0:
                    analysis['style_characteristics'].append("高信息密度，内容精细")
                elif basic_metric.information_density > 6.0:
                    analysis['style_characteristics'].append("信息丰富，内容充实")
                elif basic_metric.information_density < 4.0:
                    analysis['style_characteristics'].append("表达舒缓，易于理解")
                
                # 添加可读性特征
                if basic_metric.readability_score > 8.0:
                    analysis['style_characteristics'].append("可读性优秀，行文流畅")
                elif basic_metric.readability_score < 5.0:
                    analysis['style_characteristics'].append("表达复杂，需仔细理解")
            
            # 个性化的改进潜力分析
            dimensions = [
                ('完整性', eval_result.completeness_score),
                ('准确性', eval_result.accuracy_score),
                ('清晰度', eval_result.clarity_score), 
                ('相关性', eval_result.relevance_score)
            ]
            
            # 找出最弱和最强的维度
            sorted_dims = sorted(dimensions, key=lambda x: x[1])
            weakest = sorted_dims[0]
            strongest = sorted_dims[-1]
            
            # 个性化的改进建议
            if weakest[1] < 6.0:
                improvement_suggestions = {
                    '完整性': '可增加回答的全面性和深度',
                    '准确性': '需要提高信息的准确性和可靠性',
                    '清晰度': '表达可以更加清晰明了',
                    '相关性': '需要更好地理解和对针问题核心'
                }
                analysis['improvement_potential'].append(improvement_suggestions.get(weakest[0], f"{weakest[0]}有提升空间"))
            elif weakest[1] < 7.5:
                analysis['improvement_potential'].append(f"{weakest[0]}表现尚可，有进一步优化空间")
            
            # 优势维度分析
            if strongest[1] >= 8.5:
                excellence_descriptions = {
                    '完整性': '在内容完整性方面表现突出，可作为核心竞争力',
                    '准确性': '信息准确性是该模型的最大亮点',
                    '清晰度': '表达清晰度优秀，适合复杂问题解释',
                    '相关性': '对问题的理解和对针性是显著优势'
                }
                analysis['improvement_potential'].append(excellence_descriptions.get(strongest[0], f"在{strongest[0]}方面表现突出"))
            elif strongest[1] >= 8.0:
                analysis['improvement_potential'].append(f"{strongest[0]}表现优秀，可作为相对优势")
            
            # 添加模型特异性分析
            model_specificity = self._analyze_model_specificity(model_name, eval_result, basic_metric)
            if model_specificity:
                analysis['improvement_potential'].extend(model_specificity)
            
            individual_analysis[model_name] = analysis
        
        return individual_analysis
    
    def _analyze_model_specificity(self, model_name: str, eval_result: QualityMetrics, basic_metric: QualityMetrics) -> List[str]:
        """分析模型特异性，避免同质化分析"""
        specificity = []
        
        # 基于模型名称的特异性分析
        model_traits = {
            'claude_sonnet4': [
                f"逻辑推理能力在本次任务中评分为{eval_result.clarity_score:.1f}",
                f"适合复杂问题分析和深度思考"
            ],
            'gpt-41-0414-global': [
                f"数学和逻辑分析能力在本次表现为{eval_result.accuracy_score:.1f}分",
                f"技术问题处理能力较强"
            ],
            'qwen-max': [
                f"中文理解和处理能力在本次任务中表现为{eval_result.relevance_score:.1f}分",
                f"对中文语境和文化背景理解深入"
            ],
            'claude37_sonnet_new': [
                f"平衡性表现，本次综合评分{eval_result.overall_score:.1f}",
                f"适合日常对话和通用任务"
            ],
            'qwen-plus': [
                f"性价比较高，本次任务中效率表现良好",
                f"实用性强，适合多种场景"
            ],
            'gpt-41-mini-0414-global': [
                f"快速响应特性，适合轻量级任务",
                f"效率导向，资源节约型模型"
            ]
        }
        
        if model_name in model_traits:
            specificity.extend(model_traits[model_name][:1])  # 只取第一个特异性描述
        
        return specificity
    
    async def _deep_individualized_analysis(
        self,
        question: str,
        responses: List[Dict]
    ) -> Dict[str, Any]:
        """
        深度个性化分析 - 强制识别每个模型的独特特征
        即使评分相近，也要挖掘出风格、角度、深度的差异
        """
        if not responses or len(responses) < 2:
            return {}

        print("🔍 正在进行深度个性化分析...")

        try:
            # 构建强制差异化分析提示
            individualization_prompt = f"""
你是一位专业的内容分析专家。现有{len(responses)}个AI模型对同一问题给出了回答，即使它们质量相近，也必然存在风格、角度、深度的差异。

**问题：**
{question}

**各模型回答：**
"""
            for i, response in enumerate(responses, 1):
                model_name = response['model_name']
                answer_text = response['response']
                # 提供完整内容以便深度分析
                truncated = answer_text[:800] + "..." if len(answer_text) > 800 else answer_text
                individualization_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━
【模型{i}: {model_name}】
字符数: {len(answer_text)}
内容: {truncated}
━━━━━━━━━━━━━━━━━━━━━━

"""

            individualization_prompt += f"""
**分析任务（严格执行）：**

请为每个模型生成**完全不同**的个性化档案。禁止使用相同或相似的描述。

对每个模型，必须完成以下分析：

1. **内容风格特征**（必须具体且不同）
   - 引用该模型回答中的具体内容片段
   - 描述独特的表达方式、句式特点
   - 说明与其他模型的明显区别

2. **解答角度与深度**（必须差异化）
   - 该模型从什么独特角度切入问题
   - 侧重于理论/实践/示例/步骤中的哪些方面
   - 内容深度和广度的具体特征

3. **独特贡献点**（至少列出2-3个具体内容）
   - 引用该模型提供的独有信息、观点或方法
   - 说明这些内容在其他模型回答中缺失或弱化
   - 具体到可以定位的内容片段

4. **优势劣势对比**（必须基于实际内容）
   - 相比其他{len(responses)-1}个模型，该模型最突出的优势是什么
   - 相比其他{len(responses)-1}个模型，该模型最明显的不足是什么
   - 具体引用内容支撑判断

5. **适用场景推荐**（必须个性化）
   - 基于该模型的风格和内容特点
   - 推荐1-2个最适合使用该模型的具体场景
   - 解释为什么该模型在这些场景下表现更好

**输出要求：**

必须以JSON格式返回，确保每个模型的描述完全不同：

```json
{{
    "individualized_profiles": {{
        "模型1名称": {{
            "content_style": "具体的风格描述，引用实际内容片段",
            "approach_depth": "独特的角度和深度分析",
            "unique_contributions": [
                "具体贡献点1（引用内容）",
                "具体贡献点2（引用内容）",
                "具体贡献点3（引用内容）"
            ],
            "comparative_advantage": "相比其他模型的最大优势（具体说明）",
            "comparative_weakness": "相比其他模型的主要不足（具体说明）",
            "best_use_scenarios": [
                "最适场景1及理由",
                "最适场景2及理由"
            ],
            "signature_characteristics": "该模型的标志性特征总结（50字以内）"
        }},
        "模型2名称": {{
            ...（完全不同的分析）
        }},
        ...
    }},
    "differentiation_summary": "总结各模型之间的核心差异（100字以内）"
}}
```

**严格禁止：**
- ❌ 禁止对不同模型使用相同或高度相似的描述
- ❌ 禁止使用模糊、通用的评价（如"回答详尽"、"表达清晰"等）
- ❌ 禁止不引用具体内容就做出判断
- ❌ 禁止让所有模型都"优秀"或都"不足"

**必须执行：**
- ✅ 必须引用每个模型回答中的具体内容片段
- ✅ 必须明确指出各模型之间的具体差异
- ✅ 必须为每个模型找出至少2个独特贡献点
- ✅ 必须给出个性化的场景推荐，不能通用化
"""

            response = await call_llm_async(
                messages=[{"role": "user", "content": individualization_prompt}],
                model=self.evaluator_model,
                max_tokens=2500,  # 增加token数以支持详细分析
                temperature=0.4,   # 适当提高温度以增加多样性
                registry=self.registry,
                trace_id=self._current_trace_id,
                parent_observation_id=self._current_parent_observation_id,
                langfuse_metadata={
                    "component": "quality_analyzer",
                    "stage": "individualized_profiles"
                },
            )

            # 解析JSON结果
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    print(f"✅ 深度个性化分析完成，已识别{len(result.get('individualized_profiles', {}))}个模型的独特特征")
                    return result
                else:
                    # 尝试直接解析
                    result = json.loads(response)
                    return result
            except Exception as parse_error:
                print(f"⚠️ JSON解析失败: {str(parse_error)}, 使用回退方案")
                # 返回基础结构
                return {
                    "individualized_profiles": {},
                    "differentiation_summary": "深度分析解析失败"
                }

        except Exception as e:
            print(f"⚠️ 深度个性化分析失败: {str(e)}")
            return {}

    async def _analyze_response_approaches(
        self,
        question: str,
        responses: List[Dict]
    ) -> Dict[str, Any]:
        """分析各模型回答的方法和角度差异"""
        if not responses:
            return {}

        try:
            # 构建分析提示
            analysis_prompt = f"""
请分析以下各个AI模型对同一问题的回答方法和角度差异。

问题: {question}

各模型回答:
"""
            for i, response in enumerate(responses, 1):
                analysis_prompt += f"\n【模型{i}: {response['model_name']}】\n{response['response'][:300]}...\n"

            analysis_prompt += """
请从以下角度分析各模型的差异:
1. 解答方法 (理论分析、实例举证、步骤指导等)
2. 侧重点 (技术细节、实用建议、理论原理等)
3. 表达风格 (简洁直接、详细阐述、结构化等)
4. 独特观点 (各模型特有的见解或方法)

请以JSON格式返回分析结果:
{{
    "model_approaches": {{
        "模型名": {{"method": "方法", "focus": "侧重点", "style": "风格", "unique_insights": ["独特观点1", "独特观点2"]}}
    }},
    "differentiation_summary": "差异化总结"
}}
"""

            response = await call_llm_async(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=self.evaluator_model,
                max_tokens=1000,
                temperature=0.3,
                registry=self.registry,
                trace_id=self._current_trace_id,
                parent_observation_id=self._current_parent_observation_id,
                langfuse_metadata={
                    "component": "quality_analyzer",
                    "stage": "approach_analysis"
                },
            )

            # 尝试解析JSON
            import json
            try:
                result = json.loads(response)
                return result
            except:
                # 如果JSON解析失败，返回简化版本
                return {"model_approaches": {}, "differentiation_summary": "解析失败"}

        except Exception as e:
            print(f"⚠️ 角度分析失败: {str(e)}")
            return {}
    
    def _calculate_content_similarity(
        self, 
        responses: List[str], 
        fusion_answer: str
    ) -> Dict[str, Any]:
        """计算内容相似度矩阵"""
        similarity_matrix = {}
        
        # 简单的相似度计算（基于关键词重叠）
        def calculate_keyword_similarity(text1: str, text2: str) -> float:
            # 提取关键词
            words1 = set(re.findall(r'[\w\u4e00-\u9fff]+', text1.lower()))
            words2 = set(re.findall(r'[\w\u4e00-\u9fff]+', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        # 计算各回答间的相似度
        for i, response1 in enumerate(responses):
            for j, response2 in enumerate(responses):
                if i < j:
                    similarity = calculate_keyword_similarity(response1, response2)
                    similarity_matrix[f"model_{i+1}_vs_model_{j+1}"] = similarity
        
        # 计算与融合回答的相似度
        for i, response in enumerate(responses):
            similarity = calculate_keyword_similarity(response, fusion_answer)
            similarity_matrix[f"model_{i+1}_vs_fusion"] = similarity
        
        # 计算平均相似度
        if similarity_matrix:
            avg_similarity = sum(similarity_matrix.values()) / len(similarity_matrix)
            similarity_matrix['average_similarity'] = avg_similarity
        
        return similarity_matrix
    
    async def _extract_content_themes(
        self, 
        question: str,
        responses: List[Dict], 
        fusion_answer: str
    ) -> Dict[str, Any]:
        """提取内容主题和关键观点"""
        if not responses:
            return {}
        
        try:
            themes_prompt = f"""
请提取以下回答中的主要主题和关键观点。

问题: {question}

各模型回答:
"""
            for i, response in enumerate(responses, 1):
                themes_prompt += f"\n【模型{i}】\n{response['response'][:200]}...\n"
            
            themes_prompt += f"""
【融合回答】
{fusion_answer[:200]}...

请分析:
1. 各回答涵盖的主要主题
2. 每个模型的独特观点
3. 共同观点
4. 融合回答新增的内容

返回JSON格式:
{{
    "main_themes": ["主题1", "主题2"],
    "model_unique_points": {{"模型名": ["观点1", "观点2"]}},
    "common_points": ["共同观点1", "共同观点2"],
    "fusion_additions": ["融合新增内容1", "融合新增内容2"]
}}
"""
            
            response = await call_llm_async(
                messages=[{"role": "user", "content": themes_prompt}],
                model=self.evaluator_model,
                max_tokens=800,
                temperature=0.3,
                registry=self.registry,
                trace_id=self._current_trace_id,
                parent_observation_id=self._current_parent_observation_id,
                langfuse_metadata={
                    "component": "quality_analyzer",
                    "stage": "theme_extraction"
                },
            )
            
            try:
                import json
                result = json.loads(response)
                return result
            except:
                return {"main_themes": [], "model_unique_points": {}, "common_points": [], "fusion_additions": []}
                
        except Exception as e:
            print(f"⚠️ 主题提取失败: {str(e)}")
            return {}
    
    def _analyze_structure_patterns(
        self, 
        responses: List[Dict], 
        fusion_answer: str
    ) -> Dict[str, Any]:
        """分析回答的结构模式"""
        structure_analysis = {}
        
        def analyze_text_structure(text: str) -> Dict[str, Any]:
            # 分析文本结构特征
            lines = text.split('\n')
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # 检测列表项
            list_items = len(re.findall(r'^[•\-\*\d+\.\)]\s+', text, re.MULTILINE))
            
            # 检测标题或重点
            headers = len(re.findall(r'^[#]+\s+|^\*\*.*\*\*|^【.*】', text, re.MULTILINE))
            
            # 检测代码块
            code_blocks = len(re.findall(r'```|`[^`]+`', text))
            
            return {
                'paragraph_count': len(paragraphs),
                'list_items': list_items,
                'headers': headers,
                'code_blocks': code_blocks,
                'avg_paragraph_length': sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
                'has_structured_format': list_items > 0 or headers > 0
            }
        
        # 分析各模型的结构模式
        for response in responses:
            model_name = response['model_name']
            structure = analyze_text_structure(response['response'])
            structure_analysis[model_name] = structure
        
        # 分析融合回答的结构
        structure_analysis['fusion_answer'] = analyze_text_structure(fusion_answer)
        
        return structure_analysis
    
    def _calculate_content_uniqueness(
        self, 
        responses: List[Dict], 
        fusion_answer: str
    ) -> Dict[str, float]:
        """计算内容独特性评分"""
        uniqueness_scores = {}
        
        # 提取所有文本的关键词
        all_texts = [r['response'] for r in responses] + [fusion_answer]
        all_keywords = []
        
        for text in all_texts:
            keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', text.lower()))
            all_keywords.append(keywords)
        
        # 计算每个回答的独特性
        for i, response in enumerate(responses):
            model_keywords = all_keywords[i]
            other_keywords = set()
            
            # 收集其他回答的关键词
            for j, other_keywords_set in enumerate(all_keywords):
                if j != i:
                    other_keywords.update(other_keywords_set)
            
            # 计算独特关键词比例
            if model_keywords:
                unique_keywords = model_keywords - other_keywords
                uniqueness_score = len(unique_keywords) / len(model_keywords)
                uniqueness_scores[response['model_name']] = uniqueness_score
            else:
                uniqueness_scores[response['model_name']] = 0.0
        
        # 计算融合回答的独特性
        fusion_keywords = all_keywords[-1]
        model_keywords_union = set()
        for keywords in all_keywords[:-1]:
            model_keywords_union.update(keywords)
        
        if fusion_keywords:
            fusion_unique = fusion_keywords - model_keywords_union
            uniqueness_scores['fusion_answer'] = len(fusion_unique) / len(fusion_keywords)
        else:
            uniqueness_scores['fusion_answer'] = 0.0
        
        return uniqueness_scores
    
    def _perform_consistency_validation(
        self,
        llm_evaluations: Dict[str, QualityMetrics],
        comparison_analysis: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行逻辑一致性验证"""
        print("🔍 正在进行逻辑一致性验证...")
        
        consistency_issues = []
        corrections = {}
        validation_summary = {}
        
        # 1. 评分内部一致性检查
        score_consistency = self._check_score_consistency(llm_evaluations)
        consistency_issues.extend(score_consistency['issues'])
        corrections.update(score_consistency['corrections'])
        
        # 2. 排名逻辑一致性检查
        ranking_consistency = self._check_ranking_consistency(llm_evaluations, comparison_analysis)
        consistency_issues.extend(ranking_consistency['issues'])
        corrections.update(ranking_consistency['corrections'])
        
        # 3. 优势描述一致性检查
        strength_consistency = self._check_strength_consistency(
            llm_evaluations, comparison_analysis, content_analysis
        )
        consistency_issues.extend(strength_consistency['issues'])
        corrections.update(strength_consistency['corrections'])
        
        # 4. 内容分析与评分一致性检查
        content_score_consistency = self._check_content_score_consistency(
            llm_evaluations, content_analysis
        )
        consistency_issues.extend(content_score_consistency['issues'])
        corrections.update(content_score_consistency['corrections'])
        
        # 生成验证摘要
        validation_summary = {
            'total_issues': len(consistency_issues),
            'critical_issues': len([i for i in consistency_issues if i.get('severity', 'low') == 'critical']),
            'corrected_items': len(corrections),
            'consistency_score': max(0, 100 - len(consistency_issues) * 10)  # 100分制
        }
        
        return {
            'issues': consistency_issues,
            'corrections': corrections,
            'validation_summary': validation_summary
        }
    
    def _check_score_consistency(self, llm_evaluations: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """检查评分内部一致性"""
        issues = []
        corrections = {}
        
        for source, metrics in llm_evaluations.items():
            # 检查综合评分与各维度平均分的一致性
            dimension_scores = [
                metrics.completeness_score,
                metrics.accuracy_score,
                metrics.clarity_score,
                metrics.relevance_score
            ]
            avg_dimension_score = sum(dimension_scores) / len(dimension_scores)
            overall_score = metrics.overall_score
            
            # 如果差异超过1.5分，标记为不一致
            if abs(overall_score - avg_dimension_score) > 1.5:
                issues.append({
                    'type': 'score_inconsistency',
                    'source': source,
                    'severity': 'critical',
                    'description': f"{source}综合评分({overall_score:.1f})与维度平均分({avg_dimension_score:.1f})差异过大",
                    'suggestion': f"建议调整综合评分为{avg_dimension_score:.1f}"
                })
                corrections[f"{source}_overall_score"] = avg_dimension_score
            
            # 检查异常高分或低分
            for dim_name, score in zip(['completeness', 'accuracy', 'clarity', 'relevance'], dimension_scores):
                if score > 9.5:
                    issues.append({
                        'type': 'extreme_score',
                        'source': source,
                        'severity': 'warning',
                        'description': f"{source}的{dim_name}评分({score:.1f})过高，可能存在偏差"
                    })
                elif score < 1.0:
                    issues.append({
                        'type': 'extreme_score', 
                        'source': source,
                        'severity': 'warning',
                        'description': f"{source}的{dim_name}评分({score:.1f})过低，可能存在偏差"
                    })
        
        return {'issues': issues, 'corrections': corrections}
    
    def _check_ranking_consistency(
        self, 
        llm_evaluations: Dict[str, QualityMetrics],
        comparison_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查排名逻辑一致性"""
        issues = []
        corrections = {}
        
        # 按综合评分排序
        sorted_by_overall = sorted(
            llm_evaluations.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        # 检查是否所有模型都有相同的评分（同质化问题）
        scores = [metrics.overall_score for _, metrics in sorted_by_overall]
        if len(set(scores)) <= 2 and len(scores) > 2:
            issues.append({
                'type': 'homogeneous_scores',
                'severity': 'critical',
                'description': "检测到评分同质化问题，大多数模型评分相近",
                'suggestion': "需要重新评估模型差异性"
            })
        
        # 检查融合回答是否合理排名
        fusion_eval = llm_evaluations.get('fusion_answer')
        if fusion_eval:
            model_scores = [metrics.overall_score for name, metrics in llm_evaluations.items() if name != 'fusion_answer']
            if model_scores:
                max_model_score = max(model_scores)
                avg_model_score = sum(model_scores) / len(model_scores)
                
                # 如果融合回答评分比所有单模型都低很多
                if fusion_eval.overall_score < avg_model_score - 1.0:
                    issues.append({
                        'type': 'fusion_underperformance',
                        'severity': 'critical',
                        'description': f"融合回答评分({fusion_eval.overall_score:.1f})明显低于模型平均分({avg_model_score:.1f})",
                        'suggestion': "检查融合算法是否有效"
                    })
                
                # 如果融合回答评分异常高
                elif fusion_eval.overall_score > max_model_score + 1.0:
                    issues.append({
                        'type': 'fusion_overperformance',
                        'severity': 'warning',
                        'description': f"融合回答评分({fusion_eval.overall_score:.1f})异常高于最佳模型({max_model_score:.1f})",
                        'suggestion': "验证融合效果是否真实"
                    })
        
        return {'issues': issues, 'corrections': corrections}
    
    def _check_strength_consistency(
        self,
        llm_evaluations: Dict[str, QualityMetrics],
        comparison_analysis: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查优势描述一致性"""
        issues = []
        corrections = {}
        
        model_strengths = comparison_analysis.get('model_strengths', {})
        
        # 检查是否所有模型都有相同的优势描述
        if model_strengths:
            all_strengths = list(model_strengths.values())
            
            # 计算描述相似度
            similar_count = 0
            total_comparisons = 0
            
            for i in range(len(all_strengths)):
                for j in range(i + 1, len(all_strengths)):
                    total_comparisons += 1
                    # 简单的相似度检查（共同词汇）
                    str1 = ' '.join(all_strengths[i])
                    str2 = ' '.join(all_strengths[j])
                    common_words = len(set(str1.split()) & set(str2.split()))
                    if common_words > 3:  # 如果有超过3个共同词汇
                        similar_count += 1
            
            if total_comparisons > 0 and similar_count / total_comparisons > 0.7:
                issues.append({
                    'type': 'homogeneous_strengths',
                    'severity': 'critical',
                    'description': "检测到模型优势描述高度相似，缺乏差异化",
                    'suggestion': "需要基于实际内容重新分析各模型特色"
                })
        
        # 检查优势描述与评分的一致性
        for model_name, strengths in model_strengths.items():
            if model_name in llm_evaluations:
                metrics = llm_evaluations[model_name]
                
                # 如果描述说某个维度是优势，但评分不高
                if any("完整性" in s for s in strengths) and metrics.completeness_score < 7.0:
                    issues.append({
                        'type': 'strength_score_mismatch',
                        'source': model_name,
                        'severity': 'warning',
                        'description': f"{model_name}被描述为完整性优势，但评分仅{metrics.completeness_score:.1f}",
                        'suggestion': "调整优势描述或重新评分"
                    })
        
        return {'issues': issues, 'corrections': corrections}
    
    def _check_content_score_consistency(
        self,
        llm_evaluations: Dict[str, QualityMetrics],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查内容分析与评分一致性"""
        issues = []
        corrections = {}
        
        uniqueness_scores = content_analysis.get('content_uniqueness', {})
        
        # 检查独特性与评分的关系
        for model_name, uniqueness in uniqueness_scores.items():
            if model_name in llm_evaluations and model_name != 'fusion_answer':
                metrics = llm_evaluations[model_name]
                
                # 如果内容独特性很高但综合评分很低
                if uniqueness > 0.4 and metrics.overall_score < 6.0:
                    issues.append({
                        'type': 'uniqueness_score_mismatch',
                        'source': model_name,
                        'severity': 'warning',
                        'description': f"{model_name}内容独特性高({uniqueness:.1%})但综合评分低({metrics.overall_score:.1f})",
                        'suggestion': "检查评分标准是否过于严格或存在偏差"
                    })
                
                # 如果内容独特性很低但评分很高
                elif uniqueness < 0.1 and metrics.overall_score > 8.5:
                    issues.append({
                        'type': 'uniqueness_score_mismatch',
                        'source': model_name,
                        'severity': 'warning', 
                        'description': f"{model_name}内容独特性低({uniqueness:.1%})但综合评分高({metrics.overall_score:.1f})",
                        'suggestion': "验证高分是否合理或是否存在评分偏差"
                    })
        
        return {'issues': issues, 'corrections': corrections}
    
    def _calculate_validated_quality_ranking(
        self,
        basic_metrics: Dict[str, QualityMetrics],
        llm_evaluations: Dict[str, QualityMetrics],
        consistency_check: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """计算经过一致性校正的质量排名"""
        
        # 应用一致性校正
        corrected_evaluations = dict(llm_evaluations)
        corrections = consistency_check.get('corrections', {})
        
        for correction_key, correction_value in corrections.items():
            if '_overall_score' in correction_key:
                source_name = correction_key.replace('_overall_score', '')
                if source_name in corrected_evaluations:
                    # 创建新的评估指标，应用校正
                    original = corrected_evaluations[source_name]
                    corrected_evaluations[source_name] = QualityMetrics(
                        completeness_score=original.completeness_score,
                        accuracy_score=original.accuracy_score,
                        clarity_score=original.clarity_score,
                        relevance_score=original.relevance_score,
                        overall_score=correction_value,  # 应用校正值
                        word_count=original.word_count,
                        sentence_count=original.sentence_count,
                        readability_score=original.readability_score,
                        information_density=original.information_density
                    )
        
        # 使用校正后的评估生成排名
        return self._calculate_quality_ranking(basic_metrics, corrected_evaluations)

    def _analyze_fusion_effectiveness(
        self,
        llm_evaluations: Dict[str, QualityMetrics],
        comparison_analysis: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        融合效果量化分析 - 对比融合前后的质量提升

        返回详细的融合效果分析，包括：
        1. 各维度的提升量化
        2. 融合价值评估
        3. 内容整合效果
        4. 最佳实践建议
        """
        print("🔍 正在进行融合效果量化分析...")

        fusion_eval = llm_evaluations.get('fusion_answer')
        if not fusion_eval:
            return {
                'overall_effectiveness': 'N/A',
                'dimension_improvements': {},
                'fusion_value_score': 0,
                'recommendation': '无融合回答数据'
            }

        # 获取所有单模型的评分
        model_evaluations = {k: v for k, v in llm_evaluations.items() if k != 'fusion_answer'}

        if not model_evaluations:
            return {
                'overall_effectiveness': 'N/A',
                'dimension_improvements': {},
                'fusion_value_score': 0,
                'recommendation': '无模型回答数据'
            }

        # 1. 计算各维度的统计数据
        dimension_stats = self._calculate_dimension_statistics(model_evaluations, fusion_eval)

        # 2. 量化各维度的提升
        dimension_improvements = self._quantify_dimension_improvements(
            model_evaluations, fusion_eval, dimension_stats
        )

        # 3. 评估融合价值
        fusion_value = self._evaluate_fusion_value(
            dimension_improvements, dimension_stats, content_analysis
        )

        # 4. 内容整合效果分析
        integration_effectiveness = self._analyze_integration_effectiveness(
            model_evaluations, fusion_eval, content_analysis
        )

        # 5. 生成融合效果等级和建议
        effectiveness_level, recommendations = self._generate_effectiveness_assessment(
            dimension_improvements, fusion_value, integration_effectiveness
        )

        return {
            'overall_effectiveness': effectiveness_level,
            'dimension_stats': dimension_stats,
            'dimension_improvements': dimension_improvements,
            'fusion_value_score': fusion_value['total_score'],
            'fusion_value_breakdown': fusion_value,
            'integration_effectiveness': integration_effectiveness,
            'recommendations': recommendations,
            'summary': self._generate_fusion_summary(
                dimension_improvements, fusion_value, effectiveness_level
            )
        }

    def _calculate_dimension_statistics(
        self,
        model_evaluations: Dict[str, QualityMetrics],
        fusion_eval: QualityMetrics
    ) -> Dict[str, Any]:
        """计算各维度的统计数据"""
        dimensions = ['completeness', 'accuracy', 'clarity', 'relevance', 'overall']
        stats = {}

        for dim in dimensions:
            scores = [getattr(eval, f'{dim}_score') for eval in model_evaluations.values()]
            fusion_score = getattr(fusion_eval, f'{dim}_score')

            stats[dim] = {
                'model_avg': sum(scores) / len(scores),
                'model_max': max(scores),
                'model_min': min(scores),
                'model_std': self._calculate_std(scores),
                'fusion_score': fusion_score,
                'models_count': len(scores)
            }

        return stats

    def _calculate_std(self, scores: List[float]) -> float:
        """计算标准差"""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5

    def _quantify_dimension_improvements(
        self,
        model_evaluations: Dict[str, QualityMetrics],
        fusion_eval: QualityMetrics,
        dimension_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """量化各维度的提升"""
        improvements = {}
        dimensions = ['completeness', 'accuracy', 'clarity', 'relevance', 'overall']

        for dim in dimensions:
            stats = dimension_stats[dim]
            fusion_score = stats['fusion_score']
            model_avg = stats['model_avg']
            model_max = stats['model_max']

            # 计算提升量
            improvement_vs_avg = fusion_score - model_avg
            improvement_vs_max = fusion_score - model_max

            # 计算提升百分比
            improvement_pct_vs_avg = (improvement_vs_avg / model_avg * 100) if model_avg > 0 else 0
            improvement_pct_vs_max = (improvement_vs_max / model_max * 100) if model_max > 0 else 0

            # 判断提升显著性
            if improvement_vs_avg > 1.0:
                significance = "显著提升"
            elif improvement_vs_avg > 0.5:
                significance = "明显提升"
            elif improvement_vs_avg > 0.2:
                significance = "轻微提升"
            elif improvement_vs_avg > -0.2:
                significance = "基本持平"
            elif improvement_vs_avg > -0.5:
                significance = "轻微下降"
            else:
                significance = "明显下降"

            improvements[dim] = {
                'absolute_improvement_vs_avg': round(improvement_vs_avg, 2),
                'absolute_improvement_vs_max': round(improvement_vs_max, 2),
                'percentage_improvement_vs_avg': round(improvement_pct_vs_avg, 1),
                'percentage_improvement_vs_max': round(improvement_pct_vs_max, 1),
                'significance': significance,
                'fusion_score': fusion_score,
                'models_avg': model_avg,
                'models_max': model_max
            }

        return improvements

    def _evaluate_fusion_value(
        self,
        dimension_improvements: Dict[str, Any],
        dimension_stats: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估融合价值"""

        # 1. 质量提升价值（40分）
        quality_value = 0
        overall_imp = dimension_improvements['overall']['absolute_improvement_vs_avg']
        if overall_imp > 1.0:
            quality_value = 40
        elif overall_imp > 0.5:
            quality_value = 30
        elif overall_imp > 0.2:
            quality_value = 20
        elif overall_imp > 0:
            quality_value = 10

        # 2. 内容整合价值（30分）
        integration_value = 0
        content_themes = content_analysis.get('content_themes', {})
        fusion_additions = content_themes.get('fusion_additions', [])
        if len(fusion_additions) >= 3:
            integration_value = 30
        elif len(fusion_additions) >= 2:
            integration_value = 20
        elif len(fusion_additions) >= 1:
            integration_value = 10

        # 3. 一致性价值（15分）
        consistency_value = 15  # 默认满分
        # 如果融合回答比最佳单模型还低，扣分
        if dimension_improvements['overall']['absolute_improvement_vs_max'] < -0.5:
            consistency_value = 0
        elif dimension_improvements['overall']['absolute_improvement_vs_max'] < 0:
            consistency_value = 10

        # 4. 全面性价值（15分）
        comprehensiveness_value = 0
        positive_dims = sum(1 for dim in ['completeness', 'accuracy', 'clarity', 'relevance']
                           if dimension_improvements[dim]['absolute_improvement_vs_avg'] > 0)
        comprehensiveness_value = positive_dims * 3.75  # 每个维度3.75分

        total_score = quality_value + integration_value + consistency_value + comprehensiveness_value

        return {
            'total_score': round(total_score, 1),
            'quality_value': quality_value,
            'integration_value': integration_value,
            'consistency_value': consistency_value,
            'comprehensiveness_value': comprehensiveness_value,
            'max_score': 100,
            'level': self._get_value_level(total_score)
        }

    def _get_value_level(self, score: float) -> str:
        """获取价值等级"""
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        elif score >= 20:
            return "较差"
        else:
            return "很差"

    def _analyze_integration_effectiveness(
        self,
        model_evaluations: Dict[str, QualityMetrics],
        fusion_eval: QualityMetrics,
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析内容整合效果"""

        # 1. 内容覆盖度
        content_themes = content_analysis.get('content_themes', {})
        common_points = content_themes.get('common_points', [])
        fusion_additions = content_themes.get('fusion_additions', [])
        model_unique_points = content_themes.get('model_unique_points', {})

        # 计算总的独特观点数
        total_unique_points = sum(len(points) for points in model_unique_points.values())

        coverage_score = min(100, (len(common_points) + total_unique_points) * 10)

        # 2. 新增内容价值
        addition_value = len(fusion_additions) * 20  # 每个新增点20分
        addition_value = min(100, addition_value)

        # 3. 内容独特性
        uniqueness_scores = content_analysis.get('content_uniqueness', {})
        fusion_uniqueness = uniqueness_scores.get('fusion_answer', 0)

        # 4. 结构优化
        structure_patterns = content_analysis.get('structure_patterns', {})
        fusion_structure = structure_patterns.get('fusion_answer', {})
        model_structures = {k: v for k, v in structure_patterns.items() if k != 'fusion_answer'}

        # 计算结构改进
        if model_structures:
            avg_structure_score = sum(
                s.get('has_structured_format', False) for s in model_structures.values()
            ) / len(model_structures)
            fusion_has_structure = fusion_structure.get('has_structured_format', False)
            structure_improvement = fusion_has_structure and avg_structure_score < 0.5
        else:
            structure_improvement = False

        return {
            'coverage_score': round(coverage_score, 1),
            'addition_value': round(addition_value, 1),
            'fusion_uniqueness': round(fusion_uniqueness * 100, 1),
            'structure_improved': structure_improvement,
            'common_points_covered': len(common_points),
            'new_content_added': len(fusion_additions),
            'total_unique_perspectives': total_unique_points
        }

    def _generate_effectiveness_assessment(
        self,
        dimension_improvements: Dict[str, Any],
        fusion_value: Dict[str, Any],
        integration_effectiveness: Dict[str, Any]
    ) -> tuple:
        """生成融合效果等级和建议"""

        # 综合评估等级
        total_score = fusion_value['total_score']
        overall_imp = dimension_improvements['overall']['absolute_improvement_vs_avg']

        if total_score >= 80 and overall_imp > 0.5:
            level = "卓越"
            emoji = "🌟"
        elif total_score >= 60 and overall_imp > 0.2:
            level = "优秀"
            emoji = "⭐"
        elif total_score >= 40 and overall_imp >= 0:
            level = "良好"
            emoji = "✅"
        elif total_score >= 20:
            level = "一般"
            emoji = "⚠️"
        else:
            level = "需改进"
            emoji = "❌"

        effectiveness_level = f"{emoji} {level} ({total_score:.1f}/100)"

        # 生成建议
        recommendations = []

        # 基于各维度提升情况给出建议
        for dim, imp_data in dimension_improvements.items():
            if dim == 'overall':
                continue
            if imp_data['absolute_improvement_vs_avg'] < 0:
                dim_name_map = {
                    'completeness': '完整性',
                    'accuracy': '准确性',
                    'clarity': '清晰度',
                    'relevance': '相关性'
                }
                recommendations.append(
                    f"融合过程中{dim_name_map[dim]}有所下降，建议加强该维度的内容整合"
                )

        # 基于内容整合效果给出建议
        if integration_effectiveness['new_content_added'] == 0:
            recommendations.append("融合回答未添加新内容，建议在融合时增加综合性见解")

        if integration_effectiveness['fusion_uniqueness'] < 10:
            recommendations.append("融合回答独特性较低，建议增加独特的综合性分析")

        if not integration_effectiveness['structure_improved']:
            recommendations.append("可以进一步优化回答的结构组织，提高可读性")

        # 基于融合价值给出建议
        if fusion_value['quality_value'] < 20:
            recommendations.append("整体质量提升有限，建议优化融合算法或模型选择")

        if fusion_value['integration_value'] < 15:
            recommendations.append("内容整合价值较低，建议更好地融合各模型的优势观点")

        # 如果没有需要改进的地方，给出正面反馈
        if not recommendations:
            if total_score >= 80:
                recommendations.append("融合效果优秀，成功整合了多个模型的优势")
            elif total_score >= 60:
                recommendations.append("融合效果良好，在多个维度上实现了提升")
            else:
                recommendations.append("融合基本达到预期效果，可继续保持")

        return effectiveness_level, recommendations

    def _generate_fusion_summary(
        self,
        dimension_improvements: Dict[str, Any],
        fusion_value: Dict[str, Any],
        effectiveness_level: str
    ) -> str:
        """生成融合效果摘要"""

        overall_imp = dimension_improvements['overall']['absolute_improvement_vs_avg']
        overall_pct = dimension_improvements['overall']['percentage_improvement_vs_avg']

        # 找出提升最大的维度
        best_improved_dim = None
        best_improvement = -999
        for dim, imp_data in dimension_improvements.items():
            if dim != 'overall' and imp_data['absolute_improvement_vs_avg'] > best_improvement:
                best_improvement = imp_data['absolute_improvement_vs_avg']
                best_improved_dim = dim

        dim_name_map = {
            'completeness': '完整性',
            'accuracy': '准确性',
            'clarity': '清晰度',
            'relevance': '相关性'
        }

        summary_parts = []

        # 第一部分：整体效果
        if overall_imp > 0.5:
            summary_parts.append(
                f"融合效果{effectiveness_level}，综合质量相比模型平均提升{overall_imp:.1f}分({overall_pct:+.1f}%)"
            )
        elif overall_imp > 0:
            summary_parts.append(
                f"融合效果{effectiveness_level}，综合质量相比模型平均提升{overall_imp:.1f}分"
            )
        else:
            summary_parts.append(
                f"融合效果{effectiveness_level}，综合质量与模型平均基本持平"
            )

        # 第二部分：最佳提升维度
        if best_improved_dim and best_improvement > 0.2:
            summary_parts.append(
                f"{dim_name_map[best_improved_dim]}提升最为显著({best_improvement:+.1f}分)"
            )

        # 第三部分：价值评估
        value_score = fusion_value['total_score']
        value_level = fusion_value['level']
        summary_parts.append(
            f"融合价值{value_level}({value_score:.1f}/100分)"
        )

        return "；".join(summary_parts) + "。"

    def _analyze_speed_quality_tradeoff(
        self,
        llm_responses: List[Dict],
        llm_evaluations: Dict[str, QualityMetrics]
    ) -> Dict[str, Any]:
        """
        速度-质量权衡分析

        分析各模型的响应时间与质量之间的关系，识别：
        1. 速度最快的模型
        2. 质量最高的模型
        3. 性价比最优的模型（综合速度和质量）
        4. 场景化推荐

        Args:
            llm_responses: 各模型的响应数据（包含response_time）
            llm_evaluations: 各模型的质量评估结果

        Returns:
            速度-质量权衡分析结果
        """

        # 1. 提取成功响应的模型数据
        model_data = []
        for response in llm_responses:
            if response.get('success') and response['model_name'] in llm_evaluations:
                model_name = response['model_name']
                response_time = response.get('response_time', 0)
                quality_metrics = llm_evaluations[model_name]

                model_data.append({
                    'model_name': model_name,
                    'response_time': response_time,
                    'quality_score': quality_metrics.overall_score,
                    'completeness': quality_metrics.completeness_score,
                    'accuracy': quality_metrics.accuracy_score,
                    'clarity': quality_metrics.clarity_score,
                    'relevance': quality_metrics.relevance_score
                })

        if not model_data:
            return {
                'available': False,
                'message': '没有足够的数据进行速度-质量权衡分析'
            }

        # 2. 计算效率指标（质量/时间）
        for model in model_data:
            if model['response_time'] > 0:
                # 效率得分 = 质量分数 / 响应时间（秒）
                model['efficiency_score'] = model['quality_score'] / model['response_time']
            else:
                model['efficiency_score'] = 0

        # 3. 识别各类最佳模型
        fastest_model = min(model_data, key=lambda x: x['response_time'])
        highest_quality_model = max(model_data, key=lambda x: x['quality_score'])
        most_efficient_model = max(model_data, key=lambda x: x['efficiency_score'])

        # 4. 计算统计数据
        avg_response_time = sum(m['response_time'] for m in model_data) / len(model_data)
        avg_quality_score = sum(m['quality_score'] for m in model_data) / len(model_data)
        avg_efficiency = sum(m['efficiency_score'] for m in model_data) / len(model_data)

        # 5. 模型分类（快速型、质量型、平衡型）
        model_categories = self._categorize_models(
            model_data, avg_response_time, avg_quality_score, avg_efficiency
        )

        # 6. 相关性分析（速度与质量的关系）
        correlation_analysis = self._analyze_speed_quality_correlation(model_data)

        # 7. 场景化推荐
        scenario_recommendations = self._generate_scenario_recommendations(
            fastest_model, highest_quality_model, most_efficient_model, model_categories
        )

        # 8. 权衡评估
        tradeoff_assessment = self._assess_tradeoffs(
            fastest_model, highest_quality_model, most_efficient_model,
            avg_response_time, avg_quality_score
        )

        return {
            'available': True,
            'fastest_model': {
                'name': fastest_model['model_name'],
                'response_time': fastest_model['response_time'],
                'quality_score': fastest_model['quality_score'],
                'efficiency_score': fastest_model['efficiency_score']
            },
            'highest_quality_model': {
                'name': highest_quality_model['model_name'],
                'response_time': highest_quality_model['response_time'],
                'quality_score': highest_quality_model['quality_score'],
                'efficiency_score': highest_quality_model['efficiency_score']
            },
            'most_efficient_model': {
                'name': most_efficient_model['model_name'],
                'response_time': most_efficient_model['response_time'],
                'quality_score': most_efficient_model['quality_score'],
                'efficiency_score': most_efficient_model['efficiency_score']
            },
            'statistics': {
                'avg_response_time': avg_response_time,
                'avg_quality_score': avg_quality_score,
                'avg_efficiency': avg_efficiency,
                'speed_range': {
                    'min': fastest_model['response_time'],
                    'max': max(m['response_time'] for m in model_data)
                },
                'quality_range': {
                    'min': min(m['quality_score'] for m in model_data),
                    'max': highest_quality_model['quality_score']
                }
            },
            'model_categories': model_categories,
            'correlation_analysis': correlation_analysis,
            'scenario_recommendations': scenario_recommendations,
            'tradeoff_assessment': tradeoff_assessment,
            'all_models_data': model_data  # 完整数据供进一步分析
        }

    def _categorize_models(
        self,
        model_data: List[Dict],
        avg_time: float,
        avg_quality: float,
        avg_efficiency: float
    ) -> Dict[str, List[str]]:
        """将模型分类为快速型、质量型、平衡型"""

        fast_models = []
        quality_models = []
        balanced_models = []

        for model in model_data:
            time = model['response_time']
            quality = model['quality_score']
            efficiency = model['efficiency_score']

            # 快速型：响应时间显著低于平均值
            is_fast = time < avg_time * 0.8
            # 质量型：质量显著高于平均值
            is_high_quality = quality > avg_quality * 1.1
            # 平衡型：效率高于平均值
            is_balanced = efficiency > avg_efficiency

            if is_fast and not is_high_quality:
                fast_models.append(model['model_name'])
            elif is_high_quality and not is_fast:
                quality_models.append(model['model_name'])
            elif is_balanced or (is_fast and is_high_quality):
                balanced_models.append(model['model_name'])
            else:
                # 默认归类到平衡型
                balanced_models.append(model['model_name'])

        return {
            'fast_models': fast_models,
            'quality_models': quality_models,
            'balanced_models': balanced_models
        }

    def _analyze_speed_quality_correlation(self, model_data: List[Dict]) -> Dict[str, Any]:
        """分析速度与质量的相关性"""

        if len(model_data) < 2:
            return {
                'correlation_type': 'insufficient_data',
                'description': '数据不足，无法分析相关性'
            }

        # 简单的相关性分析
        times = [m['response_time'] for m in model_data]
        qualities = [m['quality_score'] for m in model_data]

        # 计算排序一致性（快的是否质量也高）
        time_ranks = self._get_ranks(times, reverse=True)  # 时间越短排名越高
        quality_ranks = self._get_ranks(qualities, reverse=False)  # 质量越高排名越高

        # 计算rank相关性（简化版Spearman相关）
        rank_diffs = sum(abs(time_ranks[i] - quality_ranks[i]) for i in range(len(model_data)))
        max_diff = len(model_data) * (len(model_data) - 1)
        similarity = 1 - (rank_diffs / max_diff) if max_diff > 0 else 0

        if similarity > 0.6:
            correlation_type = 'positive'
            description = '速度快的模型往往质量也较高（正相关）'
        elif similarity < 0.4:
            correlation_type = 'negative'
            description = '速度快的模型质量相对较低（负相关/权衡关系）'
        else:
            correlation_type = 'neutral'
            description = '速度与质量之间无明显相关性'

        return {
            'correlation_type': correlation_type,
            'similarity_score': similarity,
            'description': description
        }

    def _get_ranks(self, values: List[float], reverse: bool = False) -> List[int]:
        """获取数值的排名"""
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
        ranks = [0] * len(values)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        return ranks

    def _generate_scenario_recommendations(
        self,
        fastest: Dict,
        highest_quality: Dict,
        most_efficient: Dict,
        categories: Dict
    ) -> Dict[str, str]:
        """生成场景化推荐"""

        recommendations = {}

        # 时间敏感场景
        recommendations['time_critical'] = (
            f"推荐使用 {fastest['model_name']} "
            f"(响应时间: {fastest['response_time']:.2f}秒, "
            f"质量: {fastest['quality_score']:.1f}/10)"
        )

        # 质量优先场景
        recommendations['quality_critical'] = (
            f"推荐使用 {highest_quality['model_name']} "
            f"(质量: {highest_quality['quality_score']:.1f}/10, "
            f"响应时间: {highest_quality['response_time']:.2f}秒)"
        )

        # 综合场景（性价比）
        recommendations['balanced'] = (
            f"推荐使用 {most_efficient['model_name']} "
            f"(效率得分: {most_efficient['efficiency_score']:.2f}, "
            f"质量: {most_efficient['quality_score']:.1f}/10, "
            f"响应时间: {most_efficient['response_time']:.2f}秒)"
        )

        # 生产环境推荐
        if categories['balanced_models']:
            recommendations['production'] = (
                f"生产环境推荐平衡型模型: {', '.join(categories['balanced_models'])}"
            )
        else:
            recommendations['production'] = (
                f"生产环境可考虑效率最优的 {most_efficient['model_name']}"
            )

        return recommendations

    def _assess_tradeoffs(
        self,
        fastest: Dict,
        highest_quality: Dict,
        most_efficient: Dict,
        avg_time: float,
        avg_quality: float
    ) -> Dict[str, Any]:
        """评估权衡关系"""

        assessments = []

        # 1. 最快模型的质量损失
        if fastest['model_name'] != highest_quality['model_name']:
            quality_loss = highest_quality['quality_score'] - fastest['quality_score']
            time_gain = highest_quality['response_time'] - fastest['response_time']

            if quality_loss > 1.0:
                assessments.append({
                    'type': 'speed_over_quality',
                    'message': f"选择最快模型 {fastest['model_name']} 可节省 {time_gain:.2f}秒，但质量降低 {quality_loss:.1f}分",
                    'severity': 'high' if quality_loss > 2.0 else 'medium'
                })
            else:
                assessments.append({
                    'type': 'speed_benefit',
                    'message': f"选择最快模型 {fastest['model_name']} 可节省 {time_gain:.2f}秒，质量损失较小({quality_loss:.1f}分)",
                    'severity': 'low'
                })

        # 2. 最高质量模型的时间代价
        if highest_quality['model_name'] != fastest['model_name']:
            time_cost = highest_quality['response_time'] - fastest['response_time']
            quality_gain = highest_quality['quality_score'] - fastest['quality_score']

            if time_cost > 3.0:
                assessments.append({
                    'type': 'quality_over_speed',
                    'message': f"选择最高质量模型 {highest_quality['model_name']} 需额外等待 {time_cost:.2f}秒，质量提升 {quality_gain:.1f}分",
                    'severity': 'high' if time_cost > 5.0 else 'medium'
                })
            else:
                assessments.append({
                    'type': 'quality_benefit',
                    'message': f"选择最高质量模型 {highest_quality['model_name']} 仅需额外 {time_cost:.2f}秒，质量提升 {quality_gain:.1f}分",
                    'severity': 'low'
                })

        # 3. 最优效率模型推荐
        if most_efficient['model_name'] not in [fastest['model_name'], highest_quality['model_name']]:
            assessments.append({
                'type': 'efficiency_recommendation',
                'message': f"{most_efficient['model_name']} 提供最佳的速度-质量平衡（效率: {most_efficient['efficiency_score']:.2f}）",
                'severity': 'info'
            })

        # 4. 整体权衡评估
        if avg_time < 2.0:
            overall_assessment = "所有模型响应都很快，可优先考虑质量"
        elif avg_quality < 7.0:
            overall_assessment = "整体质量有待提升，建议优先选择高质量模型"
        else:
            overall_assessment = "速度和质量都表现良好，可根据具体场景灵活选择"

        return {
            'individual_tradeoffs': assessments,
            'overall_assessment': overall_assessment
        }
