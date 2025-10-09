# AI Fusion - 智能多模型融合系统

<div align="center">

🤖 多提供商、多模型并发与质量分析的智能融合系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📖 项目简介

AI Fusion 是一个可将多个 LLM 提供商与模型组合在一起、并发调用并进行智能融合与质量分析的系统：
- 🎯 智能模型选择：基于问题自动挑选最合适的 3 个模型
- ⚡ 并发调用执行：异步并行请求多个 LLM，显著缩短等待时间
- 🧠 智能回答融合：综合多模型优点，生成更高质量的最终回答
- 📊 质量分析与报告：两阶段评分、差异化分析、一致性校验、生成 Markdown 报告

核心由节点化的异步流程实现，包含：模型选择 → 并发调用 → 回答融合 → 质量分析 → 报告输出。

## ✨ 核心特性

- **多提供商支持**：OpenAI、Anthropic、通义千问（Qwen），以及任意 OpenAI 兼容服务（通过 `UniversalProvider`）
- **智能选择器**：`ai_fusion/analysis/smart_selector.py` 使用 LLM 对问题进行分析并选择互补模型
- **并发执行**：`ai_fusion/core/nodes.py` 中 `ParallelLLMNode` 异步并发调用多个模型
- **融合与评估**：`FusionAgentNode` 融合回答；`AIFusionQualityAnalyzer` 两阶段评分与一致性分析
- **报告生成**：`ai_fusion/reporting/reporter.py` 输出结构化 Markdown 报告到 `reports/`

## 🚀 快速开始

### 1) 安装依赖

```bash
git clone https://github.com/Ren1104/fusion-agent.git
cd fusion-agent
pip install -r requirements.txt
```

### 2) 配置环境变量（至少配置一个提供商）

在项目根目录创建 `.env`（可参考 `docs/CONFIG.md`）：

```env
# 通用（OpenAI 兼容服务）
OPENAI_API_KEY=your-openai-or-compatible-key
OPENAI_BASE_URL=https://api.openai.com/v1  # 使用第三方兼容服务时改成其端点

# OpenAI（可选）
OPENAI_OFFICIAL_API_KEY=your-open-ai=key

# Anthropic（可选）
ANTHROPIC_API_KEY=your-anthropic-key

# 通义千问 Qwen（可选）
QWEN_API_KEY=your-qwen-key
QWEN_BASE_URL=https://dashscope.aliyun.com/compatible-mode/v1  # 如使用兼容端点
```

更多可用配置（并发、重试、报告目录等），请见 `docs/CONFIG.md`。

### 3) 运行交互式程序

```bash
python run.py
```

启动后根据提示输入问题。系统会自动完成模型选择、并发调用、融合与质量分析，并在 `reports/` 输出报告。

### 4) 以编程方式调用

```python
import asyncio
from ai_fusion import AIFusionFlow

async def main():
    fusion = AIFusionFlow()
    answer = await fusion.process_question("解释一下深度学习中的反向传播算法")
    print(answer)

asyncio.run(main())
```

### 5) 示例与诊断

- 演示程序：`python examples/demo.py`
- 多提供商/新架构最小验证：`python examples/test_new_architecture.py`

## 🧩 架构与主要模块

```
用户问题
   ↓
ModelSelectorNode (智能模型选择)
   ↓
ParallelLLMNode (并发调用)
   ↓
FusionAgentNode (回答融合)
   ↓
AIFusionQualityAnalyzer (质量分析)
   ↓
Reporter (报告生成)
```

- `ai_fusion/core/main.py`：`AIFusionFlow` 主流程编排（连接节点、运行流程、触发报告）
- `ai_fusion/core/nodes.py`：选择、并发、融合三个核心节点
- `ai_fusion/analysis/quality_analyzer.py`：两阶段评分、差异化与一致性分析
- `ai_fusion/analysis/smart_selector.py`：LLM 驱动的模型选择
- `ai_fusion/registry/model_registry.py`：多提供商与模型发现/调用的统一入口
- `ai_fusion/providers/*`：OpenAI、Anthropic、Universal（OpenAI 兼容）等提供商实现
- `ai_fusion/reporting/reporter.py`：Markdown 报告输出到 `reports/`

## 🔌 提供商与模型

系统通过 `ModelRegistry` 自动发现可用的提供商与模型：
- `UniversalProvider`：使用 `OPENAI_API_KEY` + `OPENAI_BASE_URL` 适配任意 OpenAI 兼容服务，内置一组常用模型知识（如 `qwen-max`、`gpt-41-0414-global`、`claude_sonnet4` 等标识）
- `OpenAIProvider`：官方 OpenAI（如 `gpt-4-turbo`、`gpt-4o-mini`）
- `AnthropicProvider`：Claude（如 `claude-3.5-sonnet`、`claude-3-haiku`）
- `AlibabaProvider`: Qwen(如 `qwen-max`)
提示：具体可用模型依赖您配置的服务与账号权限；也可通过环境变量 `AVAILABLE_MODELS` 指定模型 ID 列表给 `UniversalProvider`。

## 🔧 配置与文档

- 完整配置与常见问题：`docs/CONFIG.md`
- 质量评估标准：`docs/evaluation_standards.md`

常用环境变量（节选）：
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`：第三方兼容 API key 与 url
- `OPENAI_OFFICIAL_API_KEY`: OpenAI 官方key
- `ANTHROPIC_API_KEY`：Anthropic Claude
- `QWEN_API_KEY`, `QWEN_BASE_URL`：通义千问（兼容端点时可配）

## 📂 项目结构

```
ai-fustion-agent/
├── run.py                         # 项目入口（交互式）
├── main.py                        # IDE 默认示例
├── ai_fusion/
│   ├── core/
│   │   ├── main.py               # AIFusionFlow 主流程
│   │   └── nodes.py              # 选择/并发/融合 节点
│   ├── analysis/
│   │   ├── smart_selector.py     # 智能模型选择器
│   │   └── quality_analyzer.py   # 质量分析器
│   ├── providers/                # 各提供商实现（OpenAI/Anthropic/Universal/...）
│   ├── registry/model_registry.py# 模型注册/统一调用
│   ├── reporting/reporter.py     # 报告生成
│   └── utils/helpers.py          # 工具与封装
├── examples/
│   ├── demo.py
│   └── test_new_architecture.py
├── docs/
│   ├── CONFIG.md
│   └── evaluation_standards.md
├── reports/                       # 运行后自动生成
└── requirements.txt
```

## 🧪 质量分析与报告

质量分析采用“两阶段评分 + 一致性校验”的策略：
- 阶段一：对比评分，强制拉开差距，避免同质化
- 阶段二：维度化细评（完整性/准确性/清晰度/相关性），并与对比分校准
- 一致性校验：检查评分一致性、排名逻辑、优势描述与内容差异的一致性

运行完成后会在 `reports/` 下生成 Markdown 报告，包含模型表现对比、融合优势、改进建议等。

## 📈 性能与实践建议

- 并发能显著缩短总时长（3 个模型各 3 秒 → 总约 3 秒）
- 生产环境建议：配置多个提供商，启用报告，按需开启深度个性化分析
- 成本优化：选择轻量模型、降低并发、缩短最大输出、视需要关闭深度分析