# AI Fusion

多模型智能融合系统，基于 PocketFlow 构建。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件并配置 API Key（至少配置一个）：

```bash
# OpenAI 兼容服务（推荐用于第三方服务）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.example.com/v1

# 或者 Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key_here

# 可选：指定使用的模型列表（逗号分隔）
AVAILABLE_MODELS=qwen-max,claude_sonnet4,gpt-41-0414-global

# （可选）Langfuse 监控
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. 启动应用

#### 方式 1：命令行交互

```bash
python main.py
```

启动后输入问题，系统会自动选择 3 个最适合的模型进行回答融合。

#### 方式 2：FastAPI Web 服务

```bash
python app.py
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

**API 调用示例：**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能？"}'
```

## 项目结构

```
ai-fusion-agent/
├── main.py         # 命令行入口
├── app.py          # FastAPI 服务入口
├── flow.py         # PocketFlow 流程编排
├── nodes.py        # 5 个工作流节点
├── providers.py    # 多服务商 Provider 架构
├── analyzer.py     # 智能模型选择 + 质量分析
├── reporter.py     # Markdown 报告生成
└── reports/        # 报告输出目录
```

## 工作流程

1. **智能选择** - 根据问题类型自动选择 3 个最合适的模型
2. **并发调用** - 同时调用选中的模型获取回答
3. **融合回答** - 将多个模型的回答智能融合成最终答案
4. **质量分析** - 评估各模型回答质量并生成报告
5. **报告生成** - 输出详细的 Markdown 分析报告

## Langfuse 追踪

项目内置 Langfuse 监控能力，可对每次工作流运行进行全链路追踪：

- 自动创建根 Trace，覆盖所有工作流节点
- 每个节点在 Langfuse 中都有独立 Span，失败会标记为 `ERROR`
- 每次大模型调用以 Generation 的形式记录输入、输出与 Token 统计
- 质量分析阶段的多轮评估调用同样纳入追踪，方便定位问题

启用步骤：

1. 在 `.env` 中填入 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`
2. 确保运行环境可以访问 Langfuse 服务地址及所需的 LLM API
3. 正常运行 `python main.py` 或 Web 服务，完成后到 Langfuse 控制台查看对应 Trace

> 提醒：在受限网络或沙箱环境中运行时，如无法访问 Langfuse，会看到相关 `ProxyError` 提示，追踪数据不会上传。

## 技术栈

- **PocketFlow** - 异步工作流编排框架
- **FastAPI** - Web API 框架
- **OpenAI/Anthropic SDK** - LLM 服务调用

## License

MIT
