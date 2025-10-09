# AI Fusion 配置指南

本文档详细说明了 AI Fusion 系统的所有配置选项和环境变量。

## 目录

- [快速开始](#快速开始)
- [API密钥配置](#api密钥配置)
- [模型配置](#模型配置)
- [系统配置](#系统配置)
- [质量分析配置](#质量分析配置)
- [性能优化配置](#性能优化配置)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 创建配置文件

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用您喜欢的编辑器
```

### 2. 最小化配置

AI Fusion 至少需要配置一个 LLM 提供商的 API 密钥才能运行：

```env
# 最简配置 - 仅使用 OpenAI
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. 推荐配置

为获得最佳效果，建议配置多个 LLM 提供商：

```env
# OpenAI (GPT-4 等)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic (Claude)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# 阿里云通义千问
DASHSCOPE_API_KEY=your-dashscope-api-key-here
```

---

## API密钥配置

### OpenAI 配置

**GPT 系列模型**（GPT-4, GPT-4 Turbo, GPT-3.5 等）

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

**获取方式：**
1. 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
2. 登录/注册账号
3. 创建新的 API 密钥
4. 复制密钥到配置文件

**注意事项：**
- API 密钥以 `sk-` 开头
- 妥善保管密钥，不要泄露或提交到 Git
- 建议设置使用限额避免意外费用

### Anthropic Claude 配置

**Claude 系列模型**（Claude 3.5 Sonnet, Claude 3 Opus 等）

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

**获取方式：**
1. 访问 [Anthropic Console](https://console.anthropic.com/account/keys)
2. 登录/注册账号
3. 创建 API 密钥
4. 复制密钥到配置文件

**注意事项：**
- Claude API 密钥以 `sk-ant-` 开头
- Claude 模型在长文本理解和代码分析方面表现优异

### 阿里云通义千问配置

**Qwen 系列模型**（Qwen-Max, Qwen-Plus 等）

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx
```

**获取方式：**
1. 访问 [阿里云百炼平台](https://dashscope.aliyun.com/)
2. 登录阿里云账号
3. 开通 DashScope 服务
4. 创建 API-KEY
5. 复制密钥到配置文件

**注意事项：**
- 需要实名认证的阿里云账号
- 新用户通常有免费额度
- 适合中文场景

### 自定义 LLM 提供商

如果使用兼容 OpenAI API 格式的其他服务：

```env
CUSTOM_LLM_API_KEY=your-custom-api-key
CUSTOM_LLM_BASE_URL=https://your-endpoint.com/v1
```

支持的兼容服务包括：
- Azure OpenAI
- LocalAI
- FastChat
- 其他 OpenAI 兼容服务

---

## 模型配置

### 可用模型列表

AI Fusion 支持的模型在 `ai_fusion_utils.py` 中定义。当前支持的模型：

**OpenAI 系列：**
- `gpt-41-0414-global` - GPT-4 Turbo (推荐)
- `gpt-41-mini-0414-global` - GPT-4 Mini (快速)
- `gpt-35-turbo-0125-global` - GPT-3.5 Turbo

**Anthropic Claude 系列：**
- `claude_sonnet4` - Claude 3.5 Sonnet (最新，推荐)
- `claude37_sonnet_new` - Claude 3 Sonnet
- `claude3_opus` - Claude 3 Opus (最强大)

**阿里云通义千问系列：**
- `qwen-max` - Qwen Max (最强版本)
- `qwen-plus` - Qwen Plus (平衡版本)
- `qwen-turbo` - Qwen Turbo (快速版本)

### 添加自定义模型

编辑 `ai_fusion_utils.py` 中的 `get_available_models()` 函数：

```python
def get_available_models() -> List[ModelConfig]:
    """获取所有可用的模型配置"""
    load_dotenv()

    models = []

    # 添加您的自定义模型
    if os.getenv("YOUR_API_KEY"):
        models.append(ModelConfig(
            name="your-model-name",
            api_key=os.getenv("YOUR_API_KEY"),
            base_url=os.getenv("YOUR_BASE_URL", "https://api.example.com/v1")
        ))

    return models
```

### 模型选择策略

AI Fusion 使用智能选择器根据问题类型自动选择最合适的模型组合。您可以配置：

```env
# 默认选择的模型数量
DEFAULT_MODEL_COUNT=3

# 是否使用智能模型选择
USE_SMART_SELECTOR=true

# 智能选择器使用的分析模型
SELECTOR_MODEL=claude_sonnet4
```

---

## 系统配置

### 重试和超时设置

```env
# 模型调用最大重试次数
MAX_RETRIES=3

# 模型调用超时时间（秒）
REQUEST_TIMEOUT=60

# 并发调用最大并发数
MAX_CONCURRENCY=5
```

**说明：**
- `MAX_RETRIES`: API 调用失败后的重试次数（推荐 2-5）
- `REQUEST_TIMEOUT`: 单次 API 调用的超时时间（推荐 30-120 秒）
- `MAX_CONCURRENCY`: 同时进行的 API 调用数量（推荐 3-10）

### 日志配置

```env
# 是否启用详细日志
VERBOSE_LOGGING=true

# 日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_LEVEL=INFO

# 日志文件路径（留空则仅输出到控制台）
LOG_FILE_PATH=./logs/ai_fusion.log
```

**日志级别说明：**
- `DEBUG`: 显示所有调试信息（适合开发）
- `INFO`: 显示一般信息（默认，推荐）
- `WARNING`: 仅显示警告和错误
- `ERROR`: 仅显示错误信息

---

## 质量分析配置

### 基本设置

```env
# 融合回答的最大长度（字符数）
MAX_FUSION_LENGTH=3000

# 是否启用深度个性化分析
ENABLE_DEEP_ANALYSIS=true

# 是否生成详细报告
GENERATE_DETAILED_REPORT=true

# 报告保存路径
REPORT_OUTPUT_DIR=./reports
```

### 深度个性化分析

当启用深度个性化分析时，系统会：
1. 使用 LLM 对每个模型的回答进行深度分析
2. 识别每个模型的独特贡献和风格差异
3. 生成个性化的模型档案
4. 提供场景化使用建议

**性能影响：**
- 额外调用 1 次 LLM API（使用 `SELECTOR_MODEL` 配置的模型）
- 增加 5-10 秒处理时间
- 报告质量显著提升

**建议：**
- 生产环境：启用（`true`）
- 开发调试：可禁用（`false`）以提高速度
- 演示场景：强烈建议启用

### 报告生成

```env
# 是否生成详细报告
GENERATE_DETAILED_REPORT=true

# 报告保存路径
REPORT_OUTPUT_DIR=./reports
```

报告包含：
- 问题分析和分类
- 模型选择理由
- 每个模型的回答和评分
- 质量分析和对比
- 融合策略说明
- 个性化模型档案（如启用深度分析）

报告格式：Markdown (`.md`)

---

## 性能优化配置

### 响应缓存

```env
# 是否启用响应缓存
ENABLE_RESPONSE_CACHE=false

# 缓存过期时间（秒）
CACHE_EXPIRATION=3600
```

**说明：**
- 缓存可以显著提高重复问题的响应速度
- 适用于测试和演示场景
- 生产环境建议禁用或设置较短过期时间

### 速率限制

```env
# API调用频率限制（每分钟最大请求数）
RATE_LIMIT_PER_MINUTE=60
```

**说明：**
- 防止超出 API 提供商的速率限制
- 根据您的 API 套餐调整此值
- OpenAI Tier 1: 推荐设置 50-100
- 免费套餐：推荐设置 10-20

### 内容安全

```env
# 是否启用内容安全检查
ENABLE_CONTENT_SAFETY=true
```

**说明：**
- 对用户输入和模型输出进行基本的安全检查
- 过滤敏感内容和恶意输入
- 建议始终启用

---

## 常见问题

### Q1: 如何只使用一个 LLM 提供商？

**A:** 只需配置该提供商的 API 密钥即可。系统会自动检测可用的模型。

```env
# 示例：仅使用 OpenAI
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

系统会从可用的 OpenAI 模型中选择 3 个进行融合。

### Q2: 配置文件在哪里？

**A:** 配置文件应该是项目根目录下的 `.env` 文件：

```
/Users/max/PyProject/.env
```

### Q3: 如何验证配置是否正确？

**A:** 运行以下命令检查：

```bash
# 运行主程序
python ai_fusion_main.py

# 如果配置正确，会看到：
# ✅ 环境验证成功
# 🚀 启动 AI Fusion 系统...
```

### Q4: API 密钥无效怎么办？

**A:** 检查以下事项：
1. 确认密钥拼写正确，无多余空格
2. 确认密钥未过期或被撤销
3. 确认账户有足够的余额（如适用）
4. 确认 BASE_URL 正确（特别是使用代理时）

### Q5: 如何切换不同的融合模型？

**A:** 修改 `ai_fusion_nodes.py` 中 `FusionAgentNode` 的 `exec_async` 方法：

```python
# 当前默认使用 claude_sonnet4
fused_answer = await call_llm_async(
    messages=[{"role": "user", "content": fusion_prompt}],
    model="claude_sonnet4"  # 改为您想要的模型
)
```

推荐的融合模型：
- `claude_sonnet4` - 融合质量最佳（默认）
- `gpt-41-0414-global` - 速度快，成本低
- `qwen-max` - 中文场景优秀

### Q6: 报告保存在哪里？

**A:** 默认保存在项目根目录的 `reports` 文件夹：

```
/Users/max/PyProject/reports/
```

可以通过 `REPORT_OUTPUT_DIR` 环境变量修改。

### Q7: 如何减少 API 费用？

**A:** 几种方法：
1. 使用更经济的模型组合（如 GPT-3.5, Qwen-Plus）
2. 禁用深度个性化分析：`ENABLE_DEEP_ANALYSIS=false`
3. 启用响应缓存（测试环境）
4. 减少并发调用数：`MAX_CONCURRENCY=3`
5. 降低融合回答长度：`MAX_FUSION_LENGTH=1500`

### Q8: 支持本地部署的开源模型吗？

**A:** 是的！使用兼容 OpenAI API 的本地服务：

```env
# 示例：使用 LocalAI
CUSTOM_LLM_API_KEY=not-needed
CUSTOM_LLM_BASE_URL=http://localhost:8080/v1
```

然后在 `ai_fusion_utils.py` 中添加模型配置。

---

## 高级配置示例

### 生产环境配置

```env
# API配置
OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
DASHSCOPE_API_KEY=sk-xxxxx

# 性能优化
MAX_RETRIES=3
REQUEST_TIMEOUT=90
MAX_CONCURRENCY=5
RATE_LIMIT_PER_MINUTE=100

# 质量优先
ENABLE_DEEP_ANALYSIS=true
GENERATE_DETAILED_REPORT=true

# 日志
LOG_LEVEL=INFO
VERBOSE_LOGGING=false
LOG_FILE_PATH=./logs/production.log

# 安全
ENABLE_CONTENT_SAFETY=true
```

### 开发/测试环境配置

```env
# API配置（使用一个即可）
OPENAI_API_KEY=sk-proj-xxxxx

# 快速调试
MAX_RETRIES=2
REQUEST_TIMEOUT=30
MAX_CONCURRENCY=3

# 禁用耗时功能
ENABLE_DEEP_ANALYSIS=false
GENERATE_DETAILED_REPORT=false

# 详细日志
LOG_LEVEL=DEBUG
VERBOSE_LOGGING=true

# 启用缓存
ENABLE_RESPONSE_CACHE=true
CACHE_EXPIRATION=1800
```

### 演示/展示环境配置

```env
# 全部 API
OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
DASHSCOPE_API_KEY=sk-xxxxx

# 质量优先，展示完整功能
ENABLE_DEEP_ANALYSIS=true
GENERATE_DETAILED_REPORT=true
MAX_FUSION_LENGTH=3000

# 适中的性能设置
MAX_CONCURRENCY=5
REQUEST_TIMEOUT=60

# 清晰的日志
LOG_LEVEL=INFO
VERBOSE_LOGGING=true
```

---

## 技术支持

如有配置问题，请：
1. 查看 [README.md](README.md) 中的 FAQ 部分
2. 检查日志文件获取详细错误信息
3. 在 GitHub Issues 中提问
4. 参考项目文档

---

**最后更新：** 2025-10-09
**版本：** 1.0.0
