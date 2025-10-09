# AI Fusion Agent 项目结构

## 📁 目录结构

```
ai-fusion-agent/
├── ai_fusion/                  # 核心包
│   ├── __init__.py            # 包初始化，导出 AIFusionFlow
│   ├── core/                   # 核心功能模块
│   │   ├── __init__.py
│   │   ├── main.py            # AIFusionFlow 主类
│   │   └── nodes.py           # LangGraph 节点定义
│   ├── analysis/              # 分析模块
│   │   ├── __init__.py
│   │   ├── quality_analyzer.py # 质量分析器
│   │   └── smart_selector.py   # 智能选择器
│   ├── reporting/             # 报告模块
│   │   ├── __init__.py
│   │   └── reporter.py        # 报告生成器
│   └── utils/                 # 工具模块
│       ├── __init__.py
│       └── helpers.py         # 工具函数
├── examples/                   # 示例和演示
│   ├── __init__.py
│   └── demo.py                # 演示脚本
├── tests/                      # 测试文件
│   └── __init__.py
├── docs/                       # 文档
│   ├── CONFIG.md              # 配置说明
│   └── evaluation_standards.md # 评估标准
├── outputs/                    # 输出文件
│   └── reports/               # 生成的报告
├── .old_structure/            # 旧文件备份（可删除）
├── .env.example               # 环境变量示例
├── .gitignore                 # Git 忽略文件
├── requirements.txt           # Python 依赖
├── run.py                     # 主入口文件
├── README.md                  # 项目说明
└── PROJECT_STRUCTURE.md       # 本文件
```

## 📦 模块说明

### 核心包 (ai_fusion/)

#### 1. core/ - 核心功能模块
- **main.py**: `AIFusionFlow` 主流程控制器
  - 处理用户问题
  - 协调各个节点
  - 生成质量分析和报告

- **nodes.py**: LangGraph 节点定义
  - `ModelSelectorNode`: 智能模型选择器节点
  - `ParallelLLMNode`: 并发 LLM 调用节点
  - `FusionAgentNode`: 回答融合节点

#### 2. analysis/ - 分析模块
- **quality_analyzer.py**: `AIFusionQualityAnalyzer`
  - 多维度质量分析
  - LLM 评估
  - 对比分析

- **smart_selector.py**: `AIFusionSmartSelector`
  - 智能模型选择
  - 问题分析
  - 模型推荐

#### 3. reporting/ - 报告模块
- **reporter.py**: `AIFusionReporter`
  - 生成 Markdown 报告
  - 质量分析展示
  - 控制台摘要输出

#### 4. utils/ - 工具模块
- **helpers.py**: 工具函数
  - `call_llm_async()`: 异步 LLM 调用
  - `get_available_models()`: 获取可用模型
  - `validate_environment()`: 环境验证
  - `ModelConfig`: 模型配置类

### 示例和测试

#### examples/ - 示例代码
- **demo.py**: 完整演示脚本
  - 基本使用演示
  - 错误处理演示
  - 环境配置演示
  - 交互模式演示

#### tests/ - 测试代码
- 单元测试（待添加）
- 集成测试（待添加）

### 文档和配置

#### docs/ - 文档目录
- **CONFIG.md**: API 密钥配置说明
- **evaluation_standards.md**: 质量评估标准

#### outputs/ - 输出目录
- **reports/**: 自动生成的分析报告

## 🚀 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

### 3. 运行项目

**方式一：使用主入口文件**
```bash
python run.py
```

**方式二：运行演示脚本**
```bash
python examples/demo.py
```

**方式三：作为 Python 包使用**
```python
from ai_fusion import AIFusionFlow

ai_fusion = AIFusionFlow()
answer = await ai_fusion.process_question("你的问题")
```

## 📝 导入示例

```python
# 导入主类
from ai_fusion import AIFusionFlow

# 导入工具函数
from ai_fusion.utils import validate_environment, get_available_models

# 导入分析器
from ai_fusion.analysis import AIFusionQualityAnalyzer, AIFusionSmartSelector

# 导入报告器
from ai_fusion.reporting import AIFusionReporter
```

## 🔧 维护说明

### 添加新功能
1. 确定功能属于哪个模块（core/analysis/reporting/utils）
2. 在对应目录下创建新文件或修改现有文件
3. 在模块的 `__init__.py` 中导出新功能
4. 更新文档

### 迁移注意事项
- ✅ 所有旧文件已备份到 `.old_structure/`
- ✅ 可以安全删除 `.old_structure/` 目录
- ✅ 新结构完全向后兼容
- ✅ 导入路径已全部更新

## 📊 项目优势

1. **模块化设计**: 功能清晰分离，易于维护
2. **可扩展性**: 新功能可轻松添加到对应模块
3. **代码复用**: 通过包导入机制提高代码复用
4. **规范化**: 符合 Python 项目最佳实践
5. **易于测试**: 模块化结构便于编写单元测试

## 🗑️ 清理旧文件

确认新结构工作正常后，可以删除备份：
```bash
rm -rf .old_structure/
```
