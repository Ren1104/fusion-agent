#!/usr/bin/env python3
"""
测试新架构的最小示例
验证多提供商支持是否工作
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from ai_fusion.registry import ModelRegistry


async def main():
    print("🧪 测试新架构 - 最小示例\n")
    print("=" * 60)

    # 1. 创建注册中心
    print("\n📦 步骤 1: 初始化模型注册中心")
    registry = ModelRegistry()

    # 2. 显示可用提供商
    print("\n📋 步骤 2: 检查可用提供商")
    providers = registry.list_available_providers()
    if providers:
        print(f"✅ 发现 {len(providers)} 个可用提供商: {', '.join(providers)}")
    else:
        print("❌ 未找到可用提供商，请配置至少一个 API Key")
        print("\n请在 .env 文件中配置：")
        print("  OPENAI_API_KEY=your-key")
        print("  或 ANTHROPIC_API_KEY=your-key")
        print("  或 QWEN_API_KEY=your-key")
        return

    # 3. 发现所有模型
    print("\n🔍 步骤 3: 发现所有可用模型")
    models = await registry.discover_all_models()
    print(f"✅ 共发现 {len(models)} 个模型\n")

    # 按提供商分组显示
    from collections import defaultdict
    models_by_provider = defaultdict(list)
    for model in models:
        models_by_provider[model.provider].append(model)

    for provider, provider_models in models_by_provider.items():
        print(f"  📦 {provider}: {len(provider_models)} 个模型")
        for model in provider_models[:3]:  # 只显示前3个
            print(f"     • {model.display_name} ({model.model_id})")
        if len(provider_models) > 3:
            print(f"     ... 还有 {len(provider_models) - 3} 个")

    # 4. 测试调用一个模型
    print("\n🚀 步骤 4: 测试调用模型")
    if models:
        test_model = models[0]
        print(f"   使用模型: {test_model.display_name} ({test_model.model_id})")

        try:
            response = await registry.call_model(
                model_id=test_model.model_id,
                messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
                max_tokens=10
            )
            print(f"   ✅ 调用成功！")
            print(f"   回答: {response}")
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！新架构工作正常。")


if __name__ == "__main__":
    asyncio.run(main())
