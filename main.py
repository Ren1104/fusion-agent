#!/usr/bin/env python3
"""
AI Fusion 主入口文件
启动AI Fusion交互式聊天系统
"""

import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from flow import create_ai_fusion_flow
from analyzer import ModelConfig


def get_available_models():
    """
    获取所有可用的模型配置列表

    Returns:
        List[ModelConfig]: 模型配置列表
    """
    import os

    # 从环境变量获取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("❌ 未配置 OPENAI_API_KEY 环境变量")
        return []

    # 获取可用模型列表
    if os.getenv("AVAILABLE_MODELS"):
        model_names = [m.strip() for m in os.getenv("AVAILABLE_MODELS").split(",")]
    else:
        # 默认模型列表
        model_names = [
            "qwen-max",
            "qwen-plus",
            "claude_sonnet4",
            "gpt-41-0414-global",
            "claude37_sonnet_new",
            "gpt-41-mini-0414-global",
            "glm-4.5",
            "qwen3-coder-480b-a35b-instruct"
        ]

    # 构建模型配置列表
    models = []
    for model_name in model_names:
        models.append(ModelConfig(
            name=model_name,
            provider="openai",
            api_key=api_key,
            base_url=base_url
        ))

    return models


def validate_environment():
    """验证环境配置"""
    import os

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ 环境配置检查失败！")
        print("请配置以下环境变量：")
        print("  OPENAI_API_KEY: API 密钥（必需）")
        print("  OPENAI_BASE_URL: API 基础 URL（可选）")
        print("  AVAILABLE_MODELS: 可用模型列表，逗号分隔（可选）")
        return False

    models = get_available_models()
    print(f"✅ 环境验证通过！")
    print(f"📦 API Base URL: {os.getenv('OPENAI_BASE_URL', '默认')}")
    print(f"🤖 可用模型数量: {len(models)}")

    return True


async def process_question(question: str):
    """
    处理单个用户问题

    Args:
        question: 用户输入的问题

    Returns:
        融合后的最终回答
    """
    # 创建flow
    flow = create_ai_fusion_flow()

    # 初始化shared状态
    shared = {
        "user_question": question,
        "available_models": get_available_models(),
    }

    print(f"\n🤖 AI Fusion 正在处理您的问题: {question}")
    print("=" * 50)

    try:
        # 运行AI Fusion流程（所有逻辑都在节点中）
        await flow.run_async(shared)

        # 返回最终回答
        return shared.get("final_answer", "处理失败")

    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"抱歉，处理您的问题时出现了错误: {str(e)}"


async def interactive_chat():
    """交互式聊天界面"""
    print("🌟 欢迎使用 AI Fusion 智能融合系统!")
    print("本系统会自动选择三个最合适的LLM模型来回答您的问题")
    print("输入 'exit' 或 'quit' 退出程序")
    print("=" * 60)

    # 验证环境
    if not validate_environment():
        print("❌ 环境验证失败，请检查API密钥配置")
        return

    while True:
        try:
            user_input = input("\n💬 请输入您的问题: ").strip()

            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 感谢使用 AI Fusion，再见!")
                break

            if not user_input:
                print("⚠️ 请输入有效的问题")
                continue

            # 处理用户问题
            answer = await process_question(user_input)
            print(f"\n🎯 AI Fusion 融合回答:\n{answer}")
            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生未预期的错误: {str(e)}")


def main():
    """主函数"""
    print("🚀 启动 AI Fusion 系统...")

    try:
        asyncio.run(interactive_chat())
    except Exception as e:
        print(f"❌ 系统启动失败: {str(e)}")


if __name__ == "__main__":
    main()
