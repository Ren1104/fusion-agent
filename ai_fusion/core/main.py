"""
AI Fusion - 程序入口
交互式聊天界面和主函数
"""

import asyncio

from ai_fusion.core.flow import create_ai_fusion_flow
from ai_fusion.utils.helpers import get_available_models, validate_environment


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
