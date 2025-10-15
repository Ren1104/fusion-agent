#!/usr/bin/env python3
"""AI Fusion 命令行入口"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from flow import create_ai_fusion_flow


def check_env():
    """检查环境变量"""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ 请至少配置一个 API Key:")
        print("   OPENAI_API_KEY 或 ANTHROPIC_API_KEY")
        return False
    return True


async def chat():
    """交互式聊天"""
    print("🌟 欢迎使用 AI Fusion!")
    print("输入 'quit' 或 'exit' 退出\n")

    flow = create_ai_fusion_flow()

    while True:
        try:
            question = input("💬 请输入您的问题: ").strip()

            if question.lower() in ["quit", "exit", "退出"]:
                print("👋 再见!")
                break

            if not question:
                continue

            shared = {"user_question": question}
            await flow.run_async(shared)

            print(f"\n🎯 回答:\n{shared.get('final_answer', '处理失败')}\n")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    if check_env():
        asyncio.run(chat())
