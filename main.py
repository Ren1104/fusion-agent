#!/usr/bin/env python3
"""AI Fusion 命令行入口"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from flow import create_ai_fusion_flow
from providers import ModelRegistry, ModelInfo
from analyzer import ModelConfig
from langfuse_tracer import create_trace, finish_observation, flush


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

    # 初始化 ModelRegistry (新架构)
    print("🔧 正在初始化模型注册中心...")
    registry = ModelRegistry()
    available_models_info = await registry.discover_all_models()

    if not available_models_info:
        print("❌ 未发现任何可用模型，请检查环境配置")
        return

    print(f"✅ 成功加载 {len(available_models_info)} 个模型\n")

    # 转换 ModelInfo 为 ModelConfig（向后兼容）
    available_models = [
        ModelConfig(
            name=model.model_id,
            provider=model.provider,
            api_key="",  # 不再需要，registry 会处理
            base_url=None
        )
        for model in available_models_info
    ]

    flow = create_ai_fusion_flow()

    while True:
        try:
            question = input("💬 请输入您的问题: ").strip()

            if question.lower() in ["quit", "exit", "退出"]:
                print("👋 再见!")
                break

            if not question:
                continue

            # 创建 Langfuse trace
            trace = create_trace(
                name=question[:100],  # 使用问题作为 trace 名称（截断到100字符）
                metadata={"source": "main_cli"},
                input_data={"question": question}
            )
            trace_id = trace.trace_id if trace else None
            trace_observation_id = trace.id if trace else None

            # 通过 shared 传递 registry、模型列表和 trace_id
            shared = {
                "user_question": question,
                "registry": registry,
                "available_models": available_models,
                "trace_id": trace_id,  # 传递 trace_id 给所有节点
                "langfuse_trace_observation_id": trace_observation_id,
                "langfuse_trace": trace,
            }

            flow_error: Exception | None = None
            try:
                await flow.run_async(shared)
            except Exception as exc:  # 让外层捕获后提示用户
                flow_error = exc
                raise
            finally:
                if trace:
                    finish_observation(
                        trace,
                        output_data={
                            "final_answer": shared.get("final_answer"),
                            "quality_analysis": shared.get("quality_analysis"),
                            "report_path": shared.get("report_path"),
                        },
                        metadata={
                            "selected_models": [m.name for m in shared.get("selected_models", [])],
                            "analysis_method": shared.get("analysis_method"),
                        },
                        level="ERROR"
                        if flow_error or not shared.get("final_answer")
                        else None,
                        status_message=str(flow_error) if flow_error else None,
                    )

                flush()

            print(f"\n🎯 回答:\n{shared.get('final_answer', '处理失败')}\n")

            if trace_id:
                print(f"📊 Langfuse Trace ID: {trace_id}\n")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    if check_env():
        asyncio.run(chat())
