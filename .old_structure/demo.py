#!/usr/bin/env python3
"""
AI Fusion 演示脚本
展示系统的各种功能和使用方法
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加PocketFlow路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PocketFlow'))
load_dotenv()

from ai_fusion_main import AIFusionFlow
from ai_fusion_utils import validate_environment, test_all_models, setup_example_env

async def demo_basic_usage():
    """演示基本使用方法"""
    print("🚀 AI Fusion 基本使用演示")
    print("=" * 50)
    
    # 创建AI Fusion实例
    ai_fusion = AIFusionFlow()
    
    # 准备演示问题
    demo_questions = [
        "什么是机器学习？",
        "写一首关于春天的诗",
        "解释一下快速排序算法",
        "如何学好Python编程？",
        "区块链技术有什么应用？"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 演示问题 {i}: {question}")
        print("-" * 30)
        
        try:
            answer = await ai_fusion.process_question(question)
            print(f"\n✅ 融合回答:\n{answer[:200]}..." if len(answer) > 200 else f"\n✅ 融合回答:\n{answer}")
            
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
        
        print("\n" + "=" * 50)
        
        # 添加延迟避免API限制
        await asyncio.sleep(2)

async def demo_error_handling():
    """演示错误处理机制"""
    print("🛡️ AI Fusion 错误处理演示")
    print("=" * 50)
    
    ai_fusion = AIFusionFlow()
    
    # 测试空问题
    print("\n📝 测试空问题处理:")
    try:
        await ai_fusion.process_question("")
    except Exception as e:
        print(f"✅ 正确捕获错误: {str(e)}")
    
    # 测试超长问题
    print("\n📝 测试超长问题处理:")
    long_question = "这是一个很长的问题。" * 100
    try:
        answer = await ai_fusion.process_question(long_question)
        print("✅ 超长问题处理成功")
    except Exception as e:
        print(f"⚠️ 超长问题处理异常: {str(e)}")

def demo_environment_setup():
    """演示环境配置"""
    print("⚙️ AI Fusion 环境配置演示")
    print("=" * 50)
    
    print("\n1. 检查当前环境:")
    if validate_environment():
        print("✅ 环境配置正确")
    else:
        print("❌ 环境配置需要修复")
        print("\n2. 显示配置示例:")
        setup_example_env()

async def demo_model_testing():
    """演示模型连接测试"""
    print("🧪 AI Fusion 模型测试演示")
    print("=" * 50)
    
    await test_all_models()

async def demo_interactive_mode():
    """演示交互模式"""
    print("💬 AI Fusion 交互模式演示")
    print("=" * 50)
    print("提示: 输入 'quit' 退出演示")
    
    ai_fusion = AIFusionFlow()
    
    demo_count = 0
    max_demos = 3  # 限制演示次数
    
    while demo_count < max_demos:
        try:
            question = input(f"\n[演示 {demo_count + 1}/{max_demos}] 请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                break
            
            if not question:
                print("⚠️ 请输入有效问题")
                continue
            
            answer = await ai_fusion.process_question(question)
            print(f"\n🎯 融合回答:\n{answer}")
            
            demo_count += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 演示被中断")
            break
        except Exception as e:
            print(f"❌ 演示错误: {str(e)}")

async def main():
    """主演示函数"""
    print("🌟 欢迎使用 AI Fusion 演示程序!")
    print("本程序将展示 AI Fusion 的各种功能")
    print("=" * 60)
    
    demos = [
        ("环境配置检查", demo_environment_setup, False),
        ("模型连接测试", demo_model_testing, True),
        ("基本使用方法", demo_basic_usage, True),
        ("错误处理机制", demo_error_handling, True),
        ("交互模式体验", demo_interactive_mode, True),
    ]
    
    for i, (name, demo_func, is_async) in enumerate(demos, 1):
        print(f"\n🎯 演示 {i}: {name}")
        print("=" * 40)
        
        try:
            if is_async:
                await demo_func()
            else:
                demo_func()
                
        except Exception as e:
            print(f"❌ 演示 {i} 失败: {str(e)}")
        
        # 询问是否继续
        if i < len(demos):
            try:
                continue_demo = input(f"\n继续下一个演示? (y/n): ").strip().lower()
                if continue_demo not in ['y', 'yes', '是', '']:
                    break
            except KeyboardInterrupt:
                print("\n\n👋 演示被用户中断")
                break
    
    print("\n🎉 AI Fusion 演示完成！")
    print("感谢您的使用，更多信息请查看 README.md")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行错误: {str(e)}")
        print("请检查您的环境配置和网络连接")