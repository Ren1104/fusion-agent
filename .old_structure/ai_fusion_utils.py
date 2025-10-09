"""
AI Fusion工具函数和配置管理
包含LLM调用、模型配置、环境验证等实用功能
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

@dataclass
class ModelConfig:
    """LLM模型配置类"""
    name: str
    provider: str  # openai, anthropic, google, etc.
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7

def get_available_models() -> List[ModelConfig]:
    """
    获取所有可用的LLM模型配置
    根据环境变量中的API密钥确定可用模型
    """
    models = []
    
    # OpenAI模型（支持自定义base_url）
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    
    if openai_key:
        # 根据错误信息中允许的模型列表添加所有可用模型
        allowed_models = [
            "qwen-max", "qwen-plus", "claude_sonnet4", "gpt-41-0414-global", 
            "claude37_sonnet_new", "gpt-41-mini-0414-global", "glm-4.5", 
            "openmatrix-qwen3-235b-inst-fp8", "qwen3-max-preview", 
            "gpt-5-mini-0807-global", "qwen3-coder-480b-a35b-instruct", 
            "qwen3-coder-plus1", "qwen3-coder-plus"
        ]
        
        for model_name in allowed_models:
            models.append(ModelConfig(
                name=model_name,
                provider="openai",
                api_key=openai_key,
                base_url=openai_base_url,
                max_tokens=2000
            ))
    
    
    return models

async def call_llm_async(
    messages: List[Dict[str, str]], 
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> str:
    """
    异步调用LLM的通用函数
    统一使用OpenAI客户端调用所有模型（包括Claude等）
    """
    
    # 统一使用OpenAI客户端调用所有模型
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # 如果调用失败，提供更详细的错误信息
        raise Exception(f"调用模型 {model} 失败: {str(e)}")


def validate_environment() -> bool:
    """
    验证环境配置
    检查是否配置了OpenAI API密钥
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_key:
        print("❌ 环境配置检查失败！")
        print("请配置以下环境变量：")
        print("- OPENAI_API_KEY: OpenAI API密钥")
        print("- OPENAI_BASE_URL: 自定义OpenAI兼容服务地址（可选）")
        return False
    
    available_models = get_available_models()
    print(f"✅ 环境验证通过！发现 {len(available_models)} 个可用模型:")
    for model in available_models:
        print(f"  - {model.name} ({model.provider})")
    
    return len(available_models) >= 1

def setup_example_env():
    """
    设置示例环境变量（仅用于演示）
    实际使用时请设置真实的API密钥
    """
    example_env = {
        "OPENAI_API_KEY": "your-openai-api-key-here",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",  # 可选，自定义OpenAI兼容服务地址
    }
    
    print("📝 示例环境变量配置:")
    for key, value in example_env.items():
        print(f"export {key}='{value}'")
    print("\n请根据您的实际情况设置相应的API密钥")

async def test_model_connection(model_config: ModelConfig) -> bool:
    """测试单个模型的连接"""
    try:
        test_messages = [{"role": "user", "content": "Hello, please respond with 'OK'"}]
        
        response = await call_llm_async(
            messages=test_messages,
            model=model_config.name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            max_tokens=10,
            temperature=0.1
        )
        
        print(f"✅ {model_config.name} 连接测试成功")
        return True
        
    except Exception as e:
        print(f"❌ {model_config.name} 连接测试失败: {str(e)}")
        return False

async def test_all_models():
    """测试所有可用模型的连接"""
    models = get_available_models()
    if not models:
        print("❌ 没有找到可用的模型配置")
        return
    
    print(f"🧪 开始测试 {len(models)} 个模型的连接...")
    
    tasks = [test_model_connection(model) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = sum(1 for result in results if result is True)
    print(f"\n📊 测试完成: {successful}/{len(models)} 个模型可用")

if __name__ == "__main__":
    # 运行环境验证和模型测试
    if validate_environment():
        print("\n🧪 运行模型连接测试...")
        asyncio.run(test_all_models())
    else:
        setup_example_env()