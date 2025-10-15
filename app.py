#!/usr/bin/env python3
"""AI Fusion FastAPI 服务"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flow import create_ai_fusion_flow


def check_env():
    """检查环境变量"""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("请至少配置一个 API Key: OPENAI_API_KEY 或 ANTHROPIC_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时检查
    check_env()
    yield
    # 关闭时清理（如果需要）


app = FastAPI(title="AI Fusion API", version="1.0.0", lifespan=lifespan)
flow = create_ai_fusion_flow()


class Question(BaseModel):
    question: str


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/query")
async def query(q: Question):
    """提交问题"""
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        shared = {"user_question": q.question}
        await flow.run_async(shared)
        return {
            "answer": shared.get("final_answer", "处理失败"),
            "models": [m.name for m in shared.get("selected_models", [])],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000

    print("\n🚀 正在启动 AI Fusion API 服务...")
    print(f"📡 服务地址: http://{host}:{port}")
    print(f"📚 API 文档: http://localhost:{port}/docs")
    print(f"🔍 健康检查: http://localhost:{port}/health\n")

    uvicorn.run("app:app", host=host, port=port, reload=True)
