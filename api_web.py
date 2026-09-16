# web_app.py
"""
FastAPI Web 服务：把现有的 RAG + Agent 系统包装成 HTTP 接口。

运行：
    pip install fastapi uvicorn[standard]
    python web_app.py
    或
    uvicorn web_app:app --host 0.0.0.0 --port 8000

访问：http://localhost:8000
"""
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    CACHE_ENABLED, LLM_CACHE_BACKEND, REDIS_URL,
    TOKEN_TRACK_ENABLED, TOKEN_BUDGET_PER_SESSION, TOKEN_BUDGET_TOTAL, TOKEN_WARN_RATIO,
)
from utils.token_tracker import init_tracker, get_tracker, TokenBudgetExceeded
from core.cache.cache_manager import init_llm_cache
from RAG.vector_stores.qa_chroma import QaVectorStore
from agents.medical_agent import run_agent
from utils.safety import check_input_safety

# 复用 main.py 的初始化逻辑
from main import init_knowledge_base, init_agent

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 正在启动 Web 服务...")
    print("=" * 60)

    # Token 统计
    if TOKEN_TRACK_ENABLED:
        init_tracker(
            per_session_limit=TOKEN_BUDGET_PER_SESSION,
            total_limit=TOKEN_BUDGET_TOTAL,
            warn_ratio=TOKEN_WARN_RATIO,
        )
        print(f"💰 Token 预算: 会话 {TOKEN_BUDGET_PER_SESSION} / 全局 {TOKEN_BUDGET_TOTAL}")
    # LLM 缓存
    if CACHE_ENABLED:
        init_llm_cache(backend=LLM_CACHE_BACKEND, redis_url=REDIS_URL)

    # 知识库（同步阻塞，放到线程里）
    ok = await asyncio.to_thread(init_knowledge_base, False, False)
    if not ok:
        raise RuntimeError("❌ 知识库初始化失败，Web 服务无法启动")
    
    # agent
    app.state.agent = await asyncio.to_thread(init_agent)

    # 5. QA 存储
    qa_store = QaVectorStore()
    await asyncio.to_thread(qa_store.load_or_create, False)
    app.state.qa_store = qa_store

    print("=" * 60)
    print("✅ Web 服务就绪 → http://localhost:8000")
    print("=" * 60)
    try:
        yield
    finally:
        print("👋 Web 服务关闭")

app = FastAPI(title="智医助手 · 流行病学知识问答", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==================== 数据模型 ====================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    elapsed: float = 0.0
    blocked: bool = False
    block_type: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StatusResponse(BaseModel):
    session_tokens: int
    total_tokens: int
    per_session_limit: Optional[int]
    total_limit: Optional[int]


# ==================== 工具 ====================
def _classify_block(msg: str) -> str:
    if "紧急症状" in msg:
        return "emergency"
    if "危险指令" in msg:
        return "attack"
    if "诊疗或用药" in msg:
        return "diagnosis"
    return "other"


async def _save_qa(qa_store: QaVectorStore, q: str, a: str):
    try:
        await asyncio.to_thread(qa_store.add_qa_pair, q, a, {"timestamp": time.time()})
    except Exception as e:
        print(f"[QA 存储失败] {e}")


# ==================== 路由 ====================
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status", response_model=StatusResponse)
async def status():
    tracker = get_tracker()
    s = tracker.session_usage()
    t = tracker.total_usage()
    return StatusResponse(
        session_tokens=s.total_tokens,
        total_tokens=t.total_tokens,
        per_session_limit=tracker.per_session_limit,
        total_limit=tracker.total_limit,
    )


@app.post("/api/reset")
async def reset():
    get_tracker().reset_session()
    return {"ok": True, "message": "会话 Token 预算已重置"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # ---------- 1. 统一安全检测 ----------
    try:
        safety_msg = await asyncio.to_thread(check_input_safety, question)
    except Exception as e:
        print(f"[安全检测异常，默认放行] {e}")
        safety_msg = None

    if safety_msg:
        return ChatResponse(
            answer=safety_msg,
            blocked=True,
            block_type=_classify_block(safety_msg),
        )

    # ---------- 2. 调用 Agent ----------
    tracker = get_tracker()
    tracker.begin_request()
    start = time.time()

    try:
        answer = await asyncio.to_thread(run_agent, app.state.agent, question)
    except TokenBudgetExceeded as e:
        tracker.end_request({"question": question[:30]})
        raise HTTPException(status_code=429, detail=f"💰 {e}")
    except Exception as e:
        tracker.end_request({"question": question[:30]})
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {e}")

    elapsed = time.time() - start
    usage = tracker.end_request({"question": question[:30]})

    # ---------- 3. 异步落库 ----------
    if answer:
        asyncio.create_task(_save_qa(app.state.qa_store, question, answer))

    return ChatResponse(
        answer=answer,
        elapsed=elapsed,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)   # ← 直接传 app 对象