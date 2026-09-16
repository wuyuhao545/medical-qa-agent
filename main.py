#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流行病学知识问答系统 - RAG+Agent 混合模式

用法：
    python main.py                # 正常启动（含文件监控）
    python main.py --rebuild      # 强制重建 FAISS 知识库
    python main.py --no-cache     # 禁用缓存
    python main.py --no-watch     # 禁用文件自动监控
    python main.py --help         # 显示帮助
"""
import sys
import time
import asyncio
from pathlib import Path
from typing import List

# -------- 配置导入 --------
from config import (
    CACHE_ENABLED, LLM_CACHE_BACKEND, REDIS_URL, CHROMA_QA_DIR,
    TOKEN_TRACK_ENABLED, TOKEN_BUDGET_PER_SESSION, TOKEN_BUDGET_TOTAL, TOKEN_WARN_RATIO,
)
from utils.token_tracker import (
    init_tracker, get_tracker, TokenBudgetExceeded,
)

# -------- 核心模块 --------
from core.cache.cache_manager import init_llm_cache
from RAG.Document.document_loader import load_documents, load_documents_from_paths
from RAG.splitter import SplitterManager
from RAG.vector_stores.faiss_store import FaissVectorStore
from RAG.vector_stores.qa_chroma import QaVectorStore
from utils.file_watcher import start_file_watcher
from core.tools.search_tools import set_all_chunks, get_vector_store, set_raw_documents
from agents.medical_agent import create_medical_agent, run_agent
from utils.safety import check_input_safety   # 统一安全检测

# -------- 模块级全局状态 --------
_RAW_DOCUMENTS = None
_APP_SHUTTING_DOWN = False   # 关机标志：一旦置 True，on_new_files 会提前返回


# ---------- 帮助信息 ----------
def print_help():
    print("""
使用方法:
    python main.py [选项]
选项:
    --rebuild      强制重建 FAISS 知识库
    --no-cache     禁用缓存（覆盖 config.py 中的 CACHE_ENABLED）
    --no-watch     禁用文件自动监控（不会自动更新知识库）
    --verbose      输出详细日志
    --help         显示此帮助信息
    """)
    sys.exit(0)


#  初始化 FAISS 知识库
def init_knowledge_base(rebuild: bool = False, verbose: bool = True) -> bool:
    """
    构建或加载 FAISS 医学知识库，并设置全局块列表供 BM25 使用。
    """
    global _RAW_DOCUMENTS
    try:
        print("=" * 70)
        print("📦 初始化 FAISS 医学知识库")
        if rebuild:
            print("🔄 强制重建模式")
        print("=" * 70)

        # 1. 加载原始文档
        print("📂 正在加载文档...")
        docs = load_documents(verbose=verbose)
        _RAW_DOCUMENTS = docs  # 保存到全局，供父文档检索器使用
        if not docs:
            print("❌ 没有加载到任何文档，请检查 data/medical_knowledge/ 目录")
            return False
        print(f"📄 共加载 {len(docs)} 个原始文档")

        # 2. 初始化 FAISS 存储
        vector_store = FaissVectorStore()
        exists = vector_store.load_vector_store(verbose=False)
        need_rebuild = rebuild or not exists

        # 3. 切分文档
        print("✂️ 切分文档...")
        splitter = SplitterManager()
        chunks = splitter.split_documents(docs)
        set_all_chunks(chunks)
        print(f"📦 共生成 {len(chunks)} 个文本块")

        # 4. 重建或加载
        if need_rebuild:
            print("🔄 重建 FAISS 向量库...") # ★ 首次导入大 JSONL 时，嵌入计算耗时较长，耐心等待
            vector_store.build_vector_store(chunks, verbose=verbose)
            print("✅ FAISS 向量库构建完成")
        else:
            print("✅ 已有 FAISS 向量库可用，跳过构建")

        # 5. 保存全局状态，供 BM25 / 父文档检索使用
        set_all_chunks(chunks)
        set_raw_documents(docs)
        print("📄 已加载文档切分用于 BM25 检索")

        return True

    except Exception as e:
        import traceback
        print("=" * 70)
        print("❌ 初始化知识库时发生异常：")
        traceback.print_exc()
        print("=" * 70)
        return False


# ---------- 文件变更回调（由 watchdog 触发） ----------
def on_new_files(file_paths: List[Path]):
    # 关机期间不再处理，避免与主线程退出流程打架
    if _APP_SHUTTING_DOWN:
        print("ℹ️ 应用正在退出，跳过文件更新")
        return

    print(f"📂 检测到新文件: {[f.name for f in file_paths]}")
    try:
        new_docs = load_documents_from_paths(file_paths, verbose=True)
        if not new_docs:
            print("⚠️ 新文件无内容，跳过更新")
            return

        splitter = SplitterManager()
        chunks = splitter.split_documents(new_docs)
        if not chunks:
            print("⚠️ 切分后无块，跳过更新")
            return

        # 1. 更新主 FAISS
        vector_store = get_vector_store()
        if vector_store.vector_store is None:
            vector_store.load_vector_store(verbose=False)
        vector_store.add_documents(chunks, verbose=True)

        # 2. 更新 BM25 全局块列表（用模块引用，不要 from import）
        import core.tools.search_tools as search_tools
        existing = search_tools._ALL_CHUNKS
        if existing is not None:
            existing.extend(chunks)
        else:
            existing = chunks
        set_all_chunks(existing)

        # 3. 更新原始文档列表（供父检索器用）
        global _RAW_DOCUMENTS
        if _RAW_DOCUMENTS is not None:
            _RAW_DOCUMENTS.extend(new_docs)
            set_raw_documents(_RAW_DOCUMENTS)

        # 4. 增量更新父文档索引（如果之前已经构建过）
        try:
            from core.retrievers.parent_doc_retriever import (
                add_documents_to_parent_retriever,
                clear_parent_retriever_cache,
            )
            # 先尝试增量添加
            ok = add_documents_to_parent_retriever(new_docs, verbose=True)
            if not ok:
                # 增量失败 → 清缓存，下次 deep 档会整体重建
                clear_parent_retriever_cache()
                print("ℹ️ 父文档索引已失效，下次 deep 检索将重建")
        except Exception as e:
            print(f"⚠️ 父文档索引更新失败: {e}")

        print("✅ 增量更新完成")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 增量更新失败: {e}")


# ---------- 初始化 Agent ----------
def init_agent():
    from agents.medical_agent import create_medical_agent
    print("🤖 初始化 Agent...")
    agent_llm = create_medical_agent(temperature=0.3)
    print("✅ Agent 初始化完成")
    return agent_llm


# ---------- 交互模式 ----------
async def interactive_agent(agent_llm, qa_store):
    from agents.medical_agent import run_agent

    print("\n" + "=" * 70)
    print("💬 进入智能问答模式 (输入 'quit' 退出)")
    print("=" * 70)

    tracker = get_tracker()

    while True:
        question = await asyncio.to_thread(input, "\n❓ 您: ")
        question = question.strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
        if not question:
            continue

        # 允许用户手动重置会话预算
        if question.lower() == "reset":
            tracker.reset_session()
            print("♻️ 会话 Token 预算已重置")
            continue

        # 安全检测
        safety_msg = check_input_safety(question)
        if safety_msg:
            print(f"\n🛡️ {safety_msg}")
            continue

        # 开启本次请求的 token 记账
        tracker.begin_request()

        print("\n🤖 助手: ", end="", flush=True)
        start_time = time.time()
        try:
            answer = await asyncio.to_thread(run_agent, agent_llm, question)
            print(f"\n🤖 助手: {answer}\n")
            elapsed = time.time() - start_time
            print(f"(⏱️ 耗时: {elapsed:.2f} 秒)\n")

            # 记录问答到 Chroma QA 库（异步执行，不阻塞事件循环）
            await asyncio.to_thread(
                qa_store.add_qa_pair,
                question,
                answer,
                {"timestamp": time.time()},
            )
        except TokenBudgetExceeded as e:
            print(f"\n💰 {e}")
            print("  本次会话 Token 已用尽。输入 'reset' 重置会话预算，或 'quit' 退出。")
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")

        finally:
            # 本次请求结算
            usage = tracker.end_request({"question": question[:30]})
            print(f"(🪙 本次消耗: {usage.total_tokens} tokens "
                  f"| 输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens})")


# ---------- 主函数 ----------
async def main():
    global _APP_SHUTTING_DOWN

    rebuild = False
    no_cache = False
    no_watch = False
    verbose = True

    for arg in sys.argv[1:]:
        if arg == "--rebuild":
            rebuild = True
        elif arg == "--no-cache":
            no_cache = True
        elif arg == "--no-watch":
            no_watch = True
        elif arg == "--verbose":
            verbose = True
        elif arg in ["--help", "-h"]:
            print_help()
        else:
            print(f"⚠️ 未知参数: {arg}，使用 --help 查看帮助")
            sys.exit(1)

    # 1. Token 统计与预算
    if TOKEN_TRACK_ENABLED:
        init_tracker(
            per_session_limit=TOKEN_BUDGET_PER_SESSION,
            total_limit=TOKEN_BUDGET_TOTAL,
            warn_ratio=TOKEN_WARN_RATIO,
        )
        print(f"💰 Token 预算: 会话 {TOKEN_BUDGET_PER_SESSION} / 全局 {TOKEN_BUDGET_TOTAL}")
    else:
        print("ℹ️ Token 统计已禁用")

    # 2. 缓存
    if CACHE_ENABLED and not no_cache:
        init_llm_cache(backend=LLM_CACHE_BACKEND, redis_url=REDIS_URL)
    else:
        print("ℹ️ 缓存已禁用")

    # 3. 知识库
    success = init_knowledge_base(rebuild=rebuild, verbose=verbose)
    if not success:
        print("❌ 知识库初始化失败，退出")
        return

    # 4. Agent
    agent_llm = init_agent()
    if agent_llm is None:
        print("❌ Agent 初始化失败")
        return

    # 5. QA 存储
    qa_store = QaVectorStore()
    qa_store.load_or_create(verbose=True)

    # 6. 文件监控
    watcher = None
    if not no_watch:
        try:
            watcher = start_file_watcher(
                watch_dir="data/medical_knowledge",
                on_updated=on_new_files,
                max_cache_size=1000,
                debounce_seconds=2.0,
            )
            print("👀 已启动文件自动监控，新增文件将自动更新 FAISS 知识库。")
        except ImportError:
            print("⚠️ 未安装 watchdog，文件监控功能不可用。请安装: pip install watchdog")
        except Exception as e:
            print(f"⚠️ 启动文件监控失败: {e}")

    # 7. 交互循环
    try:
        await interactive_agent(agent_llm, qa_store)
    finally:
        # 先置关机标志，让 on_new_files 提前返回
        _APP_SHUTTING_DOWN = True
        # 停监控：内部依次 observer.stop → 取消 pending timer → 等待正在执行的回调 → observer.join
        if watcher:
            watcher.stop(wait=True)
            print("🛑 文件监控已停止")

    # 8. Token 汇总
    if TOKEN_TRACK_ENABLED:
        print("\n" + get_tracker().summary())


if __name__ == "__main__":
    asyncio.run(main())