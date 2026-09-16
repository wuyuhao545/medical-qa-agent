# core/monitor/token_tracker.py
"""
Token 统计与预算控制。
- TokenTracker: 单例，线程安全，累计 token 消耗
- TokenTrackingCallback: LangChain 回调，挂 LLM 上自动统计
- 三层预算：单请求记录 / 会话上限 / 全局上限
- 超预算抛 TokenBudgetExceeded，由上层捕获
"""

import threading 
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 超出token预算异常
class TokenBudgetExceeded(Exception):
    """token 预算超限"""
    def __init__(self, scope: str, used: int, limit: int):
        self.scope = scope
        self.used = used
        self.limit = limit
        super().__init__(f"[{scope}] Token 预算超限: {used} / {limit}")

# 数据结构
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other:"TokenUsage"):
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
        }

@dataclass
class RequestRecord:
    request_id: str
    model: str
    usage: TokenUsage
    timestamp: float
    meta: Dict[str, Any] = field(default_factory=dict)

# 核心Tracker
class TokenTracker:
    def __init__(
        self,
        per_session_limit: Optional[int] = None,
        total_limit:Optional[int] = None,
        warn_ratio: float = 0.8,
        max_records: int = 1000,
    ):
        self.per_session_limit = per_session_limit
        self.total_limit       = total_limit
        self.warn_ratio        = warn_ratio
        self.max_records       = max_records

        self._lock = threading.Lock()                     # 创建一个私有线程锁（互斥锁）
        self._total = TokenUsage()                        # 全生命周期的累计用量对象
        self._session = TokenUsage()                      # 当前会话（session）的累计用量
        self._records: List[RequestRecord] = []           # 一个列表，保存每次请求的详细记录
        self._current_request_id: Optional[str] = None    # 记录当前正在进行的请求的 ID
        self._current_request_usage = TokenUsage()        # 累加当前这个请求的 token

    # 请求生命周期
    def begin_request(self) -> str:
        """开启一次新的用户请求"""
        with self._lock:
            self._current_request_id = str(uuid.uuid4())[:8]
            self._current_request_usage = TokenUsage()
        return self._current_request_id

    def end_request(self, meta: Optional[Dict] = None) -> TokenUsage:
        """结束当前用户请求，落一条 record，返回本次消耗。"""
        with self._lock:
            usage = TokenUsage(**self._current_request_usage.to_dict())
            if self._current_request_id is not None: 
                self._records.append(RequestRecord(# 构造一个 RequestRecord 对象，然后追加到 _records 列表末尾
                    request_id = self._current_request_id,
                    model = "multi",
                    usage = usage,
                    timestamp = time.time(),
                    meta = meta or {},
                ))
                if len(self._records) > self.max_records:
                    self._records = self._records[-self.max_records:]
            self._current_request_id = None
            return usage
    # ---------- 调用前检查 ----------
    def check_before_call(self):
        """LLM 调用前检查预算，超限抛 TokenBudgetExceeded。"""
        if self.per_session_limit is not None and self._session.total_tokens >= self.per_session_limit:
            raise TokenBudgetExceeded("session", self._session.total_tokens, self.per_session_limit)
        if self.total_limit is not None and self._total.total_tokens >= self.total_limit:
            raise TokenBudgetExceeded("total", self._total.total_tokens, self.total_limit)

    # ---------- 调用后记录 ----------
    def record(self, usage: TokenUsage):
        if usage.total_tokens <= 0:
            return
        with self._lock:
            self._total.add(usage)
            self._session.add(usage)
            self._current_request_usage.add(usage)
            self._maybe_warn("session", self._session.total_tokens, self.per_session_limit)
            self._maybe_warn("total",   self._total.total_tokens,   self.total_limit)

    def _maybe_warn(self, scope: str, used: int, limit: Optional[int]):
        if limit is None:
            return
        if limit * self.warn_ratio <= used < limit:
            print(f"⚠️ [{scope}] Token 使用接近上限: {used}/{limit}")

    # ---------- 查询 ----------
    def session_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(**self._session.to_dict())

    def total_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(**self._total.to_dict())

    def current_request_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(**self._current_request_usage.to_dict())

    def reset_session(self):
        with self._lock:
            self._session = TokenUsage()

    def summary(self) -> str:
        s = self.session_usage()
        t = self.total_usage()
        lines = [
            "=" * 56,
            "📊 Token 消耗统计",
            "-" * 56,
            f"  本次会话: 输入 {s.prompt_tokens}  输出 {s.completion_tokens}  合计 {s.total_tokens}",
            f"  累计总计: 输入 {t.prompt_tokens}  输出 {t.completion_tokens}  合计 {t.total_tokens}",
        ]
        if self.per_session_limit:
            lines.append(f"  会话预算: {s.total_tokens} / {self.per_session_limit}")
        if self.total_limit:
            lines.append(f"  全局预算: {t.total_tokens} / {self.total_limit}")
        lines.append("=" * 56)
        return "\n".join(lines)


# LangChain 回调 
class TokenTrackingCallback(BaseCallbackHandler):
    """挂在 LLM 上，自动提取 token 使用量交给 tracker。"""
    def __init__(self, tracker: TokenTracker):
        self.tracker = tracker

    def on_llm_start(self, serialized, prompts, **kwargs):
        # 调用前查预算
        self.tracker.check_before_call()

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = self._extract_usage(response)
        if usage is not None:
            self.tracker.record(usage)

    @staticmethod
    def _extract_usage(response: LLMResult) -> Optional[TokenUsage]:
        """兼容不同 LangChain 版本 / 供应商的 token 字段位置。"""
        # 路径 1: llm_output['token_usage'] 或 ['usage']
        llm_out = response.llm_output or {}
        usage = llm_out.get("token_usage") or llm_out.get("usage")
        if usage:
            return TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        # 路径 2: generations[0][0].generation_info['usage']
        try:
            gen = response.generations[0][0]
            info = getattr(gen, "generation_info", None) or {}
            usage = info.get("usage") or info.get("token_usage")
            if usage:
                return TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
        except (IndexError, AttributeError):
            pass
        return None

# 全局单例 
_tracker: Optional[TokenTracker] = None
_lock = threading.Lock()

def init_tracker(
    per_session_limit: Optional[int] = None,
    total_limit: Optional[int] = None,
    warn_ratio: float = 0.8,
) -> TokenTracker:
    global _tracker
    with _lock:
        _tracker = TokenTracker(
            per_session_limit=per_session_limit,
            total_limit=total_limit,
            warn_ratio=warn_ratio,
        )
    return _tracker

def get_tracker() -> TokenTracker:
    global _tracker
    if _tracker is None:
        init_tracker()
    return _tracker

def get_callback() -> TokenTrackingCallback:
    return TokenTrackingCallback(get_tracker())