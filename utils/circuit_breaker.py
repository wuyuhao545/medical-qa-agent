# utils/circuit_breaker.py
"""
通用熔断器 + 慢调用熔断：
- CLOSED：正常放行
- OPEN：连续失败 / 慢调用比例超阈值 → 快速失败
- HALF_OPEN：冷却到时间，试探性放行
线程安全；按 name 全局注册，便于监控和手动重置。
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, Type
from functools import wraps
from collections import deque


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断器处于 OPEN 状态时抛出。"""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"[熔断] {name} 当前不可用，请 {retry_after:.1f} 秒后重试")


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        call_timeout: Optional[float] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.call_timeout = call_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    # ---------- 状态查询 ----------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self):
        """仅在持锁时调用。"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

    # ---------- 同步调用 ----------
    def call(self, func: Callable, *args, **kwargs) -> Any:
        # 1. 检查是否允许放行（持锁判断，随后立刻释放）
        with self._lock:
            self._maybe_half_open()
            if self._state == CircuitState.OPEN:
                retry_after = self.recovery_timeout - (time.time() - self._opened_at)
                raise CircuitOpenError(self.name, max(retry_after, 0.0))

        # 2. 不在锁内执行 func（关键修复：避免串行化、避免死锁）
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    # ---------- 异步调用 ----------
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            self._maybe_half_open()
            if self._state == CircuitState.OPEN:
                retry_after = self.recovery_timeout - (time.time() - self._opened_at)
                raise CircuitOpenError(self.name, max(retry_after, 0.0))

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    # ---------- 成功 / 失败处理 ----------
    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()

    # ---------- 手动重置 ----------
    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = 0.0


class SlowCallCircuitBreaker(CircuitBreaker):
    """
    在通用熔断基础上，增加"慢调用"触发 OPEN：
    - 最近 N 次调用中，慢调用比例 >= slow_call_rate → OPEN
    - 慢调用定义：单次耗时 > slow_call_threshold
    - 前 min_calls 次不判断，避免冷启动误判
    """

    def __init__(
        self,
        name: str,
        slow_call_threshold: float = 10.0,
        slow_call_rate: float = 0.5,
        window_size: int = 10,
        min_calls: int = 5,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.slow_call_threshold = slow_call_threshold
        self.slow_call_rate = slow_call_rate
        self.window_size = window_size
        self.min_calls = min_calls
        self._durations = deque(maxlen=window_size)

    def call(self, func, *args, **kwargs):
        start = time.time()
        try:
            result = super().call(func, *args, **kwargs)
        except Exception:
            raise
        else:
            self._record_duration(time.time() - start)
            return result

    async def acall(self, func, *args, **kwargs):
        start = time.time()
        try:
            result = await super().acall(func, *args, **kwargs)
        except Exception:
            raise
        else:
            self._record_duration(time.time() - start)
            return result

    def _record_duration(self, elapsed: float):
        with self._lock:
            self._durations.append(elapsed)
            if len(self._durations) < self.min_calls:
                return
            slow_count = sum(1 for d in self._durations if d > self.slow_call_threshold)
            slow_ratio = slow_count / len(self._durations)
            if slow_ratio >= self.slow_call_rate and self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                print(f"🐢 [{self.name}] 慢调用比例 {slow_ratio:.0%}，熔断 OPEN")


# ---------- 全局注册表 ----------
_BREAKERS: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_breaker(
    name: str,
    breaker_cls: Type[CircuitBreaker] = CircuitBreaker,
    **kwargs,
) -> CircuitBreaker:
    """按名字获取（或创建）熔断器；同名共享，首次创建决定类型。"""
    with _REGISTRY_LOCK:
        if name not in _BREAKERS:
            _BREAKERS[name] = breaker_cls(name, **kwargs)
        return _BREAKERS[name]


def breaker(name: str, breaker_cls: Type[CircuitBreaker] = CircuitBreaker, **kwargs):
    """装饰器：@breaker("llm", failure_threshold=5)"""
    def decorator(func):
        b = get_breaker(name, breaker_cls=breaker_cls, **kwargs)

        @wraps(func)
        def wrapper(*args, **kw):
            return b.call(func, *args, **kw)

        wrapper.breaker = b
        return wrapper
    return decorator


def all_breaker_status() -> Dict[str, str]:
    with _REGISTRY_LOCK:
        return {n: b.state.value for n, b in _BREAKERS.items()}


def reset_all_breakers():
    with _REGISTRY_LOCK:
        for b in _BREAKERS.values():
            b.reset()