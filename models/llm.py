# models/llm.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from config import ZHIPU_API_KEY, ZHIPU_API_BASE, DEFAULT_MODEL, TOKEN_TRACK_ENABLED
from utils.circuit_breaker import get_breaker, SlowCallCircuitBreaker


class CircuitBreakerChatOpenAI(ChatOpenAI):
    """带熔断的 ChatOpenAI。"""

    def _cb(self):
        return get_breaker(
            "llm_api",
            breaker_cls=SlowCallCircuitBreaker,
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=2,
            slow_call_threshold=15.0,
            slow_call_rate=0.5,
        )

    def invoke(self, input, config=None, **kwargs):
        return self._cb().call(super().invoke, input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._cb().acall(super().ainvoke, input, config=config, **kwargs)


def get_llm(model: str = None, temperature: float = 0.2):
    """
    获取智谱 AI 大模型实例（带熔断 + Token 统计）。
    """
    callbacks = []
    if TOKEN_TRACK_ENABLED:
        from utils.token_tracker import get_callback
        callbacks.append(get_callback())   # 修正：append 不返回值，要分开写

    return CircuitBreakerChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        openai_api_key=ZHIPU_API_KEY,
        openai_api_base=ZHIPU_API_BASE,
        callbacks=callbacks,
        timeout=60,
        max_retries=0,   # 交给熔断器统一管理
    )