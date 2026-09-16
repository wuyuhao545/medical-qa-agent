# utils/file_watcher.py
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Set, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DocumentUpdateHandler(FileSystemEventHandler):
    """
    文件监控处理器：
    - 监听文件创建 / 修改事件；
    - 使用 (file_path, mtime, size) 作为唯一键，确保同一文件修改后能再次触发；
    - 缓存使用 LRU 策略，限制最大条目数；
    - 延迟 2 秒触发，确保文件写入完成；
    - 维护活跃 Timer 集合，支持 shutdown 时统一取消。
    """

    def __init__(
        self,
        on_updated: Callable[[List[Path]], None],
        processed_cache: Optional[Set[str]] = None,   # 兼容旧参数
        max_cache_size: int = 1000,
        debounce_seconds: float = 2.0,
    ):
        self.on_updated = on_updated
        self._max_cache_size = max_cache_size
        self._debounce = debounce_seconds

        # 文件指纹缓存（LRU）
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

        # 活跃 Timer 集合
        self._timers: Set[threading.Timer] = set()
        self._timers_lock = threading.Lock()

        # 关机协调
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        # 正在执行回调的数量（用于 shutdown 时等待）
        self._active_callbacks = 0
        self._callback_cond = threading.Condition()

    # ---------- watchdog 事件 ----------
    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    # ---------- 内部处理 ----------
    def _handle(self, file_path: str):
        # 1. 关机后不再接受新任务
        with self._shutdown_lock:
            if self._shutdown:
                return

        # 2. 计算文件指纹
        try:
            stat = os.stat(file_path)
            key = (file_path, stat.st_mtime, stat.st_size)
        except OSError:
            return

        # 3. LRU 缓存去重
        with self._cache_lock:
            if key in self._cache:
                return
            self._cache[key] = True
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        # 4. 创建并登记 Timer
        timer = threading.Timer(self._debounce, self._trigger_update, args=[[file_path]])
        timer.daemon = True

        with self._timers_lock:
            # 二次检查：可能在创建 Timer 期间程序已关机
            with self._shutdown_lock:
                if self._shutdown:
                    timer.cancel()
                    return
            self._timers.add(timer)

        timer.start()

    def _trigger_update(self, paths: List[str]):
        """Timer 到期执行；执行前后维护活跃集合与回调计数。"""
        # 从集合中移除自己（Timer 已触发，无需再 cancel）
        current = threading.current_thread()
        with self._timers_lock:
            self._timers.discard(current)

        # 关机后即使触发也不执行回调
        with self._shutdown_lock:
            if self._shutdown:
                return

        # 标记"正在执行回调"
        with self._callback_cond:
            self._active_callbacks += 1
        try:
            valid = [Path(p) for p in paths if Path(p).exists()]
            if valid:
                try:
                    self.on_updated(valid)
                except Exception as e:
                    print(f"❌ 文件更新回调失败: {e}")
        finally:
            with self._callback_cond:
                self._active_callbacks -= 1
                self._callback_cond.notify_all()

    # ---------- 关机接口 ----------
    def shutdown(self, wait: bool = True, timeout: float = 10.0):
        """
        优雅关机：
        1. 置 shutdown 标志，拒绝新任务
        2. 取消所有未触发的 Timer
        3. 等待正在执行的回调完成（可选）
        """
        # 1. 置标志
        with self._shutdown_lock:
            self._shutdown = True

        # 2. 取消所有未触发的 Timer
        with self._timers_lock:
            timers = list(self._timers)
            self._timers.clear()

        for t in timers:
            t.cancel()

        # 3. 等待正在执行的回调
        if wait:
            import time
            deadline = time.time() + timeout
            with self._callback_cond:
                while self._active_callbacks > 0:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        print(f"⚠️ 关机超时，仍有 {self._active_callbacks} 个回调在跑")
                        break
                    self._callback_cond.wait(timeout=remaining)


class WatcherHandle:
    """
    把 Observer 和 Handler 打包，暴露统一的 start / stop 接口。
    """
    def __init__(self, observer: Observer, handler: DocumentUpdateHandler):
        self.observer = observer
        self.handler = handler
        self._stopped = False
        self._lock = threading.Lock()

    def stop(self, wait: bool = True):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        # 1. 先停 watchdog，不再产生新事件
        self.observer.stop()
        # 2. 再取消 handler 里所有 pending timer + 等待回调结束
        self.handler.shutdown(wait=wait)
        # 3. 等 observer 线程退出
        self.observer.join(timeout=5.0)

    def join(self, timeout: float = 5.0):
        self.observer.join(timeout=timeout)


def start_file_watcher(
    watch_dir: str,
    on_updated: Callable,
    processed_cache: Optional[Set[str]] = None,
    max_cache_size: int = 1000,
    debounce_seconds: float = 2.0,
) -> WatcherHandle:
    """
    启动文件监控，返回 WatcherHandle。
    调用方用 handle.stop() 优雅关闭。
    """
    handler = DocumentUpdateHandler(
        on_updated=on_updated,
        processed_cache=processed_cache,
        max_cache_size=max_cache_size,
        debounce_seconds=debounce_seconds,
    )
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=True)
    observer.start()
    return WatcherHandle(observer, handler)