from typing import Optional

from hello_agents.core.AgentMiddleware import AgentMiddleware


class LoggingMiddleware(AgentMiddleware):
    """日志中间件"""

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        print(f"[{self.get_name()}] 输入: {input_text[:50]}...")
        return None  # 不修改输入

    def on_after_run(self, input_text: str, result: str, **kwargs) -> Optional[str]:
        print(f"[{self.get_name()}] 输出长度: {len(result)}")
        return None


class InputSanitizeMiddleware(AgentMiddleware):
    """输入清洗中间件"""

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        return input_text.strip()  # 返回修改后的输入


class RateLimitMiddleware(AgentMiddleware):
    """限流中间件 - 条件启用"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._count = 0
        self._max_calls = 10

    def is_enabled(self, **kwargs) -> bool:
        return self._count < self._max_calls

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        self._count += 1
        return None