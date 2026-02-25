from typing import Callable, Optional, Any
from abc import ABC


class AgentMiddleware(ABC):
    """Agent 中间件基类 - 支持两种扩展方式：
    1. 传入 callable：before_run / after_run
    2. 继承重写：on_before_run / on_after_run
    """

    def __init__(
            self,
            before_run: Optional[Callable[..., Any]] = None,
            after_run: Optional[Callable[..., Any]] = None
    ):
        self.before_run = before_run
        self.after_run = after_run

    # ========== 执行入口（Agent 调用） ==========

    def execute_before(self, input_text: str, **kwargs) -> Optional[str]:
        """执行前置逻辑，可返回修改后的 input_text"""
        # 1. 子类钩子
        modified = self.on_before_run(input_text, **kwargs)
        if modified is not None:
            input_text = modified
        # 2. 传入的 callable
        if self.before_run:
            ret = self.before_run(input_text, **kwargs)
            if ret is not None:
                input_text = ret
        return input_text

    def execute_after(self, input_text: str, result: str, **kwargs) -> Optional[str]:
        """执行后置逻辑，可返回修改后的 result"""
        # 1. 子类钩子
        modified = self.on_after_run(input_text, result, **kwargs)
        if modified is not None:
            result = modified
        # 2. 传入的 callable
        if self.after_run:
            ret = self.after_run(input_text, result, **kwargs)
            if ret is not None:
                result = ret
        return result

    # ========== 子类可重写的钩子方法 ==========

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        """前置钩子，子类重写。返回 None 表示不修改，返回 str 表示替换 input_text"""
        return None

    def on_after_run(self, input_text: str, result: str, **kwargs) -> Optional[str]:
        """后置钩子，子类重写。返回 None 表示不修改，返回 str 表示替换 result"""
        return None

    # ========== 可选：异常处理钩子 ==========

    def on_error(self, input_text: str, error: Exception, **kwargs) -> Optional[str]:
        """异常钩子，run 出错时调用。返回 str 可作为降级回复"""
        return None

    # ========== 可选：工具方法 ==========

    def get_name(self) -> str:
        """返回中间件名称，便于日志和调试"""
        return self.__class__.__name__

    def is_enabled(self, **kwargs) -> bool:
        """是否启用，子类可重写实现条件启用"""
        return True