from typing import Optional
from transformers import pipeline

from hello_agents.core.AgentMiddleware import AgentMiddleware


class ToxicBertMiddleware(AgentMiddleware):
    """基于 unitary/toxic-bert 的毒性检测中间件

    在 Agent 执行前检测输入文本毒性，若超过阈值则拦截并返回安全提示。
    依赖: pip install transformers torch
    """

    def __init__(
            self,
            model_id: str = "unitary/toxic-bert",
            toxicity_threshold: float = 0.5,
            block_message: str = "检测到输入内容可能包含不当言论，请修改后重试。",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.toxicity_threshold = toxicity_threshold
        self.block_message = block_message
        self._classifier = None

    def _get_classifier(self):
        """延迟加载模型"""
        if self._classifier is None:
            self._classifier = pipeline(
                "text-classification",
                model=self.model_id,
                top_k=None  # 返回所有标签分数
            )
        return self._classifier

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        """前置毒性检测：若检测到毒性则返回 block_message 拦截，否则返回 None 放行"""
        if not input_text or not input_text.strip():
            return None

        try:
            classifier = self._get_classifier()
            results = classifier(input_text[:512], top_k=None)[0]  # 限制长度，避免超长

            # 取各毒性标签的最高分
            max_toxic_score = 0.0
            for item in results:
                label = item.get("label", "").lower()
                score = item.get("score", 0.0)
                if any(t in label for t in ["toxic", "obscene", "threat", "insult", "hate"]):
                    max_toxic_score = max(max_toxic_score, score)

            if max_toxic_score >= self.toxicity_threshold:
                return self.block_message  # 拦截：返回此字符串作为 Agent 的“回复”

            return None  # 放行：不修改输入
        except Exception as e:
            print(f"[ToxicBertMiddleware] 检测异常: {e}，放行输入")
            return None