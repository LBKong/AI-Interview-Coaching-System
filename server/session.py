"""会话：出题 + 汇总统计量 + 落库。Step 5 先做出题，Step 6 补汇总/落库。"""
from __future__ import annotations

import random

from server import config


def pick_question(index: int | None = None) -> str:
    """从手写题库选一题。index 给定则确定性返回（测试/复现用）。"""
    if index is not None:
        return config.QUESTIONS[index]
    return random.choice(config.QUESTIONS)
