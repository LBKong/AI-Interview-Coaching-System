"""L1 Step 2 · LLM 生成反馈：题目 + 回答 + (RAG 知识) + (非语言信号) → Gemini → 五段式教练报告。

设计：
- 两个消融开关从这里穿过（RAG: rag.retrieve 关掉返回 []；多模态: summary 有没有 wpm/gaze 字段）
- 报告结构在 RAG 开/关下完全一致，只有内容依据变（否则 RQ2a 被污染）
- 领域知识进知识库、反馈风格进提示词（见设计决定③）
- Gemini 用新版 google-genai SDK；客户端懒加载
- 职责单一：只生成文本，不落库、不改 summary
配套设计文档：REPORT_STRUCTURE.md
"""
from __future__ import annotations

import time

from server import config, rag

# 教练人设 + 五段结构 + 硬规则 + "ASR 转录"声明 + 输出要求。
# ⚠ 反馈"风格"全在这里、不受任何消融开关影响（设计决定③）——RAG 开/关只改"内容依据"，不改结构。
# ⚠ .format(language=...) 要求：除 {language} 外，全文不得有别的裸大括号（坑④）。
# ⚠ 不写死任何语速阈值（坑⑤）——语速没有普适标准，只当情境信息交给模型。
_SYSTEM_INSTRUCTION = """You are an experienced interview coach giving written feedback on one answer to one interview question. This feedback is the only thing the candidate receives, so it must be honest, specific, and easy to act on.

The answer you are given is an automatic speech-to-text transcript of the candidate speaking aloud. Judge only what they said. Never comment on transcription quality, punctuation, capitalisation, or recognition errors — those are artefacts of the tool, not the candidate's behaviour.

Write the feedback in {language}. Produce exactly the following sections, in this fixed order, each under its own heading written exactly as shown. Do not prefix any heading with a number or a list marker (no '1.', no '-', no '*'). Keep each heading as a markdown heading exactly as written above. The order is the priority: the most important judgement comes first.

1. Did you answer the question?
   Always present, always first. One or two sentences giving a direct verdict on whether the answer actually addresses what the question was assessing. A fluent, confident answer that misses the point still misses the point — say so plainly.

2. What worked
   One or two genuine strengths, each pointing to something the candidate actually said — not generic praise. This section is NOT mandatory: if the answer has no real strength, do not invent one. You may acknowledge something real but minor (a suitable choice of example, being candid about the outcome) without endorsing the answer's quality, or you may omit this section entirely. Never describe a weak answer as strong.

3. What to improve
   One or two concrete, actionable points. Say what to do differently next time, not merely what was wrong.

4. Delivery
   Include this section ONLY when delivery signals (speaking rate and/or on-camera gaze) are provided in the input below. When they are provided, always say something — either flag what is notable, or state in one line that nothing stood out. Every number must be tied to what it may communicate; never report a bare number. There is no universally correct speaking rate, so only remark on rate when it is clearly extreme, and always explain what it might signal rather than scoring it against a fixed target. When no delivery signals are provided, omit this section completely.

5. Next time
   A single sentence restating the single most important action from the sections above. Do not introduce a new point here.

Hard rules:
- Across "What worked" and "What to improve" combined, give AT MOST three substantive points in total. More than that and the candidate remembers nothing.
- Never output any score, rating, grade, percentage, or number that judges the candidate. You are evaluating the answer to help them improve — you are not scoring the person.
- Length is not quality: keep the whole report to roughly 150 to 220 words. A short, precise report beats a long, padded one.
- Ground your content in sound interviewing knowledge. If background reference material is provided below, use it to inform your points, but do not quote it verbatim and do not mention that you were given any notes.
"""


def _knowledge_block(knowledge: list[str]) -> str:
    """RAG 命中的知识块 → 「背景参考」段。空列表（RAG 关/未命中）→ 空串。

    只用来 ground 反馈；提示模型别逐字引用、别说"我被给了笔记"。
    """
    if not knowledge:
        return ""
    # 用空行分隔：每个块自身含换行（如 "[Question: X]\n[Common pitfall] Y"），
    # 单个 \n 拼接会让下一块的首行没有项目符号、块边界对模型变模糊。
    joined = "\n\n".join(f"- {k}" for k in knowledge)
    return (
        "\n\nBackground reference material (use it to ground your feedback; "
        "do not quote it verbatim and do not mention that you were given any notes):\n"
        + joined
    )


def _delivery_block(gaze_ratio: float | None, avg_wpm: float | None) -> str:
    """非语言信号 → delivery 素材。多模态关（两个都缺）→ 空串。

    ⚠ avg_wpm 为真才写（0.0 是 <2 词时的返回值，要跳过 —— 坑③）。
    ⚠ 不写死"多少算正常"的阈值（坑⑤）——只给数值 + 情境，判断交给模型。
    """
    lines: list[str] = []
    if avg_wpm:  # 0.0 / None 都跳过（坑③）
        lines.append(f"- Speaking rate: about {round(avg_wpm)} words per minute.")
    if gaze_ratio is not None:  # 0.0 有意义（完全没看镜头），用 is not None
        lines.append(f"- Looking at the camera: about {round(gaze_ratio * 100)}% of the answer.")
    if not lines:
        return ""
    return (
        "\n\nDelivery signals measured during the answer (context only — content matters "
        "first; comment on these only if clearly notable, and always say what they might "
        "communicate rather than judging them against a fixed standard):\n"
        + "\n".join(lines)
    )


def build_prompt(
    question: str,
    answer: str,
    knowledge: list[str],
    *,
    gaze_ratio: float | None = None,
    avg_wpm: float | None = None,
) -> str:
    """拼用户提示（纯函数，两个门控一眼可测：知识块 = RAG，delivery 块 = 多模态）。"""
    prompt = (
        f"Question: {question}\n\n"
        f"Answer (automatic speech-to-text transcript): {answer}"
    )
    prompt += _knowledge_block(knowledge)
    prompt += _delivery_block(gaze_ratio, avg_wpm)
    prompt += "\n\nWrite the coaching feedback now, following the required section structure and rules."
    return prompt


_client = None


def _get_client():
    """懒加载 Gemini 客户端。缺 key 给明确报错，而非 SDK 原始异常。import 时不建连接。"""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY 未配置：请在 .env 设置")
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _generate(prompt: str) -> tuple[str, str]:
    """调 Gemini 生成，返回 (反馈文本, modelVersion)。唯一碰网络的地方。"""
    from google.genai import types

    client = _get_client()
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION.format(language=config.FEEDBACK_LANGUAGE),
            temperature=config.FEEDBACK_TEMPERATURE,
        ),
    )
    text = (resp.text or "").strip()
    finish = resp.candidates[0].finish_reason if resp.candidates else None
    # 远程无人监督：宁可抛异常（L1 Step 3 能捕获兜底），也不能把空/截断的报告当真反馈返回。
    # 否则被试会对一份空白或半截报告打分，坏数据混进研究分析而没人察觉。
    if not text:
        # 被安全过滤拦截或无候选 → resp.text 为 None
        raise RuntimeError(f"Gemini 未返回可用文本 (finish_reason={finish})")
    if finish is not None and finish != types.FinishReason.STOP:
        # 非 STOP（如 MAX_TOKENS：3.x 思考 token 也占输出预算，可能把可见回答截断）
        raise RuntimeError(f"Gemini 响应不完整、疑似被截断 (finish_reason={finish})")
    # 设计决定⑤：存确切模型版本（论文方法论要追溯）。SDK 字段是 model_version。
    # 回退时标 (unconfirmed)：别把"配置的模型"冒充成"实际服务的模型"——存这个字段就是为了抓这种错配。
    served = getattr(resp, "model_version", None)
    model_version = served or f"{config.GEMINI_MODEL} (unconfirmed)"
    return text, model_version


# 瞬时故障（5xx/429/网络超时）才重试；空/截断（_generate 抛的 RuntimeError）不重试——
# 安全拦截不会因重试改变，截断是 token 预算问题会复发。异常类型见 google.genai.errors。
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _generate_with_retry(prompt: str, *, backoffs: tuple[int, ...] = (2, 4)) -> tuple[str, str]:
    """对 _generate 加薄重试层。最多 3 次（首次 + 2 次重试），退避 2s、4s（总墙钟 ~30s 内）。"""
    from google.genai import errors
    import httpx

    max_attempts = len(backoffs) + 1
    for attempt in range(max_attempts):
        try:
            return _generate(prompt)
        except errors.APIError as e:
            # 只重试瞬时状态码；其余（如 400/401/403）立即上抛
            if getattr(e, "code", None) not in _TRANSIENT_STATUS or attempt == max_attempts - 1:
                raise
        except httpx.TransportError:  # 网络超时/连接错误：瞬时
            if attempt == max_attempts - 1:
                raise
        # 注意：_generate 抛的 RuntimeError（空/截断）不在上面两个 except 内 → 直接上抛，不重试
        time.sleep(backoffs[attempt])
    raise RuntimeError("unreachable")  # 逻辑上到不了


def generate_feedback(
    summary: dict,
    *,
    k: int = 3,
    rag_enabled: bool | None = None,
    return_chunks: bool = False,
):
    """入口：吃 summary → (反馈文本, modelVersion)[, 检索到的知识块]。两个消融开关在这里自然生效。

    - RAG（RQ2a）：knowledge 来自 rag.retrieve()，per-question 用 rag_enabled 覆盖，None 时回退全局。
    - 多模态（RQ2b）：gaze/wpm 只在 summary 带这两个字段时才进提示词（build_summary 关掉时本就不写）。
    两个开关都只有单一真源，feedback.py 不再重读全局 flag（设计决定①②）。

    return_chunks=True → 额外返回检索到的知识块列表，供调用方拿 knowledge_chunks_used，
    避免 /transcribe 里再检索一遍（双重检索既浪费又可能不一致）。
    """
    question = summary.get("question", "")
    answer = (summary.get("transcript") or "").strip()
    if not answer:  # 空转录：不调 LLM，省额度
        # 这条路径没调过任何模型 → 模型版本返回空串；否则落库后与"真跑过模型"的记录无法区分。
        msg = ("(No answer was transcribed, so no feedback could be generated.)", "")
        return (*msg, []) if return_chunks else msg

    knowledge = rag.retrieve(question, answer, k=k, enabled=rag_enabled)  # [] 当 RAG 关（RQ2a）
    prompt = build_prompt(
        question,
        answer,
        knowledge,
        gaze_ratio=summary.get("gaze_on_camera_ratio"),  # None 当多模态关（RQ2b）
        avg_wpm=summary.get("avg_wpm"),
    )
    text, model_version = _generate_with_retry(prompt)  # 瞬时故障自动重试
    return (text, model_version, knowledge) if return_chunks else (text, model_version)
