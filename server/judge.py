"""LLM-as-Judge（离线批处理）：读 results/*.json 里已存的反馈，按五维 rubric(1–5) 打分。

⚠ 这是离线评估工具，不接入 /transcribe、不参与被试 session。研究数据采集完后，
   在 results/*.json 上批量跑，分数写到独立的 scores/ 层，绝不回写 results。

设计（仿 feedback.py）：
- 客户端懒加载；build_judge_prompt 纯函数；中文注释。
- 裁判对实验条件【盲】：build_judge_prompt 只含 题目+回答+被评反馈，不含 flags/知识块/指标/
  "rag"/"multimodal"（RAG 开/关的反馈结构本就一致，裁判天然分不出，正是要的效果）。
- 裁判 = 生成模型（gemini-3.6-flash，原因见 config.py）：self-preference 由专家校准检验，
  故 judge_model_version 每条都记录、跑完要核对一致性。
- 防御式解析：坏 JSON / 缺维度 / 分数越界一律【抛错】，绝不填默认分（解析失败=裁判失败，必须可见）。
"""
from __future__ import annotations

import json
from pathlib import Path

from server import config

SCORES_DIR = Path(__file__).resolve().parent.parent / "scores"

_DIMENSIONS = ("accuracy", "specificity", "actionability", "coverage", "overall_usefulness")

# rubric v2 在系统指令里。要求 JSON only（无散文、无 markdown 围栏），每条 rationale 一句话。
_JUDGE_SYSTEM = """You are a strict, calibrated evaluator of interview-coaching feedback. You are given an interview question, a candidate's answer, and a piece of written feedback about that answer. Score ONLY the feedback, on five dimensions, each an integer from 1 to 5.

1. accuracy — Is the feedback's judgement of the answer correct? Judge correctness FIRST. Feedback that praises a weak answer (e.g. calls a poor answer "solid" or "strong") scores 1 on accuracy, however positive or fluent it sounds. 3 = broadly correct with some misjudgement; 5 = its assessment of the answer is fully correct.

2. specificity — Is the feedback tied to this specific answer, or generic platitudes that would fit any answer? 1 = generic ("be more specific", "add detail") with nothing anchored to what was actually said; 5 = clearly references the actual content of this answer.

3. actionability — Can the candidate actually act on it; is there a concrete next step? 1 = only says what is wrong with no way forward; 5 = a concrete, doable next step.

4. coverage — Did the feedback address what it should for the aspects of the answer it engages with? Do NOT penalise the feedback for not commenting on delivery if it contains no delivery section — some feedback legitimately has no delivery information available. Score coverage relative to what the feedback could address, not against a fixed content-plus-delivery checklist.

5. overall_usefulness — Overall, how much would this feedback help the candidate improve?

Output JSON only. No prose, no markdown code fence, nothing before or after the JSON. Exactly this shape, with each rationale a single sentence:
{"accuracy":{"score":<1-5>,"rationale":"..."},"specificity":{"score":<1-5>,"rationale":"..."},"actionability":{"score":<1-5>,"rationale":"..."},"coverage":{"score":<1-5>,"rationale":"..."},"overall_usefulness":{"score":<1-5>,"rationale":"..."}}"""


def build_judge_prompt(question: str, answer: str, feedback_text: str) -> str:
    """纯函数：只拼 题目 + 回答 + 被评反馈。绝不含 flags/知识块/指标/条件信息（裁判对条件盲）。"""
    return (
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer (speech-to-text transcript):\n{answer}\n\n"
        f"Feedback to evaluate:\n{feedback_text}\n\n"
        "Score the feedback on the five dimensions and output JSON only."
    )


def _parse_scores(raw: str) -> dict:
    """防御式解析：去围栏/散文 → json.loads → 校验五维齐全且 score 为 1..5 的 int。任何失败都抛。"""
    s = raw.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"裁判输出里找不到 JSON 对象: {raw[:120]!r}")
    obj = json.loads(s[start:end + 1])  # 坏 JSON → JSONDecodeError(ValueError 子类)，上抛
    for dim in _DIMENSIONS:
        if dim not in obj:
            raise ValueError(f"裁判输出缺维度: {dim}")
        entry = obj[dim]
        score = entry.get("score") if isinstance(entry, dict) else None
        # bool 是 int 子类，要排除；分数必须是 1..5 的整数
        if not isinstance(score, int) or isinstance(score, bool) or not (1 <= score <= 5):
            raise ValueError(f"维度 {dim} 的 score 非法（应为 1..5 整数）: {score!r}")
    return obj


_client = None


def _get_client():
    """懒加载 Gemini 客户端。缺 key 明确报错。import 时不建连接。"""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY 未配置：请在 .env 设置")
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _judge_generate(prompt: str) -> tuple[str, str]:
    """调裁判模型，返回 (原始文本, judge_model_version)。空响应抛错（不能当成分数）。"""
    from google.genai import types

    client = _get_client()
    resp = client.models.generate_content(
        model=config.JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_JUDGE_SYSTEM,
            temperature=config.JUDGE_TEMPERATURE,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        raise RuntimeError(f"裁判未返回可用文本 (finish_reason={finish})")
    # 与 feedback.py 同款：记录实际服务版本，回退时标 (unconfirmed)（同模型 → 更要能追溯版本）
    served = getattr(resp, "model_version", None)
    model_version = served or f"{config.JUDGE_MODEL} (unconfirmed)"
    return text, model_version


def judge_feedback(question: str, answer: str, feedback_text: str) -> dict:
    """对一条反馈打分 → {"scores": <五维>, "judge_model_version": <版本>}。"""
    prompt = build_judge_prompt(question, answer, feedback_text)
    raw, model_version = _judge_generate(prompt)
    scores = _parse_scores(raw)  # 解析失败即抛，不填默认分
    return {"scores": scores, "judge_model_version": model_version}


def judge_record(path) -> dict | None:
    """读一条 results 记录并打分。失败/空转录（无可评反馈）→ 返回 None（跳过）。"""
    data = json.loads(Path(path).read_text())
    fb = data.get("feedback") or {}
    if fb.get("status") != "ok" or not fb.get("text"):
        return None  # 反馈生成失败或空转录 → 无反馈可评
    result = judge_feedback(data.get("question", ""), data.get("transcript", ""), fb["text"])
    return {
        "session_id": data.get("session_id"),
        "question_index": data.get("question_index"),
        "scores": result["scores"],
        "judge_model_version": result["judge_model_version"],
    }


def main() -> None:
    """批处理：遍历 results/*.json → judge_record → 写 scores/（独立层，绝不回写 results）。"""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    judged = skipped = failed = 0
    for p in sorted(config.RESULTS_DIR.glob("session_*.json")):
        try:
            res = judge_record(p)
        except Exception as e:  # 单条失败不影响整批；打印可见，绝不静默
            failed += 1
            print(f"[judge] 失败 {p.name}: {type(e).__name__}: {e}")
            continue
        if res is None:
            skipped += 1
            continue
        (SCORES_DIR / p.name).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        judged += 1
    print(f"[judge] 完成：judged={judged} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
