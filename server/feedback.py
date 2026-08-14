"""L1 Step 2 · LLM feedback generation: question + answer + (RAG knowledge) + (nonverbal signals) → Gemini → five-section coaching report.

Design:
- Both ablation switches pass through here (RAG: rag.retrieve returns [] when disabled; multimodal: whether summary has wpm/gaze fields)
- Report structure is identical with RAG on/off; only the evidence basis changes (otherwise RQ2a is confounded)
- Domain knowledge belongs in the knowledge base; feedback style belongs in the prompt (see design decision 3)
- Gemini uses the current google-genai SDK; the client is lazy-loaded
- Single responsibility: generate text only; do not persist data or modify summary
Companion design document: REPORT_STRUCTURE.md
"""
from __future__ import annotations

import time

from server import config, rag

# Coach persona + five-section structure + hard rules + "ASR transcript" disclosure + output requirements.
# ⚠ All feedback "style" lives here and is unaffected by ablation switches (design decision 3)—RAG on/off changes only the evidence basis, not structure.
# ⚠ .format(language=...) requirement: no bare braces may appear anywhere except {language} (pitfall 4).
# ⚠ Do not hard-code any speaking-rate threshold (pitfall 5)—there is no universal standard, so pass rate only as contextual information.
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
    """Convert RAG knowledge hits into a background-reference section. Empty list (RAG off/no hit) → empty string.

    Use them only to ground feedback; instruct the model not to quote verbatim or say "I was given notes".
    """
    if not knowledge:
        return ""
    # Separate with blank lines: each chunk contains its own newline (for example, "[Question: X]\n[Common pitfall] Y").
    # Joining with one \n leaves the next chunk's first line without a bullet and makes chunk boundaries ambiguous to the model.
    joined = "\n\n".join(f"- {k}" for k in knowledge)
    return (
        "\n\nBackground reference material (use it to ground your feedback; "
        "do not quote it verbatim and do not mention that you were given any notes):\n"
        + joined
    )


def _delivery_block(gaze_ratio: float | None, avg_wpm: float | None) -> str:
    """Convert nonverbal signals into delivery evidence. Multimodal off (both absent) → empty string.

    ⚠ Include avg_wpm only when truthy (0.0 is returned for <2 words and must be skipped—pitfall 3).
    ⚠ Do not hard-code a "normal" threshold (pitfall 5)—provide only the value + context and let the model judge.
    """
    lines: list[str] = []
    if avg_wpm:  # Skip both 0.0 and None (pitfall 3)
        lines.append(f"- Speaking rate: about {round(avg_wpm)} words per minute.")
    if gaze_ratio is not None:  # 0.0 is meaningful (never looked at camera), so use is not None
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
    """Build the user prompt (pure function; both gates are directly testable: knowledge chunks = RAG, delivery block = multimodal)."""
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
    """Lazy-load the Gemini client. Missing keys produce a clear error instead of a raw SDK exception. Do not connect at import time."""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY 未配置：请在 .env 设置")
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _generate(prompt: str) -> tuple[str, str]:
    """Call Gemini generation and return (feedback text, modelVersion). This is the only network-touching function."""
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
    # Unsupervised remote use: raise an exception (L1 Step 3 can catch and handle it) rather than return an empty/truncated report as real feedback.
    # Otherwise participants would rate a blank or partial report, silently contaminating the research analysis with bad data.
    if not text:
        # Blocked by safety filters or no candidate → resp.text is None
        raise RuntimeError(f"Gemini 未返回可用文本 (finish_reason={finish})")
    if finish is not None and finish != types.FinishReason.STOP:
        # Non-STOP (for example MAX_TOKENS: 3.x thinking tokens also consume the output budget and may truncate the visible answer)
        raise RuntimeError(f"Gemini 响应不完整、疑似被截断 (finish_reason={finish})")
    # Design decision 5: store the exact model version (required for thesis-method traceability). The SDK field is model_version.
    # Mark fallback as (unconfirmed): never pass the "configured model" off as the "actually served model"—this field exists to catch that mismatch.
    served = getattr(resp, "model_version", None)
    model_version = served or f"{config.GEMINI_MODEL} (unconfirmed)"
    return text, model_version


# Retry only transient failures (5xx/429/network timeouts); do not retry empty/truncated responses (_generate RuntimeError)—
# retries do not change safety blocks, and token-budget truncation will recur. See google.genai.errors for exception types.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _generate_with_retry(prompt: str, *, backoffs: tuple[int, ...] = (2, 4)) -> tuple[str, str]:
    """Add a thin retry layer around _generate. At most 3 attempts (initial + 2 retries), backing off 2s and 4s (total wall time within ~30s)."""
    from google.genai import errors
    import httpx

    max_attempts = len(backoffs) + 1
    for attempt in range(max_attempts):
        try:
            return _generate(prompt)
        except errors.APIError as e:
            # Retry transient status codes only; immediately re-raise others (such as 400/401/403)
            if getattr(e, "code", None) not in _TRANSIENT_STATUS or attempt == max_attempts - 1:
                raise
        except httpx.TransportError:  # Network timeout/connection error: transient
            if attempt == max_attempts - 1:
                raise
        # Note: RuntimeError from _generate (empty/truncated) is outside the two except clauses above → re-raise directly without retrying
        time.sleep(backoffs[attempt])
    raise RuntimeError("unreachable")  # Logically unreachable


def generate_feedback(
    summary: dict,
    *,
    k: int = 3,
    rag_enabled: bool | None = None,
    return_chunks: bool = False,
):
    """Entry point: consume summary → (feedback text, modelVersion)[, retrieved knowledge chunks]. Both ablation switches naturally apply here.

    - RAG (RQ2a): knowledge comes from rag.retrieve(); rag_enabled overrides per question and None falls back to global.
    - Multimodal (RQ2b): gaze/wpm enter the prompt only when summary contains those fields (build_summary omits them when disabled).
    Both switches have one source of truth; feedback.py does not reread global flags (design decisions 1 and 2).

    return_chunks=True → also return the retrieved knowledge-chunk list so the caller can obtain knowledge_chunks_used,
    avoiding a second retrieval in /transcribe (double retrieval is wasteful and may be inconsistent).
    """
    question = summary.get("question", "")
    answer = (summary.get("transcript") or "").strip()
    if not answer:  # Empty transcript: do not call the LLM, saving quota
        # This path called no model → return an empty model version; otherwise persisted records cannot distinguish it from a real model call.
        msg = ("(No answer was transcribed, so no feedback could be generated.)", "")
        return (*msg, []) if return_chunks else msg

    knowledge = rag.retrieve(question, answer, k=k, enabled=rag_enabled)  # [] when RAG is off (RQ2a)
    prompt = build_prompt(
        question,
        answer,
        knowledge,
        gaze_ratio=summary.get("gaze_on_camera_ratio"),  # None when multimodal is off (RQ2b)
        avg_wpm=summary.get("avg_wpm"),
    )
    text, model_version = _generate_with_retry(prompt)  # Automatically retry transient failures
    return (text, model_version, knowledge) if return_chunks else (text, model_version)
