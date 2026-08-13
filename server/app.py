from pathlib import Path

import json as _json
import time

from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from server import config
from server.asr import AudioChunk, transcribe
from server.feedback import generate_feedback
from server.metrics import GazeSample
from server.session import (
    assert_no_media,
    build_summary,
    pick_question,
    save_questionnaire,
    save_summary,
)

app = FastAPI(title="L0 Foundation")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # 本连接内累积凝视样本。L0 汇总实际走 HTTP(见 /transcribe)，
    # 此 WS 通道为 L2 实时 HUD 预留（设计文档 §5）。
    gaze_samples: list[GazeSample] = []
    ws.state.gaze_samples = gaze_samples
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "hello":
                await ws.send_json({"type": "echo", "text": "hi"})
            elif mtype == "start":
                question = pick_question()
                ws.state.question = question
                await ws.send_json({"type": "question", "text": question})
            elif mtype == "gaze":
                gaze_samples.append(GazeSample(t=msg["t"], looking=bool(msg["looking"])))
            else:
                await ws.send_json({"type": "echo", "text": msg.get("text", "")})
    except WebSocketDisconnect:
        return


@app.get("/config")
async def get_config():
    """前端拉取需要的配置。GAZE_ON_CAMERA_DEG 单一真源在 server/config.py，前端不再硬编码。"""
    return {"gaze_on_camera_deg": config.GAZE_ON_CAMERA_DEG}


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile,
    session_id: str = Form(...),
    question: str = Form(""),
    question_index: int = Form(0),        # 一题一份记录（Task 1）
    gaze: str = Form("[]"),
    rag: str = Form(""),                  # per-question 覆盖："on"/"off"/""(空→全局)（Task 2）
):
    raw = await audio.read()
    if not raw:
        return {"error": "收到空音频（前端未录到数据）"}
    try:
        # 阻塞调用放线程池，避免 stall 事件循环（两个被试同时到不互相阻塞）
        result = await run_in_threadpool(
            transcribe, iter([AudioChunk(data=raw, t_start=0.0)]))
    except Exception as e:  # 返回 JSON 而非 500，便于前端显示真实原因
        print(f"[/transcribe] 转录失败: {e}")
        return {"error": str(e)}
    del raw  # 音频转完即丢，不落盘

    gaze_samples = [GazeSample(t=g["t"], looking=bool(g["looking"]))
                    for g in _json.loads(gaze)]
    rag_enabled = None if rag == "" else (rag == "on")   # 空 → 回退全局
    effective_rag = config.RAG_ENABLED if rag_enabled is None else rag_enabled
    summary = build_summary(
        session_id, question, result, gaze_samples,
        question_index=question_index,
        multimodal=config.MULTIMODAL_ENABLED,
        rag=effective_rag,   # 记录实际生效的条件，不是全局默认
    )

    # 生成反馈：失败绝不能看起来像成功，也绝不能丢掉行为数据（转录/指标照存）。
    t0 = time.perf_counter()
    try:
        text, model_version, chunks = await run_in_threadpool(
            generate_feedback, summary, rag_enabled=rag_enabled, return_chunks=True)
        summary["feedback"] = {
            "status": "ok",
            "text": text,
            "model_version": model_version,
            "knowledge_chunks_used": len(chunks),  # flags.rag 真但 0 块 = 实为 RAG-off，分析要能分辨
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "error": None,
        }
    except Exception as e:
        print(f"[/transcribe] 反馈生成失败: {e}")  # 服务端留痕，绝不静默吞掉
        summary["feedback"] = {
            "status": "failed",
            "text": None,
            "model_version": "",
            "knowledge_chunks_used": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "error": f"{type(e).__name__}: {e}",  # 只存类型+短消息，绝不存完整 traceback
        }

    path = save_summary(summary)  # 无论 ok/failed 都落库，让被试能继续
    assert_no_media()  # 落库后立即自检红线（反馈是派生文本，不碰红线）
    return {"summary": summary, "saved_to": path.name}


@app.post("/questionnaire")
async def questionnaire_endpoint(
    session_id: str = Form(...),
    question_index: int = Form(...),
    q1: int = Form(..., ge=1, le=7),
    q2: int = Form(..., ge=1, le=7),
    q3: int = Form(..., ge=1, le=7),
    q4: int = Form(..., ge=1, le=7),
    q5: int = Form(..., ge=1, le=7),
    q6: int = Form(..., ge=1, le=7),
    q7: int = Form(..., ge=1, le=7),
    comment: str = Form(""),
):
    responses = {
        "items": {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "q5": q5,
            "q6": q6,
            "q7": q7,
        },
        "comment": comment,
    }
    try:
        path = save_questionnaire(session_id, question_index, responses)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"saved_to": path.name}


# 静态前端挂在最后，避免盖过 API 路由
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
