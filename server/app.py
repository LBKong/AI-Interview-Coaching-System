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
    # Accumulate gaze samples within this connection. L0 aggregation actually uses HTTP (see /transcribe);
    # this WS channel is reserved for the L2 real-time HUD (design document §5).
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
    """Return frontend configuration. GAZE_ON_CAMERA_DEG has a single source of truth in server/config.py and is no longer hard-coded in the frontend."""
    return {"gaze_on_camera_deg": config.GAZE_ON_CAMERA_DEG}


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile,
    session_id: str = Form(...),
    question: str = Form(""),
    question_index: int = Form(0),        # One record per question (Task 1)
    gaze: str = Form("[]"),
    rag: str = Form(""),                  # Per-question override: "on"/"off"/"" (empty → global) (Task 2)
):
    raw = await audio.read()
    if not raw:
        return {"error": "收到空音频（前端未录到数据）"}
    try:
        # Put the blocking call in a thread pool to avoid stalling the event loop (two simultaneous participants do not block each other)
        result = await run_in_threadpool(
            transcribe, iter([AudioChunk(data=raw, t_start=0.0)]))
    except Exception as e:  # Return JSON rather than a 500 so the frontend can show the actual reason
        print(f"[/transcribe] 转录失败: {e}")
        return {"error": str(e)}
    del raw  # Discard audio immediately after transcription; never write it to disk

    gaze_samples = [GazeSample(t=g["t"], looking=bool(g["looking"]))
                    for g in _json.loads(gaze)]
    rag_enabled = None if rag == "" else (rag == "on")   # Empty → fall back to global setting
    effective_rag = config.RAG_ENABLED if rag_enabled is None else rag_enabled
    summary = build_summary(
        session_id, question, result, gaze_samples,
        question_index=question_index,
        multimodal=config.MULTIMODAL_ENABLED,
        rag=effective_rag,   # Record the condition actually applied, not the global default
    )

    # Generate feedback: failure must never look like success or discard behavioural data (the transcript/metrics are still stored).
    t0 = time.perf_counter()
    try:
        text, model_version, chunks = await run_in_threadpool(
            generate_feedback, summary, rag_enabled=rag_enabled, return_chunks=True)
        summary["feedback"] = {
            "status": "ok",
            "text": text,
            "model_version": model_version,
            "knowledge_chunks_used": len(chunks),  # flags.rag true but 0 chunks = effectively RAG-off; analysis must distinguish this
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "error": None,
        }
    except Exception as e:
        print(f"[/transcribe] 反馈生成失败: {e}")  # Preserve a server-side trace; never swallow the failure silently
        summary["feedback"] = {
            "status": "failed",
            "text": None,
            "model_version": "",
            "knowledge_chunks_used": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "error": f"{type(e).__name__}: {e}",  # Store only the type + short message, never a full traceback
        }

    path = save_summary(summary)  # Persist both ok and failed outcomes so the participant can continue
    assert_no_media()  # Immediately enforce the hard-line check after persistence (feedback is derived text and does not violate it)
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


# Mount the static frontend last so it does not shadow API routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
